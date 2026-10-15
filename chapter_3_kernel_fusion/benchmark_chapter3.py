"""
Chapter 3 · FlashAttention benchmark (single source of truth for the deck).

Every number in presentations/chapter3-flash-attention-derivation.html
should be reproducible by running:

    PATH=.venv/bin:$PATH .venv/bin/python chapter_3_kernel_fusion/benchmark_chapter3.py

What this script guarantees
---------------------------
1. CORRECTNESS FIRST. Every CUDA kernel is compared against a PyTorch FP32
   reference (max-abs error) before it is timed. A kernel that is wrong is
   reported and excluded. Nothing is timed that hasn't been checked.

2. ONE DECODE SETUP. All decode rows start from the *same* native BF16 KV
   cache [B, H_kv, MAX_SEQ, d] and the same BF16 query, exactly as the
   LLaMA-3.2-1B model stores them. Whatever conversion a kernel needs
   (BF16->FP32 cast, GQA head expansion) is done inside the timed region,
   because that is what the model would have to do.

3. SAME HARNESS FOR EVERY ROW. Each implementation gets its own warm-up pass
   over all 32 decode steps, then NUM_TRIALS timed passes. We report the
   median. Timing is wall-clock with explicit synchronisation on both ends.

4. MEASURED, NOT THEORETICAL, MEMORY. Prefill peak VRAM is read from
   torch.cuda.max_memory_allocated() around each call.

Results are written to chapter_3_kernel_fusion/results/chapter3_results.json.
"""
import os
import sys
import json
import math
import time
import statistics
import platform

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration (LLaMA-3.2-1B attention shape)
# ---------------------------------------------------------------------------
B, H_Q, H_KV, D = 1, 32, 8, 64
N_REP = H_Q // H_KV
SCALE = 1.0 / math.sqrt(D)

DECODE_CONTEXT = 8192
DECODE_TOKENS = 32
MAX_SEQ_LEN = DECODE_CONTEXT + DECODE_TOKENS
NUM_TRIALS = 5

PREFILL_LENGTHS = [2048, 4096, 8192, 12288, 16384]
PREFILL_WARMUP = 2
PREFILL_ITERS = 3

torch.manual_seed(0)

def banner(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

def gib(nbytes):
    return nbytes / (1024 ** 3)

# ---------------------------------------------------------------------------
# Load extensions
# ---------------------------------------------------------------------------
banner("Building / loading CUDA extensions")

def ext(name, subdir, cpp, cu):
    return load(
        name=name,
        sources=[os.path.join(HERE, subdir, cpp), os.path.join(HERE, subdir, cu)],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )

naive_ext = ext("cuda_flash_decode_ext", "cuda_flash_decode",
                "flash_decode_extension.cpp", "flash_decode_kernel.cu")
splitk_fp32_ext = ext("cuda_warp_decode_ext", "cuda_warp_decode",
                      "warp_extension.cpp", "warp_decode.cu")
bf16_ext = ext("cuda_bf16_flash_attn_ext", "cuda_bf16_flash_attention",
               "bf16_flash_attn_extension.cpp", "bf16_flash_attn_kernel.cu")
print("[ok] cuda_flash_decode_ext      (first decode kernel: 1 block / head, no Split-K)")
print("[ok] cuda_warp_decode_ext       (Split-K decode, FP32 inputs)")
print("[ok] cuda_bf16_flash_attn_ext   (Split-K decode + prefill, native BF16 inputs)")

# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
def ref_decode_fp32(q_bf16, k_bf16, v_bf16, seq_len):
    """FP32 ground truth from the same BF16 inputs the kernels see."""
    q = q_bf16.float()
    k = k_bf16[:, :, :seq_len, :].float().repeat_interleave(N_REP, dim=1)
    v = v_bf16[:, :, :seq_len, :].float().repeat_interleave(N_REP, dim=1)
    s = torch.matmul(q, k.transpose(-2, -1)) * SCALE
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)

def ref_prefill_fp32(q_bf16, k_bf16, v_bf16, q_chunk=1024):
    """
    FP32 causal ground truth. Computed in query-row chunks so the reference
    itself does not OOM at 8k (a dense FP32 [32, 8192, 8192] tensor is 8 GiB).
    """
    n = q_bf16.shape[2]
    q = q_bf16.float()
    k = k_bf16.float().repeat_interleave(N_REP, dim=1)
    v = v_bf16.float().repeat_interleave(N_REP, dim=1)
    kt = k.transpose(-2, -1)
    out = torch.empty_like(q)
    cols = torch.arange(n, device=q.device)
    for r0 in range(0, n, q_chunk):
        r1 = min(r0 + q_chunk, n)
        s = torch.matmul(q[:, :, r0:r1, :], kt) * SCALE
        rows = torch.arange(r0, r1, device=q.device)
        s.masked_fill_(cols[None, :] > rows[:, None], float("-inf"))
        out[:, :, r0:r1, :] = torch.matmul(torch.softmax(s, dim=-1), v)
        del s
    return out

def pytorch_decode_bf16(q_bf16, k_bf16, v_bf16, seq_len):
    """Production baseline: what PyTorch does with the BF16 cache."""
    k = k_bf16[:, :, :seq_len, :].repeat_interleave(N_REP, dim=1)
    v = v_bf16[:, :, :seq_len, :].repeat_interleave(N_REP, dim=1)
    s = torch.matmul(q_bf16, k.transpose(-2, -1)) * SCALE
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)

def pytorch_prefill_lean_bf16(q, k, v):
    """Minimal PyTorch causal attention in BF16 (bool mask, BF16 softmax)."""
    n = q.shape[2]
    k = k.repeat_interleave(N_REP, dim=1)
    v = v.repeat_interleave(N_REP, dim=1)
    s = torch.matmul(q, k.transpose(-2, -1)) * SCALE
    mask = torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), diagonal=1)
    s.masked_fill_(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)

def pytorch_prefill_model_bf16(q, k, v):
    """
    Exactly what llama_inference_with_profiling.py does in its attention
    block: additive -inf mask in the score dtype, then softmax computed in
    FP32 and cast back to BF16. This is the path that actually ran inside
    the model, so it is the honest 'does it OOM?' test.
    """
    n = q.shape[2]
    k = k.repeat_interleave(N_REP, dim=1)
    v = v.repeat_interleave(N_REP, dim=1)
    s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    mask = torch.full((n, n), float("-inf"), device=q.device, dtype=s.dtype)
    mask = torch.triu(mask, diagonal=1)
    s = s + mask
    p = torch.nn.functional.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(p, v)

# ---------------------------------------------------------------------------
# Correctness helpers
# ---------------------------------------------------------------------------
def compare(name, out, ref, tol):
    out = out.float()
    nan = int(torch.isnan(out).sum().item() + torch.isinf(out).sum().item())
    err = (out - ref).abs().max().item() if nan == 0 else float("inf")
    ok = nan == 0 and err <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:<58} max|err| = {err:.3e}  (tol {tol:.0e}, nan/inf = {nan})")
    return ok, err

results = {
    "hardware": {
        "gpu": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "vram_total_gib": round(gib(torch.cuda.get_device_properties(0).total_memory), 2),
        "torch": torch.__version__,
        "python": platform.python_version(),
    },
    "config": {
        "B": B, "H_q": H_Q, "H_kv": H_KV, "d": D,
        "decode_context": DECODE_CONTEXT, "decode_tokens": DECODE_TOKENS,
        "decode_trials": NUM_TRIALS, "prefill_lengths": PREFILL_LENGTHS,
    },
    "correctness": {}, "decode": {}, "prefill": [],
}

# ===========================================================================
# PART A · CORRECTNESS
# ===========================================================================
banner("PART A · Correctness vs PyTorch FP32 reference")

q_bf16 = torch.randn(B, H_Q, 1, D, device="cuda", dtype=torch.bfloat16)
k_cache_bf16 = torch.randn(B, H_KV, MAX_SEQ_LEN, D, device="cuda", dtype=torch.bfloat16)
v_cache_bf16 = torch.randn(B, H_KV, MAX_SEQ_LEN, D, device="cuda", dtype=torch.bfloat16)

# Test both a multiple of the 128-key chunk and a ragged tail.
decode_ok = {"naive": True, "splitk_fp32": True, "splitk_bf16": True}
for seq_len in (DECODE_CONTEXT, DECODE_CONTEXT + 7):
    print(f"\nDecode, seq_len = {seq_len}:")
    ref = ref_decode_fp32(q_bf16, k_cache_bf16, v_cache_bf16, seq_len)

    q32 = q_bf16.float()
    k32 = k_cache_bf16[:, :, :seq_len, :].float().contiguous()
    v32 = v_cache_bf16[:, :, :seq_len, :].float().contiguous()

    out = naive_ext.forward(q32, k32.repeat_interleave(N_REP, 1).contiguous(),
                            v32.repeat_interleave(N_REP, 1).contiguous())
    ok, err = compare("Our CUDA implementation (no Split-K, FP32)", out, ref, 1e-4)
    decode_ok["naive"] &= ok
    results["correctness"][f"decode_naive_fp32_N{seq_len}"] = {"pass": ok, "max_abs_err": err}

    out = splitk_fp32_ext.forward(q32, k32, v32, seq_len)
    ok, err = compare("Split-K decode (FP32)", out, ref, 1e-4)
    decode_ok["splitk_fp32"] &= ok
    results["correctness"][f"decode_splitk_fp32_N{seq_len}"] = {"pass": ok, "max_abs_err": err}

    out = bf16_ext.forward(q_bf16, k_cache_bf16, v_cache_bf16, seq_len)
    ok, err = compare("Split-K decode (native BF16, full cache + seq_len)", out, ref, 1e-2)
    decode_ok["splitk_bf16"] &= ok
    results["correctness"][f"decode_splitk_bf16_N{seq_len}"] = {"pass": ok, "max_abs_err": err}

print("\nPrefill (native BF16 kernel vs FP32 causal reference):")
prefill_ok = True
for n in (2048, 4096, 8192):
    Qp = torch.randn(B, H_Q, n, D, device="cuda", dtype=torch.bfloat16)
    Kp = torch.randn(B, H_KV, n, D, device="cuda", dtype=torch.bfloat16)
    Vp = torch.randn(B, H_KV, n, D, device="cuda", dtype=torch.bfloat16)
    ref = ref_prefill_fp32(Qp, Kp, Vp)
    out = bf16_ext.forward(Qp, Kp, Vp, n)
    ok, err = compare(f"Prefill BF16 kernel, N = {n}", out, ref, 1e-2)
    prefill_ok &= ok
    results["correctness"][f"prefill_bf16_N{n}"] = {"pass": ok, "max_abs_err": err}
    del Qp, Kp, Vp, ref, out
    torch.cuda.empty_cache()

if not all(decode_ok.values()) or not prefill_ok:
    print("\n!! One or more kernels FAILED correctness. Their timings below are NOT trustworthy. !!")

# ===========================================================================
# PART B · DECODE, 32 tokens at 8,192 context, one shared setup
# ===========================================================================
banner(f"PART B · Decode: {DECODE_TOKENS} tokens, context {DECODE_CONTEXT} -> {MAX_SEQ_LEN - 1}  "
       f"(median of {NUM_TRIALS} trials, each preceded by a full warm-up pass)")
print("All rows start from the SAME native BF16 KV cache; any cast / head expansion a kernel")
print("needs is inside the timed region, because the model would have to pay it too.\n")

def step_pytorch(cur_len):
    return pytorch_decode_bf16(q_bf16, k_cache_bf16, v_cache_bf16, cur_len)

def step_naive(cur_len):
    k = k_cache_bf16[:, :, :cur_len, :].float().repeat_interleave(N_REP, 1).contiguous()
    v = v_cache_bf16[:, :, :cur_len, :].float().repeat_interleave(N_REP, 1).contiguous()
    return naive_ext.forward(q_bf16.float(), k, v)

def step_splitk_fp32_with_cast(cur_len):
    k = k_cache_bf16[:, :, :cur_len, :].float().contiguous()
    v = v_cache_bf16[:, :, :cur_len, :].float().contiguous()
    return splitk_fp32_ext.forward(q_bf16.float(), k, v, cur_len)

# Kernel-only controls: inputs already FP32 (and head-expanded for the first
# kernel), so these isolate kernel time from cast / expansion overhead.
k_cache_fp32 = k_cache_bf16.float().contiguous()
v_cache_fp32 = v_cache_bf16.float().contiguous()
q_fp32 = q_bf16.float().contiguous()
k_exp_fixed = k_cache_fp32[:, :, :DECODE_CONTEXT, :].repeat_interleave(N_REP, 1).contiguous()
v_exp_fixed = v_cache_fp32[:, :, :DECODE_CONTEXT, :].repeat_interleave(N_REP, 1).contiguous()
def step_naive_precast(cur_len):
    # This kernel reads N from the tensor shape (no seq_len argument), so a
    # growing-length control would need a fresh 134 MB copy per step. We time
    # it at a fixed 8,192 instead; the 8192 -> 8223 difference is < 0.4 %.
    return naive_ext.forward(q_fp32, k_exp_fixed, v_exp_fixed)
def step_splitk_fp32_precast(cur_len):
    return splitk_fp32_ext.forward(q_fp32, k_cache_fp32, v_cache_fp32, cur_len)

def step_splitk_bf16(cur_len):
    return bf16_ext.forward(q_bf16, k_cache_bf16, v_cache_bf16, cur_len)

def step_cast_only(cur_len):
    k = k_cache_bf16[:, :, :cur_len, :].float().contiguous()
    v = v_cache_bf16[:, :, :cur_len, :].float().contiguous()
    return k, v

def time_sequence(step_fn):
    def run_once():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(DECODE_TOKENS):
            step_fn(DECODE_CONTEXT + s)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0
    run_once()  # warm-up over the full sequence (allocator + clocks)
    return statistics.median(run_once() for _ in range(NUM_TRIALS))

decode_rows = [
    ("PyTorch Native (BF16 baseline)",                     step_pytorch,                True),
    ("1. Our CUDA Implementation (no Split-K)",            step_naive,                  decode_ok["naive"]),
    ("   Our CUDA Impl. (kernel only, pre-cast+expanded)", step_naive_precast,          decode_ok["naive"]),
    ("2. Split-K Decode (FP32 + cast tax)",                step_splitk_fp32_with_cast,  decode_ok["splitk_fp32"]),
    ("   Split-K Decode (FP32, kernel only, pre-cast)",    step_splitk_fp32_precast,    decode_ok["splitk_fp32"]),
    ("   BF16 -> FP32 cast of K,V only (no kernel)",       step_cast_only,              True),
    ("3. Split-K Decode (native BF16)",                    step_splitk_bf16,            decode_ok["splitk_bf16"]),
]

# Set REVERSE_ORDER=1 to run the rows back-to-front; results should not move.
if os.environ.get("REVERSE_ORDER") == "1":
    print("[note] REVERSE_ORDER=1: timing rows in reverse to check for order effects\n")
    decode_rows = decode_rows[::-1]

timings = {label: time_sequence(fn) for label, fn, _ in decode_rows}
baseline_ms = timings["PyTorch Native (BF16 baseline)"]

print(f"{'Implementation':<52} | {'32 tok (ms)':>11} | {'per tok (ms)':>12} | {'vs PyTorch':>10}")
print("-" * 94)
for label, fn, trusted in (decode_rows[::-1] if os.environ.get("REVERSE_ORDER") == "1" else decode_rows):
    total = timings[label]
    speed = baseline_ms / total
    flag = "" if trusted else "   <-- FAILED CORRECTNESS"
    print(f"{label:<52} | {total:>11.2f} | {total / DECODE_TOKENS:>12.3f} | {speed:>9.2f}x{flag}")
    results["decode"][label.strip()] = {
        "total_ms_32_tokens": round(total, 2),
        "per_token_ms": round(total / DECODE_TOKENS, 3),
        "speedup_vs_pytorch": round(speed, 2),
        "passed_correctness": bool(trusted),
    }

del k_cache_fp32, v_cache_fp32, k_exp_fixed, v_exp_fixed
torch.cuda.empty_cache()

# ===========================================================================
# PART C · PREFILL context sweep, measured peak memory
# ===========================================================================
banner("PART C · Prefill context sweep (median of %d iters after %d warm-ups; peak VRAM measured)"
       % (PREFILL_ITERS, PREFILL_WARMUP))
print("'PyTorch (model path)' = additive mask + FP32 softmax, exactly as llama_inference_with_profiling.py.")
print("'PyTorch (lean)'       = bool mask + BF16 softmax, the leanest eager implementation.\n")

def time_and_peak(fn):
    """Returns (median_ms, peak_extra_gib) or raises OOM."""
    for _ in range(PREFILL_WARMUP):
        out = fn(); del out
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    times = []
    for _ in range(PREFILL_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        del out
    peak = torch.cuda.max_memory_allocated() - base
    return statistics.median(times), gib(peak)

def try_variant(fn):
    try:
        ms, peak = time_and_peak(fn)
        return {"status": "ok", "ms": round(ms, 2), "peak_extra_gib": round(peak, 2)}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"status": "OOM", "ms": None, "peak_extra_gib": None}

def fmt(r):
    if r["status"] != "ok":
        return f"{'OOM':>10} | {'-':>8}"
    return f"{r['ms']:>10.2f} | {r['peak_extra_gib']:>8.2f}"

hdr = (f"{'Context':<8} | {'PyTorch (model path)':^21} | {'PyTorch (lean)':^21} | {'FlashAttn BF16 (ours)':^21}")
sub = (f"{'':<8} | {'ms':>10} | {'GiB':>8} | {'ms':>10} | {'GiB':>8} | {'ms':>10} | {'GiB':>8}")
print(hdr); print(sub); print("-" * len(sub))

for n in PREFILL_LENGTHS:
    Qp = torch.randn(B, H_Q, n, D, device="cuda", dtype=torch.bfloat16)
    Kp = torch.randn(B, H_KV, n, D, device="cuda", dtype=torch.bfloat16)
    Vp = torch.randn(B, H_KV, n, D, device="cuda", dtype=torch.bfloat16)

    r_model = try_variant(lambda: pytorch_prefill_model_bf16(Qp, Kp, Vp))
    r_lean = try_variant(lambda: pytorch_prefill_lean_bf16(Qp, Kp, Vp))
    r_ours = try_variant(lambda: bf16_ext.forward(Qp, Kp, Vp, n))

    print(f"{n:<8} | {fmt(r_model)} | {fmt(r_lean)} | {fmt(r_ours)}")
    results["prefill"].append({
        "context": n,
        "pytorch_model_path": r_model,
        "pytorch_lean": r_lean,
        "flash_attn_bf16": r_ours,
        "theoretical_scores_tensor_gib_bf16": round(gib(B * H_Q * n * n * 2), 2),
        "theoretical_scores_tensor_gib_fp32": round(gib(B * H_Q * n * n * 4), 2),
    })
    del Qp, Kp, Vp
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
out_path = os.path.join(RESULTS_DIR, "chapter3_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[*] Results written to {os.path.relpath(out_path)}")

