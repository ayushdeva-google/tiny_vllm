import os
import math
import time
import torch
from torch.utils.cpp_extension import load

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
curr_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 80)
print("   UNIFIED DECODE BENCHMARK: STANDARDIZED HARNESS ON NVIDIA L4 (58 SMs)")
print("   Model: LLaMA-3.2-1B Config (B=1, H_q=32, H_kv=8, d=64, N=8,192)")
print("=" * 80)

# Load all 4 pre-compiled extensions
print("\n[*] Loading compiled CUDA extensions...")
naive_ext = load(
    name='cuda_flash_decode_ext',
    sources=[f'{curr_dir}/cuda_flash_decode/flash_decode_extension.cpp', f'{curr_dir}/cuda_flash_decode/flash_decode_kernel.cu'],
    verbose=False
)
initial_splitk_ext = load(
    name='cuda_flash_attention_ext',
    sources=[f'{curr_dir}/cuda_flash_attention/flash_attn_extension.cpp', f'{curr_dir}/cuda_flash_attention/flash_attn_kernel.cu'],
    verbose=False
)
warp_ext = load(
    name='cuda_warp_decode_ext',
    sources=[f'{curr_dir}/cuda_warp_decode/warp_extension.cpp', f'{curr_dir}/cuda_warp_decode/warp_decode.cu'],
    verbose=False
)
bf16_ext = load(
    name='cuda_bf16_flash_attn_ext',
    sources=[f'{curr_dir}/cuda_bf16_flash_attention/bf16_flash_attn_extension.cpp', f'{curr_dir}/cuda_bf16_flash_attention/bf16_flash_attn_kernel.cu'],
    verbose=False
)
print("[*] All extensions ready!\n")

B, H_q, H_kv, d = 1, 32, 8, 64
N_CONTEXT = 8192
DECODE_TOKENS = 32
MAX_SEQ_LEN = N_CONTEXT + DECODE_TOKENS
scale = 1.0 / math.sqrt(d)
WARMUP = 15
ITERS = 100

# Allocate test tensors
# 1. Native BF16 tensors
q_bf16 = torch.randn(B, H_q, 1, d, device='cuda', dtype=torch.bfloat16)
k_cache_bf16 = torch.randn(B, H_kv, MAX_SEQ_LEN, d, device='cuda', dtype=torch.bfloat16)
v_cache_bf16 = torch.randn(B, H_kv, MAX_SEQ_LEN, d, device='cuda', dtype=torch.bfloat16)

# 2. FP32 tensors
q_fp32 = q_bf16.to(torch.float32)
k_cache_fp32 = k_cache_bf16.to(torch.float32)
v_cache_fp32 = v_cache_bf16.to(torch.float32)

# 3. Pre-expanded FP32 & BF16 for raw cuBLAS attention (isolated QK + Softmax + PV)
k_expanded_fp32 = k_cache_fp32[:, :, :N_CONTEXT, :].repeat_interleave(H_q // H_kv, dim=1).contiguous()
v_expanded_fp32 = v_cache_fp32[:, :, :N_CONTEXT, :].repeat_interleave(H_q // H_kv, dim=1).contiguous()
k_expanded_bf16 = k_cache_bf16[:, :, :N_CONTEXT, :].repeat_interleave(H_q // H_kv, dim=1).contiguous()
v_expanded_bf16 = v_cache_bf16[:, :, :N_CONTEXT, :].repeat_interleave(H_q // H_kv, dim=1).contiguous()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def measure_kernel(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

# =========================================================================
# Part 1: Single-Step Latency at N = 8,192 (ms / token)
# =========================================================================
print("=" * 80)
print(f"PART 1: SINGLE-STEP DECODE LATENCY AT N = {N_CONTEXT} TOKENS")
print("=" * 80)

# 1. PyTorch cuBLAS (Raw isolated attention, pre-expanded)
# FP32
def pt_cublas_fp32():
    scores = torch.matmul(q_fp32, k_expanded_fp32.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_expanded_fp32)
t_pt_cublas_fp32 = measure_kernel(pt_cublas_fp32)

# BF16
def pt_cublas_bf16():
    scores = torch.matmul(q_bf16, k_expanded_bf16.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_expanded_bf16)
t_pt_cublas_bf16 = measure_kernel(pt_cublas_bf16)

# 2. PyTorch Native GQA Attention (Full forward step with repeat_interleave)
# FP32
def pt_gqa_fp32():
    k = k_cache_fp32[:, :, :N_CONTEXT, :].repeat_interleave(H_q // H_kv, dim=1)
    v = v_cache_fp32[:, :, :N_CONTEXT, :].repeat_interleave(H_q // H_kv, dim=1)
    scores = torch.matmul(q_fp32, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
t_pt_gqa_fp32 = measure_kernel(pt_gqa_fp32)

# BF16
def pt_gqa_bf16():
    k = k_cache_bf16[:, :, :N_CONTEXT, :].repeat_interleave(H_q // H_kv, dim=1)
    v = v_cache_bf16[:, :, :N_CONTEXT, :].repeat_interleave(H_q // H_kv, dim=1)
    scores = torch.matmul(q_bf16, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
t_pt_gqa_bf16 = measure_kernel(pt_gqa_bf16)

# 3. CUDA Naive Online Softmax (No Split-K, 1 block/head)
# Needs expanded K/V (32 heads)
def cuda_naive_decode():
    return naive_ext.forward(q_fp32, k_expanded_fp32, v_expanded_fp32)
t_cuda_naive = measure_kernel(cuda_naive_decode)

# 4. CUDA Initial Split-K Decode (from cuda_flash_attention: 2048 blocks, smem staging, tx==0 serial loop)
# Raw FP32 kernel
def cuda_initial_splitk():
    return initial_splitk_ext.forward(q_fp32, k_cache_fp32, v_cache_fp32, N_CONTEXT)
t_cuda_initial_splitk = measure_kernel(cuda_initial_splitk)

# Initial Split-K WITH Python BF16 -> FP32 cast tax (as originally executed in inference loop)
def cuda_initial_splitk_with_cast():
    k_cast = k_cache_bf16[:, :, :N_CONTEXT, :].to(torch.float32).contiguous()
    v_cast = v_cache_bf16[:, :, :N_CONTEXT, :].to(torch.float32).contiguous()
    return initial_splitk_ext.forward(q_fp32, k_cast, v_cast, N_CONTEXT)
t_cuda_initial_splitk_with_cast = measure_kernel(cuda_initial_splitk_with_cast)

# 5. CUDA Warp-Optimized Split-K Decode (FP32 registers + tree reduction)
def cuda_warp_fp32():
    return warp_ext.forward(q_fp32, k_cache_fp32, v_cache_fp32, N_CONTEXT)
t_cuda_warp_fp32 = measure_kernel(cuda_warp_fp32)

# 6. CUDA Warp-Optimized Native BFloat16 Decode (__nv_bfloat16, zero cast)
def cuda_warp_bf16():
    return bf16_ext.forward(q_bf16, k_cache_bf16, v_cache_bf16, N_CONTEXT)
t_cuda_warp_bf16 = measure_kernel(cuda_warp_bf16)

# Measure isolated Python Cast Overhead
def isolated_cast():
    _ = k_cache_bf16[:, :, :N_CONTEXT, :].to(torch.float32).contiguous()
    _ = v_cache_bf16[:, :, :N_CONTEXT, :].to(torch.float32).contiguous()
t_cast_tax = measure_kernel(isolated_cast)

print(f"{'Implementation':<45} | {'Latency (ms)':<14} | {'Speedup vs PyTorch GQA':<24}")
print("-" * 88)
print(f"{'PyTorch Native cuBLAS (Raw Isolated, FP32)':<45} | {t_pt_cublas_fp32:<14.4f} | {t_pt_gqa_bf16 / t_pt_cublas_fp32:.2f}x")
print(f"{'PyTorch Native cuBLAS (Raw Isolated, BF16)':<45} | {t_pt_cublas_bf16:<14.4f} | {t_pt_gqa_bf16 / t_pt_cublas_bf16:.2f}x")
print(f"{'PyTorch Native GQA Step (FP32)':<45} | {t_pt_gqa_fp32:<14.4f} | {t_pt_gqa_bf16 / t_pt_gqa_fp32:.2f}x")
print(f"{'PyTorch Native GQA Step (BF16, Production Baseline)':<45} | {t_pt_gqa_bf16:<14.4f} | 1.00x (Baseline)")
print("-" * 88)
print(f"{'1. Our CUDA Implementation (No Split-K, 1 blk/head)':<45} | {t_cuda_naive:<14.4f} | {t_pt_gqa_bf16 / t_cuda_naive:.2f}x ({t_cuda_naive / t_pt_gqa_bf16:.2f}x SLOWER)")
print(f"{'2. Initial Split-K (Raw Kernel, tx==0 loop)':<45} | {t_cuda_initial_splitk:<14.4f} | {t_pt_gqa_bf16 / t_cuda_initial_splitk:.2f}x")
print(f"{'   + Python BFloat16 Cast Tax Overhead':<45} | {t_cast_tax:<14.4f} | --")
print(f"{'   = Initial Split-K End-to-End (with Cast)':<45} | {t_cuda_initial_splitk_with_cast:<14.4f} | {t_pt_gqa_bf16 / t_cuda_initial_splitk_with_cast:.2f}x ({t_cuda_initial_splitk_with_cast / t_pt_gqa_bf16:.2f}x SLOWER)")
print(f"{'3. Warp-Optimized Split-K (FP32 registers)':<45} | {t_cuda_warp_fp32:<14.4f} | {t_pt_gqa_bf16 / t_cuda_warp_fp32:.2f}x")
print(f"{'4. Warp-Optimized Native BF16 (__nv_bfloat16)':<45} | {t_cuda_warp_bf16:<14.4f} | {t_pt_gqa_bf16 / t_cuda_warp_bf16:.2f}x FASTER!")


# =========================================================================
# Part 2: 32-Token Decode Sequence Benchmark (N = 8,192 to 8,224)
# =========================================================================
print("\n" + "=" * 80)
print(f"PART 2: 32-TOKEN DECODE SEQUENCE BENCHMARK (N = {N_CONTEXT} -> {MAX_SEQ_LEN})")
print("=" * 80)

# Warmup sequence
for step in range(5):
    cur_len = N_CONTEXT + step
    _ = pt_gqa_bf16()
    _ = cuda_warp_bf16()
torch.cuda.synchronize()

# 1. PyTorch Native BF16 (32 tokens)
pt_32_start = time.perf_counter()
for step in range(DECODE_TOKENS):
    cur_len = N_CONTEXT + step
    k = k_cache_bf16[:, :, :cur_len, :].repeat_interleave(H_q // H_kv, dim=1)
    v = v_cache_bf16[:, :, :cur_len, :].repeat_interleave(H_q // H_kv, dim=1)
    scores = torch.matmul(q_bf16, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    _ = torch.matmul(probs, v)
torch.cuda.synchronize()
pt_32_time = (time.perf_counter() - pt_32_start) * 1000.0

# 2. CUDA Naive Decode (32 tokens)
cuda_naive_32_start = time.perf_counter()
for step in range(DECODE_TOKENS):
    cur_len = N_CONTEXT + step
    k_exp = k_cache_fp32[:, :, :cur_len, :].repeat_interleave(H_q // H_kv, dim=1).contiguous()
    v_exp = v_cache_fp32[:, :, :cur_len, :].repeat_interleave(H_q // H_kv, dim=1).contiguous()
    _ = naive_ext.forward(q_fp32, k_exp, v_exp)
torch.cuda.synchronize()
cuda_naive_32_time = (time.perf_counter() - cuda_naive_32_start) * 1000.0

# 3. CUDA Initial Split-K with Cast (32 tokens)
cuda_splitk_cast_32_start = time.perf_counter()
for step in range(DECODE_TOKENS):
    cur_len = N_CONTEXT + step
    k_cast = k_cache_bf16[:, :, :cur_len, :].to(torch.float32).contiguous()
    v_cast = v_cache_bf16[:, :, :cur_len, :].to(torch.float32).contiguous()
    _ = initial_splitk_ext.forward(q_fp32, k_cast, v_cast, cur_len)
torch.cuda.synchronize()
cuda_splitk_cast_32_time = (time.perf_counter() - cuda_splitk_cast_32_start) * 1000.0

# 4. CUDA Warp-Optimized FP32 (32 tokens)
cuda_warp_fp32_32_start = time.perf_counter()
for step in range(DECODE_TOKENS):
    cur_len = N_CONTEXT + step
    _ = warp_ext.forward(q_fp32, k_cache_fp32, v_cache_fp32, cur_len)
torch.cuda.synchronize()
cuda_warp_fp32_32_time = (time.perf_counter() - cuda_warp_fp32_32_start) * 1000.0

# 5. CUDA Warp-Optimized Native BF16 (32 tokens)
cuda_warp_bf16_32_start = time.perf_counter()
for step in range(DECODE_TOKENS):
    cur_len = N_CONTEXT + step
    _ = bf16_ext.forward(q_bf16, k_cache_bf16, v_cache_bf16, cur_len)
torch.cuda.synchronize()
cuda_warp_bf16_32_time = (time.perf_counter() - cuda_warp_bf16_32_start) * 1000.0

print(f"{'Implementation':<45} | {'Total (32 tok)':<15} | {'Per Token':<12} | {'Speedup':<10}")
print("-" * 88)
print(f"{'PyTorch Native BF16 (Standard Baseline)':<45} | {pt_32_time:<15.2f} ms | {pt_32_time/32:<12.2f} ms | 1.00x")
print(f"{'1. Our CUDA Implementation (No Split-K)':<45} | {cuda_naive_32_time:<15.2f} ms | {cuda_naive_32_time/32:<12.2f} ms | {pt_32_time/cuda_naive_32_time:.2f}x ({cuda_naive_32_time/pt_32_time:.2f}x SLOWER)")
print(f"{'2. Initial Split-K with Cast Tax (End-to-End)':<45} | {cuda_splitk_cast_32_time:<15.2f} ms | {cuda_splitk_cast_32_time/32:<12.2f} ms | {pt_32_time/cuda_splitk_cast_32_time:.2f}x ({cuda_splitk_cast_32_time/pt_32_time:.2f}x SLOWER)")
print(f"{'3. Warp-Optimized Split-K (FP32)':<45} | {cuda_warp_fp32_32_time:<15.2f} ms | {cuda_warp_fp32_32_time/32:<12.2f} ms | {pt_32_time/cuda_warp_fp32_32_time:.2f}x")
print(f"{'4. Warp-Optimized Native BF16 (__nv_bfloat16)':<45} | {cuda_warp_bf16_32_time:<15.2f} ms | {cuda_warp_bf16_32_time/32:<12.2f} ms | {pt_32_time/cuda_warp_bf16_32_time:.2f}x FASTER!")
print("=" * 88)

