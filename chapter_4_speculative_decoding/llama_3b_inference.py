"""
Standalone Llama-3.2-3B inference engine with fine-grained PyTorch Profiling and Visualizer integration.

Instruments the 3B model with the exact profiling categories as Chapter 2:
1. Layer Sub-operators:
   - Embedding, RMSNorm_Attn, Q_Linear, K_Linear, V_Linear, RoPE, KV_Cache_Update
   - Attn_Compute, O_Linear, RMSNorm_FFN, FFN_Gate_Up_Linear, FFN_SiLU_Mul, FFN_Down_Linear
   - RMSNorm_Final, LM_Head, Sampling
2. Multi-Shard Safetensors loading for 3B checkpoint.
3. PyTorch Profiler traces with CPU/GPU timeline extraction.
4. Generates token_metrics_3b.json and profile_dashboard_3b.html via profile_visualizer.py.
"""

import os
import sys
import glob
import math
import time
import json
import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from transformers import AutoTokenizer

# Import visualizer helpers locally from Chapter 4
try:
    from profile_visualizer import (
        extract_single_token_metric,
        extract_timeline_from_trace,
        render_terminal_dashboard,
        save_json_metrics,
        generate_html_dashboard,
    )
except ImportError:
    from chapter_4_speculative_decoding.profile_visualizer import (
        extract_single_token_metric,
        extract_timeline_from_trace,
        render_terminal_dashboard,
        save_json_metrics,
        generate_html_dashboard,
    )


# -----------------------------------------------------------------------------
# 1. Model Configuration
# -----------------------------------------------------------------------------

@dataclass
class ModelArgs:
    dim: int = 3072
    n_layers: int = 28
    n_heads: int = 24
    n_kv_heads: int = 8
    vocab_size: int = 128256
    hidden_dim: int = 8192
    head_dim: int = 128
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_seq_len: int = 8192
    tie_word_embeddings: bool = True
    rope_scaling: Optional[dict] = None

    @classmethod
    def llama_3_2_3b(cls) -> "ModelArgs":
        return cls(
            dim=3072,
            n_layers=28,
            n_heads=24,
            n_kv_heads=8,
            vocab_size=128256,
            hidden_dim=8192,
            head_dim=128,
            norm_eps=1e-5,
            rope_theta=500000.0,
            max_seq_len=8192,
            tie_word_embeddings=True,
            rope_scaling={
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed.to(input_dtype)


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    theta: float = 500000.0,
    rope_scaling: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
        factor = rope_scaling.get("factor", 32.0)
        low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
        old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2.0 * math.pi / inv_freq
        inv_freq_scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1.0 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq
        is_medium_freq = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
        inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_scaled)

    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos.to(dtype=x.dtype)) + (rotate_half(x) * sin.to(dtype=x.dtype))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seqlen, n_kv_heads * n_rep, head_dim)
    )


# -----------------------------------------------------------------------------
# 2. Key-Value Cache Buffer
# -----------------------------------------------------------------------------

class KVCache:
    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
    ):
        self.max_seq_len = max_seq_len
        self.k = torch.zeros(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v = torch.zeros(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

    def update(
        self,
        layer_idx: int,
        start_pos: int,
        xk: torch.Tensor,
        xv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_input_tokens, _, _ = xk.shape
        end_pos = start_pos + num_input_tokens
        if end_pos > self.max_seq_len:
            raise ValueError(f"Context length ({end_pos}) exceeds max_seq_len ({self.max_seq_len}).")

        self.k[layer_idx, :bsz, start_pos:end_pos] = xk
        self.v[layer_idx, :bsz, start_pos:end_pos] = xv
        return self.k[layer_idx, :bsz, :end_pos], self.v[layer_idx, :bsz, :end_pos]

    def get_memory_mb(self) -> float:
        total_bytes = (self.k.numel() + self.v.numel()) * self.k.element_size()
        return total_bytes / (1024.0 * 1024.0)


# -----------------------------------------------------------------------------
# 3. Profiled Architecture Modules
# -----------------------------------------------------------------------------

class ProfiledAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_idx: int = 0,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        bsz, num_input_tokens, _ = x.shape

        with torch.profiler.record_function("Q_Linear"):
            xq = self.q_proj(x).view(bsz, num_input_tokens, self.n_heads, self.head_dim)
        with torch.profiler.record_function("K_Linear"):
            xk = self.k_proj(x).view(bsz, num_input_tokens, self.n_kv_heads, self.head_dim)
        with torch.profiler.record_function("V_Linear"):
            xv = self.v_proj(x).view(bsz, num_input_tokens, self.n_kv_heads, self.head_dim)

        with torch.profiler.record_function("RoPE"):
            xq = apply_rotary_emb(xq, cos, sin)
            xk = apply_rotary_emb(xk, cos, sin)

        if kv_cache is not None:
            with torch.profiler.record_function("KV_Cache_Update"):
                keys, values = kv_cache.update(layer_idx, start_pos, xk, xv)
        else:
            keys, values = xk, xv

        with torch.profiler.record_function("Attn_Compute"):
            keys = repeat_kv(keys, self.n_rep)
            values = repeat_kv(values, self.n_rep)

            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)

            total_keys = keys.shape[2]
            if num_input_tokens > 1 and total_keys > num_input_tokens:
                q_idx = torch.arange(num_input_tokens, device=xq.device)[:, None] + (total_keys - num_input_tokens)
                k_idx = torch.arange(total_keys, device=xq.device)[None, :]
                attn_mask = (k_idx <= q_idx).unsqueeze(0).unsqueeze(0)
                output = F.scaled_dot_product_attention(xq, keys, values, attn_mask=attn_mask)
            elif num_input_tokens > 1:
                output = F.scaled_dot_product_attention(xq, keys, values, is_causal=True)
            else:
                output = F.scaled_dot_product_attention(xq, keys, values, is_causal=False)

            output = output.transpose(1, 2).contiguous().view(bsz, num_input_tokens, -1)

        with torch.profiler.record_function("O_Linear"):
            out = self.o_proj(output)
        return out


class ProfiledFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.profiler.record_function("FFN_Gate_Up_Linear"):
            with torch.profiler.record_function("Gate_Linear"):
                g = self.gate_proj(x)
            with torch.profiler.record_function("Up_Linear"):
                u = self.up_proj(x)

        with torch.profiler.record_function("FFN_SiLU_Mul"):
            act = F.silu(g) * u

        with torch.profiler.record_function("FFN_Down_Linear"):
            out = self.down_proj(act)
        return out


class ProfiledTransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, args: ModelArgs):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.self_attn = ProfiledAttention(args)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = ProfiledFeedForward(args.dim, args.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        with torch.profiler.record_function("RMSNorm_Attn"):
            norm_x1 = self.input_layernorm(x)
        x = x + self.self_attn(norm_x1, cos, sin, layer_idx=self.layer_idx, start_pos=start_pos, kv_cache=kv_cache)

        with torch.profiler.record_function("RMSNorm_FFN"):
            norm_x2 = self.post_attention_layernorm(x)
        x = x + self.mlp(norm_x2)
        return x


class ProfiledTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([ProfiledTransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        if args.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_rope_freqs(
            dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            theta=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        bsz, num_input_tokens = input_ids.shape
        with torch.profiler.record_function("Embedding"):
            h = self.embed_tokens(input_ids)

        cos = self.cos_cached[start_pos : start_pos + num_input_tokens].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[start_pos : start_pos + num_input_tokens].unsqueeze(0).unsqueeze(2)

        for layer in self.layers:
            h = layer(h, cos, sin, start_pos=start_pos, kv_cache=kv_cache)

        with torch.profiler.record_function("RMSNorm_Final"):
            h = self.norm(h)

        with torch.profiler.record_function("LM_Head"):
            if return_all_logits:
                logits = self.lm_head(h)
            else:
                logits = self.lm_head(h[:, [-1], :])
        return logits


# -----------------------------------------------------------------------------
# 4. Multi-Shard Safetensors Loading
# -----------------------------------------------------------------------------

def resolve_model_files(model_path_or_repo: str) -> List[str]:
    if os.path.isfile(model_path_or_repo) and model_path_or_repo.endswith(".safetensors"):
        return [model_path_or_repo]

    if os.path.isdir(model_path_or_repo):
        files = sorted(glob.glob(os.path.join(model_path_or_repo, "*.safetensors")))
        if files:
            return files

    hf_hub_cache = os.path.expanduser("~/.cache/huggingface/hub")
    repo_sanitized = "models--" + model_path_or_repo.replace("/", "--")
    cand_dir = os.path.join(hf_hub_cache, repo_sanitized, "snapshots")
    if os.path.exists(cand_dir):
        snapshots = sorted(glob.glob(os.path.join(cand_dir, "*")))
        if snapshots:
            files = sorted(glob.glob(os.path.join(snapshots[-1], "*.safetensors")))
            if files:
                return files

    from huggingface_hub import snapshot_download
    print(f"[*] Downloading snapshot for {model_path_or_repo}...")
    snapshot_dir = snapshot_download(model_path_or_repo, allow_patterns=["*.safetensors", "*.json"])
    return sorted(glob.glob(os.path.join(snapshot_dir, "*.safetensors")))


def load_hf_safetensors(
    model: ProfiledTransformer,
    model_path_or_repo: str = "unsloth/Llama-3.2-3B-Instruct",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> ProfiledTransformer:
    import gc
    files = resolve_model_files(model_path_or_repo)
    print(f"[*] Loading weights from {len(files)} shard(s) (streaming to conserve RAM)...")

    for idx, shard in enumerate(files, 1):
        print(f"    - Loading shard {idx}/{len(files)}: {os.path.basename(shard)}...")
        raw_state = safetensors.torch.load_file(shard, device=device)
        converted = {
            (k[len("model."):] if k.startswith("model.") else k): v.to(dtype=dtype)
            for k, v in raw_state.items()
        }
        del raw_state
        model.load_state_dict(converted, strict=False)
        del converted
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"[✓] Weights successfully mapped and loaded from {model_path_or_repo}.")
    return model


# -----------------------------------------------------------------------------
# 5. Token Sampling & Generation with Sampled Profiler
# -----------------------------------------------------------------------------

def sample_next_token(logits: torch.Tensor, temperature: float = 0.0, top_k: Optional[int] = None) -> int:
    with torch.profiler.record_function("Sampling"):
        last_logits = logits[:, -1, :]
        if temperature == 0.0:
            return torch.argmax(last_logits, dim=-1).item()

        scaled = last_logits / temperature
        if top_k is not None:
            v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
            scaled[scaled < v[:, [-1]]] = -float("inf")
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()


@torch.inference_mode()
def generate_profiled(
    model: ProfiledTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    profile_interval: int = 25,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    device: str = "cuda",
    profile_output_dir: str = "profile_results",
    ignore_eos: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]

    stop_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id else set()
    for tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)

    kv_cache = KVCache(
        n_layers=model.args.n_layers,
        max_batch_size=1,
        max_seq_len=min(model.args.max_seq_len, prompt_len + max_new_tokens + 64),
        n_kv_heads=model.args.n_kv_heads,
        head_dim=model.args.head_dim,
        dtype=next(model.parameters()).dtype,
        device=device,
    )

    traces_dir = os.path.join(profile_output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    print(f"\n--- Prompt ({prompt_len} tokens) --- \n{prompt}")
    print(f"\n--- Model Response (streaming tokens with KV Cache) ---")

    generated_ids: List[int] = []
    token_records: List[Dict[str, Any]] = []

    def execute_step(
        tokens: torch.Tensor,
        start_pos: int,
        step_idx: int,
        total_context_len: int,
        is_sample: bool,
    ) -> int:
        if device == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        prof_ctx = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            )
            if is_sample
            else nullcontext()
        )

        with prof_ctx as prof:
            with (torch.profiler.record_function(f"token_{step_idx}") if is_sample else nullcontext()):
                logits = model(tokens, start_pos=start_pos, kv_cache=kv_cache)
                next_tok = sample_next_token(logits, temperature=temperature, top_k=top_k)

        if device == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        native_latency_ms = (t_end - t_start) * 1000.0

        tok_str = tokenizer.decode([next_tok], clean_up_tokenization_spaces=False)
        print(tok_str, end="", flush=True)

        if is_sample:
            trace_file = os.path.join(traces_dir, f"step_{step_idx}_trace.json")
            prof.export_chrome_trace(trace_file)

            step_record = extract_single_token_metric(prof, step_idx, next_tok, tok_str)
            timeline = extract_timeline_from_trace(
                trace_path=trace_file,
                step_idx=step_idx,
                token_id=next_tok,
                token_text=tok_str,
                seq_len=total_context_len,
            )
            step_record["timeline"] = timeline
            if "three_metrics" in timeline and timeline["three_metrics"]:
                step_record["three_metrics"] = timeline["three_metrics"]
            step_record["total_latency_ms"] = timeline.get("three_metrics", {}).get("total_latency_ms", round(native_latency_ms, 2))
            step_record["is_sampled"] = True
            step_record["phase"] = "prefill" if step_idx == 0 else "decode"
            token_records.append(step_record)
        else:
            token_records.append({
                "step": step_idx,
                "token_id": next_tok,
                "token_text": tok_str,
                "total_latency_ms": round(native_latency_ms, 2),
                "is_sampled": False,
                "phase": "decode",
                "breakdown": None,
                "three_metrics": None,
                "timeline": None,
            })

        return next_tok

    # Prefill
    next_tok = execute_step(prompt_ids, start_pos=0, step_idx=0, total_context_len=prompt_len, is_sample=True)
    generated_ids.append(next_tok)

    # Decode
    curr_tok_tensor = torch.tensor([[next_tok]], device=device)
    for step in range(1, max_new_tokens):
        if not ignore_eos and next_tok in stop_ids:
            break

        start_pos = prompt_len + step - 1
        total_context_len = start_pos + 1
        is_sample = (step % profile_interval == 0) or (step == max_new_tokens - 1)

        next_tok = execute_step(
            curr_tok_tensor,
            start_pos=start_pos,
            step_idx=step,
            total_context_len=total_context_len,
            is_sample=is_sample,
        )
        curr_tok_tensor[0, 0] = next_tok
        generated_ids.append(next_tok)

    print("\n-----------------------------------------")
    full_text = tokenizer.decode(generated_ids, clean_up_tokenization_spaces=False)
    return full_text, token_records


# -----------------------------------------------------------------------------
# 6. Main CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Llama-3.2-3B Inference Engine with Fine-Grained PyTorch Profiling.")
    parser.add_argument("--prompt", type=str, default="Write a quicksort implementation in Python with comments.", help="Input prompt")
    parser.add_argument("--model_path", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="HF model ID or local directory")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--profile_interval", type=int, default=25, help="Sampling interval for PyTorch Profiler")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ignore_eos", action="store_true", default=False)
    args = parser.parse_args()

    print("================================================================================")
    print(" 🦙 Tiny-vLLM Chapter 4: Llama-3.2-3B Profiled Inference Engine")
    print("================================================================================")
    print(f"[*] Device: {args.device} ({torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'})")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model_args = ModelArgs.llama_3_2_3b()
    torch.set_default_dtype(dtype)
    with torch.device(args.device):
        model = ProfiledTransformer(model_args)
    torch.set_default_dtype(torch.float32)
    model.eval()

    load_hf_safetensors(model, model_path_or_repo=args.model_path, device=args.device, dtype=dtype)

    # Tokenizer
    files = resolve_model_files(args.model_path)
    tok_dir = os.path.dirname(files[0]) if files else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=os.path.exists(tok_dir))

    # 4. Model Warmup (Model, cuBLAS prompt shape heuristics, sampling kernel, and PyTorch Profiler / CUPTI context)
    if args.device == "cuda":
        print("[*] Performing comprehensive model & profiler warmup to eliminate CUDA context and CUPTI overhead...")
        with torch.no_grad():
            warm_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(args.device)
            warm_seq_len = warm_ids.shape[-1]

            dummy_cache = KVCache(
                model.args.n_layers, 1, warm_seq_len + 16,
                model.args.n_kv_heads, model.args.head_dim,
                dtype=dtype, device=args.device,
            )

            # Un-profiled warmup passes for cuBLAS heuristics and sampling kernel
            for _ in range(2):
                dummy_logits = model(warm_ids, start_pos=0, kv_cache=dummy_cache)
                _ = sample_next_token(dummy_logits, temperature=args.temperature, top_k=args.top_k)
            torch.cuda.synchronize()

            # Profiled warmup pass to initialize CUPTI and Kineto activity buffers
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ):
                dummy_logits = model(warm_ids, start_pos=0, kv_cache=dummy_cache)
                _ = sample_next_token(dummy_logits, temperature=args.temperature, top_k=args.top_k)
            torch.cuda.synchronize()
        print("[*] Warmup complete.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "profile_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[*] Starting profiled generation (max_new_tokens={args.max_new_tokens}, profile_interval={args.profile_interval})...")
    _, token_records = generate_profiled(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        profile_interval=args.profile_interval,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        profile_output_dir=output_dir,
        ignore_eos=args.ignore_eos,
    )

    if token_records:
        sampled_records = [r for r in token_records if r.get("is_sampled", True) and r.get("breakdown")]
        print(f"\n[*] Rendering dashboard for {len(token_records)} tokens ({len(sampled_records)} sampled checkpoints)...")
        render_terminal_dashboard(token_records, prompt=args.prompt)

        timeline_records = {r["step"]: r["timeline"] for r in token_records if "timeline" in r and r["timeline"]}
        json_file = os.path.join(output_dir, "token_metrics_3b.json")
        save_json_metrics(token_records, prompt=args.prompt, output_file=json_file, timeline_records=timeline_records)

        # Generate HTML Dashboard
        dashboard_html = os.path.join(output_dir, "profile_dashboard_3b.html")
        generate_html_dashboard(token_records, prompt=args.prompt, output_file=dashboard_html, timeline_records=timeline_records)
        print(f"\n[✓] Profiling complete.")
        print(f"[*] Token Metrics JSON saved: {json_file}")
        print(f"[*] Interactive HTML Dashboard: {dashboard_html}")


if __name__ == "__main__":
    main()
