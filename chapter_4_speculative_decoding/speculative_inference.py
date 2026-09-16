"""
Self-contained Speculative Decoding Engine: Llama-3.2-3B (Target) + Llama-3.2-1B (Draft)

Clean Systems Implementation:
1. Dual-Model VRAM Management:
   - Target: Llama-3.2-3B (28 layers, dim=3072, head_dim=128)
   - Draft:  Llama-3.2-1B (16 layers, dim=2048, head_dim=64)
   - Both reside concurrently in GPU memory (~8.8 GB total weights).
2. Clean Alignment & Caching Protocol:
   - Both models process prompt tokens during prefill.
   - Target emits initial token x_0.
   - At each cycle, draft generates K tokens starting from x_last.
   - Target evaluates [x_last, d_1, ..., d_K] in a single parallel forward pass (length K+1).
   - Target produces K+1 logits, simultaneously verifying d_1..d_K and producing the bonus token.
   - Emits accepted draft tokens + 1 target token (bonus or correction).
   - Both caches update in-place without memory allocation.
"""

import os
import glob
import math
import time
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from transformers import AutoTokenizer


# -----------------------------------------------------------------------------
# 1. Model Configuration & Architecture Primitives
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

    @classmethod
    def llama_3_2_1b(cls) -> "ModelArgs":
        return cls(
            dim=2048,
            n_layers=16,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=128256,
            hidden_dim=8192,
            head_dim=64,
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
    return (x * cos) + (rotate_half(x) * sin)


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
# 3. Model Architecture Modules
# -----------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.k_proj = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.v_proj = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.o_proj = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        bsz, num_input_tokens, _ = x.shape

        xq = self.q_proj(x).view(bsz, num_input_tokens, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, num_input_tokens, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, num_input_tokens, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_emb(xq, cos, sin)
        xk = apply_rotary_emb(xk, cos, sin)

        if kv_cache is not None:
            keys, values = kv_cache.update(layer_idx, start_pos, xk, xv)
        else:
            keys, values = xk, xv

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        total_keys = keys.shape[2]
        if num_input_tokens > 1 and total_keys > num_input_tokens:
            # Multi-token speculative verification with past KV cache history
            q_idx = torch.arange(num_input_tokens, device=xq.device)[:, None] + (total_keys - num_input_tokens)
            k_idx = torch.arange(total_keys, device=xq.device)[None, :]
            attn_mask = (k_idx <= q_idx).unsqueeze(0).unsqueeze(0)  # (1, 1, Q, K)
            output = F.scaled_dot_product_attention(xq, keys, values, attn_mask=attn_mask)
        elif num_input_tokens > 1:
            # Standard prefill (square Q == K)
            output = F.scaled_dot_product_attention(xq, keys, values, is_causal=True)
        else:
            # Single-token decode (Q=1)
            output = F.scaled_dot_product_attention(xq, keys, values, is_causal=False)

        output = output.transpose(1, 2).contiguous().view(bsz, num_input_tokens, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.self_attn = Attention(args)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = FeedForward(args.dim, args.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), start_pos, cos, sin, kv_cache, self.layer_id)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class LlamaTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_rope_freqs(args.head_dim, args.max_seq_len, args.rope_theta, args.rope_scaling)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        kv_cache: Optional[KVCache] = None,
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        h = self.embed_tokens(tokens)

        cos = self.cos_cached[start_pos : start_pos + seqlen].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[start_pos : start_pos + seqlen].unsqueeze(0).unsqueeze(2)

        for layer in self.layers:
            h = layer(h, start_pos, cos, sin, kv_cache)

        h = self.norm(h)
        if return_all_logits:
            logits = self.lm_head(h)
        else:
            logits = self.lm_head(h[:, [-1], :])
        return logits


# -----------------------------------------------------------------------------
# 4. Weight Loading Helper
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
    files = sorted(glob.glob(os.path.join(snapshot_dir, "*.safetensors")))
    return files


def load_model_safetensors(
    model: LlamaTransformer,
    model_path_or_repo: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> LlamaTransformer:
    files = resolve_model_files(model_path_or_repo)
    print(f"[*] Loading {model_path_or_repo} ({len(files)} shard(s))...")
    combined = {}
    for shard in files:
        state = safetensors.torch.load_file(shard)
        for k, v in state.items():
            key = k[len("model."):] if k.startswith("model.") else k
            combined[key] = v.to(dtype=dtype, device=device)
    model.load_state_dict(combined, strict=False)
    print(f"[✓] {model_path_or_repo} loaded.")
    return model


# -----------------------------------------------------------------------------
# 5. Clean Speculative Decoding Engine
# -----------------------------------------------------------------------------

@torch.inference_mode()
def generate_speculative(
    target_model: LlamaTransformer,
    draft_model: LlamaTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    k: int = 2,
    device: str = "cuda",
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_tokens)
    bsz = 1
    max_buf_len = prompt_len + max_new_tokens + 64

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    target_cache = KVCache(
        n_layers=target_model.args.n_layers,
        max_batch_size=bsz,
        max_seq_len=max_buf_len,
        n_kv_heads=target_model.args.n_kv_heads,
        head_dim=target_model.args.head_dim,
        dtype=dtype,
        device=device,
    )
    draft_cache = KVCache(
        n_layers=draft_model.args.n_layers,
        max_batch_size=bsz,
        max_seq_len=max_buf_len,
        n_kv_heads=draft_model.args.n_kv_heads,
        head_dim=draft_model.args.head_dim,
        dtype=dtype,
        device=device,
    )

    # 1. Prefill Phase on prompt tokens [0 .. prompt_len - 1]
    if device == "cuda":
        torch.cuda.synchronize()
    t_start_prefill = time.perf_counter()

    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    target_prefill_logits = target_model(prompt_tensor, start_pos=0, kv_cache=target_cache)
    _ = draft_model(prompt_tensor, start_pos=0, kv_cache=draft_cache)

    # First token predicted by Target model for position prompt_len
    first_token = torch.argmax(target_prefill_logits[:, -1, :], dim=-1).item()

    if device == "cuda":
        torch.cuda.synchronize()
    prefill_time_ms = (time.perf_counter() - t_start_prefill) * 1000.0

    print(f"[Prefill] {prompt_len} tokens in {prefill_time_ms:.2f} ms | First token: {tokenizer.decode([first_token])!r}")

    generated_ids = [first_token]
    valid_len = prompt_len  # Both caches currently have prompt_len tokens (0 .. prompt_len - 1)
    x_last = first_token    # Token that needs to be evaluated next

    cycle_records = []
    total_proposed = 0
    total_accepted = 0
    cycle_num = 0

    t_decode_start = time.perf_counter()

    while len(generated_ids) < max_new_tokens:
        cycle_num += 1

        # -------------------------------------------------------------
        # Phase A: Draft Model proposes K tokens
        # -------------------------------------------------------------
        if device == "cuda":
            torch.cuda.synchronize()
        t_draft_start = time.perf_counter()

        draft_tokens = []
        curr_draft_in = x_last
        for step in range(k):
            in_t = torch.tensor([[curr_draft_in]], dtype=torch.long, device=device)
            d_logits = draft_model(in_t, start_pos=valid_len + step, kv_cache=draft_cache)
            d_pred = torch.argmax(d_logits[:, -1, :], dim=-1).item()
            draft_tokens.append(d_pred)
            curr_draft_in = d_pred

        if device == "cuda":
            torch.cuda.synchronize()
        draft_time_ms = (time.perf_counter() - t_draft_start) * 1000.0

        # -------------------------------------------------------------
        # Phase B: Target Model evaluates [x_last, d_1, ..., d_K] (K+1 tokens)
        # -------------------------------------------------------------
        if device == "cuda":
            torch.cuda.synchronize()
        t_verify_start = time.perf_counter()

        target_eval_tokens = [x_last] + draft_tokens
        eval_tensor = torch.tensor([target_eval_tokens], dtype=torch.long, device=device)
        target_logits = target_model(eval_tensor, start_pos=valid_len, kv_cache=target_cache, return_all_logits=True)

        if device == "cuda":
            torch.cuda.synchronize()
        verify_time_ms = (time.perf_counter() - t_verify_start) * 1000.0

        # -------------------------------------------------------------
        # Phase C: Verification Logic & Emitted Tokens
        # -------------------------------------------------------------
        # target_logits[0, 0] predicts candidate 0 (d_1)
        # target_logits[0, 1] predicts candidate 1 (d_2)
        # ...
        # target_logits[0, k-1] predicts candidate k-1 (d_K)
        # target_logits[0, k] predicts bonus token!
        n_accepted = 0
        accepted_tokens = []

        for j in range(k):
            t_pred = torch.argmax(target_logits[0, j, :], dim=-1).item()
            if draft_tokens[j] == t_pred:
                n_accepted += 1
                accepted_tokens.append(draft_tokens[j])
            else:
                # Mismatch at candidate j: take Target's prediction
                corrected_token = t_pred
                break
        else:
            # All K matched: bonus token from position k
            corrected_token = torch.argmax(target_logits[0, k, :], dim=-1).item()

        # Total tokens emitted this round: accepted draft proposals + 1 target token
        emitted_this_round = accepted_tokens + [corrected_token]
        generated_ids.extend(emitted_this_round)

        total_proposed += k
        total_accepted += n_accepted

        # -------------------------------------------------------------
        # Phase D: Cache Synchronization for Next Cycle
        # -------------------------------------------------------------
        # In this cycle:
        # - Target model evaluated [x_last, d_1, ..., d_{n_accepted-1}] successfully.
        # - The valid context length advances by 1 (for x_last) + n_accepted (for accepted draft tokens).
        # - The new x_last is corrected_token.
        new_valid_len = valid_len + 1 + n_accepted

        # Synchronize draft cache so its context exactly matches the accepted sequence:
        # Draft cache currently has entries for [x_last, d_1, ...].
        # It needs x_last at valid_len, and any accepted draft tokens up to new_valid_len - 1.
        # Since draft cache already ran [x_last, d_1, ..., d_K], its slots up to valid_len + n_accepted
        # already contain [x_last, d_1, ..., d_{n_accepted-1}].
        # The new draft proposals for the next cycle will start from corrected_token at new_valid_len!
        valid_len = new_valid_len
        x_last = corrected_token

        cycle_latency_ms = draft_time_ms + verify_time_ms
        instant_alpha = n_accepted / k
        cumulative_alpha = total_accepted / total_proposed

        cycle_records.append({
            "cycle": cycle_num,
            "k": k,
            "proposed": k,
            "accepted": n_accepted,
            "emitted_count": len(emitted_this_round),
            "emitted_tokens_text": [tokenizer.decode([t]) for t in emitted_this_round],
            "instant_acceptance_rate": instant_alpha,
            "cumulative_acceptance_rate": cumulative_alpha,
            "draft_time_ms": draft_time_ms,
            "verify_time_ms": verify_time_ms,
            "cycle_latency_ms": cycle_latency_ms,
            "effective_ms_per_token": cycle_latency_ms / len(emitted_this_round),
            "curr_total_tokens": len(generated_ids),
        })

        print(f"  Cycle {cycle_num:02d} | Prop: {k} | Acc: {n_accepted} (+1) | Latency: {cycle_latency_ms:.1f}ms ({cycle_latency_ms/len(emitted_this_round):.1f} ms/tok) | Text: {' '.join([repr(tokenizer.decode([t])) for t in emitted_this_round])}")

        if any(t == tokenizer.eos_token_id for t in emitted_this_round):
            break

    total_decode_time = time.perf_counter() - t_decode_start
    total_tokens_gen = len(generated_ids)
    overall_tok_per_sec = (total_tokens_gen - 1) / total_decode_time if total_decode_time > 0 else 0.0

    summary = {
        "mode": "Speculative (3B Target + 1B Draft)",
        "prompt": prompt,
        "k": k,
        "total_generated": total_tokens_gen,
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "overall_acceptance_rate": total_accepted / total_proposed if total_proposed > 0 else 0.0,
        "prefill_latency_ms": prefill_time_ms,
        "decode_throughput_tok_per_sec": overall_tok_per_sec,
        "avg_ms_per_token": (total_decode_time * 1000.0) / (total_tokens_gen - 1) if total_tokens_gen > 1 else 0.0,
        "target_cache_mb": target_cache.get_memory_mb(),
        "draft_cache_mb": draft_cache.get_memory_mb(),
        "total_cache_mb": target_cache.get_memory_mb() + draft_cache.get_memory_mb(),
    }

    full_text = tokenizer.decode(generated_ids, clean_up_tokenization_spaces=False)
    return full_text, cycle_records, summary


# -----------------------------------------------------------------------------
# 6. Main CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding with Llama-3.2-3B (Target) and Llama-3.2-1B (Draft).")
    parser.add_argument("--prompt", type=str, default="Write a quicksort implementation in Python with comments.", help="Input prompt")
    parser.add_argument("--target_model", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="Target HF repo or path")
    parser.add_argument("--draft_model", type=str, default="unsloth/Llama-3.2-1B-Instruct", help="Draft HF repo or path")
    parser.add_argument("--k", type=int, default=2, help="Lookahead speculative window K (default: 2)")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    print("================================================================================")
    print(" 🚀 Tiny-vLLM Chapter 4: Speculative Decoding (3B Target + 1B Draft)")
    print("================================================================================")
    print(f"[*] Device: {args.device} ({torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'})")
    print(f"[*] Speculative Lookahead K: {args.k}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

    # 1. Load Target Model (3B)
    print("\n[1/3] Loading Target Model (Llama-3.2-3B)...")
    target_args = ModelArgs.llama_3_2_3b()
    target_model = LlamaTransformer(target_args).to(device=args.device, dtype=dtype)
    target_model.eval()
    load_model_safetensors(target_model, args.target_model, device=args.device, dtype=dtype)

    # 2. Load Draft Model (1B)
    print("\n[2/3] Loading Draft Model (Llama-3.2-1B)...")
    draft_args = ModelArgs.llama_3_2_1b()
    draft_model = LlamaTransformer(draft_args).to(device=args.device, dtype=dtype)
    draft_model.eval()
    load_model_safetensors(draft_model, args.draft_model, device=args.device, dtype=dtype)

    # 3. Load Tokenizer
    print("\n[3/3] Loading Shared Tokenizer...")
    t_files = resolve_model_files(args.target_model)
    tok_dir = os.path.dirname(t_files[0]) if t_files else args.target_model
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=os.path.exists(tok_dir))

    # Warmup
    if args.device == "cuda":
        print("[*] Warming up models on GPU...")
        with torch.no_grad():
            dummy = torch.tensor([[1]], device=args.device)
            _ = target_model(dummy, start_pos=0)
            _ = draft_model(dummy, start_pos=0)
            torch.cuda.synchronize()

    print(f"\n[*] Starting Speculative Generation (K={args.k}, max_new_tokens={args.max_new_tokens})...\n")
    full_text, cycle_records, summary = generate_speculative(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        k=args.k,
        device=args.device,
    )

    print("\n---------------- Generated Text ----------------")
    print(full_text)
    print("------------------------------------------------")

    print("\n--- Speculative Performance Summary ---")
    print(f"  Lookahead K:               {summary['k']}")
    print(f"  Overall Acceptance Rate:   {summary['overall_acceptance_rate']*100:.1f}% ({summary['total_accepted']}/{summary['total_proposed']})")
    print(f"  Decode Throughput:         {summary['decode_throughput_tok_per_sec']:.2f} tok/s")
    print(f"  Avg Latency per Token:     {summary['avg_ms_per_token']:.2f} ms/tok")
    print(f"  Target Cache VRAM:         {summary['target_cache_mb']:.2f} MB")
    print(f"  Draft Cache VRAM:          {summary['draft_cache_mb']:.2f} MB")
    print(f"  Total Cache VRAM:          {summary['total_cache_mb']:.2f} MB")
    print("----------------------------------------\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "profile_results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = args.output_file or os.path.join(out_dir, "speculative_metrics.json")

    with open(out_file, "w") as f:
        json.dump({"summary": summary, "cycles": cycle_records}, f, indent=2)
    print(f"[✓] Speculative metrics saved to: {out_file}")


if __name__ == "__main__":
    main()
