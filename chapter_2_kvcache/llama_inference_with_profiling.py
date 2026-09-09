"""
Minimal, standalone Llama-3.2-1B inference engine with KV Cache and sampled PyTorch Profiling.

Core Systems Concepts:
1. Static KV Cache Buffer:
   Pre-allocated directly in GPU VRAM to eliminate memory fragmentation and tensor re-allocations during decoding.
2. Grouped-Query Attention (GQA) Memory Savings:
   Stores only the 8 unrepeated KV heads in the cache (saving 75% VRAM and memory bandwidth vs storing 32 heads).
   Expands to 32 query heads via repeat_kv only during attention computation.
3. Prefill vs. Decode Phase:
   - Prefill (Step 0): Full prompt processed in parallel (num_input_tokens = prompt_len); applies causal mask over (S x S).
   - Decode (Steps 1..N): Feeds strictly 1 token (num_input_tokens = 1); linear projections are O(1) compute;
     causal mask is bypassed because a single query can attend to all past tokens and itself.
4. Clear Variable Terminology:
   - `num_input_tokens`: The length of the token tensor fed into the current forward pass (prompt_len in prefill, 1 in decode).
   - `total_context_len`: The total cumulative sequence length stored in the KV Cache (start_pos + num_input_tokens).
5. Sampled Step Profiling via contextlib.nullcontext:
   Zero code duplication! Profiles step 0 (prefill), every profile_interval steps (decode), and the final step.
   Un-sampled steps run at full native GPU speed using nullcontext().
"""

import os
import glob
import math
import time
import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from transformers import AutoTokenizer

try:
    from profile_visualizer import (
        extract_single_token_metric,
        extract_timeline_from_trace,
        render_terminal_dashboard,
        save_json_metrics,
    )
except ImportError:
    from chapter_2_kvcache.profile_visualizer import (
        extract_single_token_metric,
        extract_timeline_from_trace,
        render_terminal_dashboard,
        save_json_metrics,
    )


# -----------------------------------------------------------------------------
# 1. Model Configuration & Architecture Primitives
# -----------------------------------------------------------------------------

@dataclass
class ModelArgs:
    """Hyperparameters for Llama-3.2-1B-Instruct architecture."""
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128256
    hidden_dim: int = 8192  # intermediate_size in MLP
    head_dim: int = 64      # dim // n_heads
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_seq_len: int = 8192
    tie_word_embeddings: bool = True
    rope_scaling: Optional[dict] = None

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
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Computes variance in float32 for numerical stability, then casts back to input dtype.
    """
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
    """
    Precomputes cosine and sine frequency tables for Rotary Position Embeddings (RoPE).
    Includes wavelength-based frequency scaling for Llama 3 / 3.2.
    """
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

    # HuggingFace layout duplicates frequencies across the second half of head_dim
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies Rotary Position Embedding (RoPE) to tensor x."""
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Broadcasts key/value heads across query groups for Grouped-Query Attention (GQA).
    Input shape:  (bsz, seqlen, n_kv_heads, head_dim)
    Output shape: (bsz, seqlen, n_kv_heads * n_rep, head_dim)
    """
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
    """
    Pre-allocated static buffer for Key-Value caching across all layers.
    
    Key Systems Optimization:
    - Pre-allocates memory directly on GPU (device="cuda") to avoid CUDA reallocations during generation.
    - Stores only unrepeated 8 KV heads (saving 75% VRAM and memory bandwidth vs caching 32 heads).
    - Uses in-place slice updates: self.k[layer_idx, :, start_pos:end_pos] = xk.

    Buffer shape per layer:
      k_cache: (max_batch_size, max_seq_len, n_kv_heads, head_dim)
      v_cache: (max_batch_size, max_seq_len, n_kv_heads, head_dim)
    """
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

        # Pre-allocate contiguous GPU buffers directly in VRAM
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
        """
        Inserts new keys and values into the cache at [start_pos : start_pos + num_input_tokens],
        and returns views of all cached keys and values from 0 up to (start_pos + num_input_tokens).
        """
        bsz, num_input_tokens, _, _ = xk.shape
        end_pos = start_pos + num_input_tokens
        if end_pos > self.max_seq_len:
            raise ValueError(f"Context length ({end_pos}) exceeds pre-allocated max_seq_len ({self.max_seq_len}).")

        # Zero-allocation in-place slice update
        self.k[layer_idx, :bsz, start_pos:end_pos] = xk
        self.v[layer_idx, :bsz, start_pos:end_pos] = xv

        # Return history up to current position
        return self.k[layer_idx, :bsz, :end_pos], self.v[layer_idx, :bsz, :end_pos]

    def get_memory_mb(self) -> float:
        """Returns total VRAM footprint of the pre-allocated cache buffers in Megabytes."""
        total_bytes = (self.k.numel() + self.v.numel()) * self.k.element_size()
        return total_bytes / (1024.0 * 1024.0)


# -----------------------------------------------------------------------------
# 3. Profiled Architecture Modules
# -----------------------------------------------------------------------------

class ProfiledAttention(nn.Module):
    """
    Multi-Head Grouped-Query Attention with fine-grained sub-operator profiling and KV Cache:
    - Q_Linear, K_Linear, V_Linear
    - RoPE
    - KV_Cache_Update
    - Attn_Compute (GQA repeat, QK^T, mask if prefill, softmax, PV)
    - O_Linear
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim

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
        # num_input_tokens:
        #   - In prefill: prompt_len (> 1 tokens)
        #   - In decode:  1 token (strictly O(1) linear projection compute!)

        # 1. Linear Projections (separated to track GQA 4:1:1 compute asymmetry)
        with torch.profiler.record_function("Q_Linear"):
            xq = self.q_proj(x).view(bsz, num_input_tokens, self.n_heads, self.head_dim)
        with torch.profiler.record_function("K_Linear"):
            xk = self.k_proj(x).view(bsz, num_input_tokens, self.n_kv_heads, self.head_dim)
        with torch.profiler.record_function("V_Linear"):
            xv = self.v_proj(x).view(bsz, num_input_tokens, self.n_kv_heads, self.head_dim)

        # 2. RoPE Rotation (only rotates the new incoming token's Q and K)
        with torch.profiler.record_function("RoPE"):
            xq = apply_rotary_emb(xq, cos, sin)
            xk = apply_rotary_emb(xk, cos, sin)

        # 3. KV Cache Update (in-place slice write into static buffer)
        if kv_cache is not None:
            with torch.profiler.record_function("KV_Cache_Update"):
                xk, xv = kv_cache.update(layer_idx, start_pos, xk, xv)
        
        # total_context_len: total sequence history available in the KV cache so far
        total_context_len = xk.shape[1]

        # 4. Attention Computation
        with torch.profiler.record_function("Attn_Compute"):
            # Expand 8 KV heads to 32 query heads for GQA
            xk = repeat_kv(xk, self.n_rep)
            xv = repeat_kv(xv, self.n_rep)

            xq = xq.transpose(1, 2)  # (bsz, n_heads, num_input_tokens, head_dim)
            xk = xk.transpose(1, 2)  # (bsz, n_heads, total_context_len, head_dim)
            xv = xv.transpose(1, 2)  # (bsz, n_heads, total_context_len, head_dim)

            # Dot product: (num_input_tokens, head_dim) @ (head_dim, total_context_len) -> (num_input_tokens, total_context_len)
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Causal mask is ONLY needed during prefill (num_input_tokens > 1).
            # During decode (num_input_tokens == 1), single query attends to all past keys [0 : total_context_len],
            # so causal masking is mathematically unnecessary and completely bypassed!
            if num_input_tokens > 1:
                mask = torch.full((num_input_tokens, total_context_len), float("-inf"), device=scores.device, dtype=scores.dtype)
                mask = torch.triu(mask, diagonal=start_pos + 1)
                scores = scores + mask

            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)
            output = torch.matmul(probs, xv)  # (bsz, n_heads, num_input_tokens, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, num_input_tokens, -1)

        # 5. Output Projection
        with torch.profiler.record_function("O_Linear"):
            out = self.o_proj(output)
        return out


class ProfiledFeedForward(nn.Module):
    """SwiGLU Feed-Forward Network with fine-grained profiling."""
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
    """Transformer block tracking RMSNorm, Attention (with KV cache), and SwiGLU MLP."""
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
    """Top-level Transformer tracking Embedding, Layers, RMSNorm_Final, and LM_Head."""
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

        # In autoregressive generation, we only compute logits for the final position
        with torch.profiler.record_function("LM_Head"):
            logits = self.lm_head(h[:, [-1], :])  # shape: (bsz, 1, vocab_size)
        return logits


# -----------------------------------------------------------------------------
# 4. Safetensors Weight Loading
# -----------------------------------------------------------------------------

def resolve_safetensors_path(model_path_or_repo: str) -> str:
    """Resolves local file, directory, or cached snapshot path."""
    if os.path.isfile(model_path_or_repo) and model_path_or_repo.endswith(".safetensors"):
        return model_path_or_repo
    if os.path.isdir(model_path_or_repo):
        cand = os.path.join(model_path_or_repo, "model.safetensors")
        if os.path.exists(cand):
            return cand

    hf_hub_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(hf_hub_cache):
        for repo_sub in ["models--unsloth--Llama-3.2-1B-Instruct", "models--meta-llama--Llama-3.2-1B-Instruct"]:
            matches = glob.glob(os.path.join(hf_hub_cache, repo_sub, "snapshots", "*", "model.safetensors"))
            if matches:
                return matches[0]

    try:
        from huggingface_hub import hf_hub_download
        print(f"[*] Resolving weights via HuggingFace Hub: {model_path_or_repo}...")
        token = os.environ.get("HF_TOKEN", None)
        try:
            return hf_hub_download(repo_id=model_path_or_repo, filename="model.safetensors", token=token)
        except Exception:
            return hf_hub_download(repo_id="unsloth/Llama-3.2-1B-Instruct", filename="model.safetensors")
    except ImportError:
        raise RuntimeError(f"Could not locate model.safetensors for {model_path_or_repo}.")


def load_hf_safetensors(
    model: ProfiledTransformer,
    model_path_or_repo: str = "unsloth/Llama-3.2-1B-Instruct",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> ProfiledTransformer:
    """Loads weights from HuggingFace safetensors format into custom model."""
    safetensors_file = resolve_safetensors_path(model_path_or_repo)
    print(f"[*] Loading weights from: {safetensors_file}")

    raw_state = safetensors.torch.load_file(safetensors_file)
    custom_state = {
        (k[len("model."):] if k.startswith("model.") else k): v.to(dtype=dtype, device=device)
        for k, v in raw_state.items()
    }
    model.load_state_dict(custom_state, strict=False)
    print("[*] Weights successfully mapped and loaded.")
    return model


# -----------------------------------------------------------------------------
# 5. Token Sampling Helper
# -----------------------------------------------------------------------------

def sample_next_token(logits: torch.Tensor, temperature: float = 0.0, top_k: Optional[int] = None) -> int:
    """Samples next token ID with profiling instrumentation."""
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


# -----------------------------------------------------------------------------
# 6. Sampled Profiled Autoregressive Generation Loop
# -----------------------------------------------------------------------------

@torch.inference_mode()
def generate_profiled(
    model: ProfiledTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    profile_interval: int = 100,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    device: str = "cuda",
    profile_output_dir: str = "profile_results",
    ignore_eos: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Autoregressive generation with KV Cache and unified PyTorch Profiler tracking:
    - Step 0 (Prefill): Processes prompt tokens and initializes the KV cache. Always sampled!
    - Decode steps: Sampled every `profile_interval` steps and the final step.
    - Zero code duplication: Uses contextlib.nullcontext() on un-sampled steps to run natively!
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]

    # Stop tokens for Llama-3.2-Instruct
    stop_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id else set()
    for tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)

    # Pre-allocate static KV Cache directly in GPU VRAM
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
        """
        Executes a forward pass and sampling step.
        - tokens: (bsz, num_input_tokens) where num_input_tokens is prompt_len during prefill, 1 during decode.
        - total_context_len: total cumulative tokens in the KV cache so far (for timeline visualization).
        - is_sample: whether to profile this step via PyTorch Profiler or run natively with nullcontext().
        """
        if device == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        # Dynamic context manager: active profiler when sampled, zero-overhead no-op when un-sampled!
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

        # Stream decoded token
        tok_str = tokenizer.decode([next_tok], clean_up_tokenization_spaces=False)
        print(tok_str, end="", flush=True)

        if is_sample:
            # Export Chrome trace for Gantt timeline
            trace_file = os.path.join(traces_dir, f"step_{step_idx}_trace.json")
            prof.export_chrome_trace(trace_file)

            # Extract fine-grained operator metrics and timeline
            step_record = extract_single_token_metric(prof, step_idx, next_tok, tok_str)
            timeline = extract_timeline_from_trace(
                trace_path=trace_file,
                step_idx=step_idx,
                token_id=next_tok,
                token_text=tok_str,
                seq_len=total_context_len,  # Visualizer uses this for context-length scaling
            )
            step_record["timeline"] = timeline
            if "three_metrics" in timeline and timeline["three_metrics"]:
                step_record["three_metrics"] = timeline["three_metrics"]
            step_record["total_latency_ms"] = timeline.get("three_metrics", {}).get("total_latency_ms", round(native_latency_ms, 2))
            step_record["is_sampled"] = True
            step_record["phase"] = "prefill" if step_idx == 0 else "decode"
            token_records.append(step_record)
        else:
            # Lightweight record for native un-profiled steps
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

    # -------------------------------------------------------------------------
    # PHASE 1: PREFILL (Step 0)
    # Full prompt processed in parallel (num_input_tokens = prompt_len), seeding KV cache
    # -------------------------------------------------------------------------
    next_tok = execute_step(
        prompt_ids,
        start_pos=0,
        step_idx=0,
        total_context_len=prompt_len,
        is_sample=True,
    )
    generated_ids.append(next_tok)

    # -------------------------------------------------------------------------
    # PHASE 2: DECODE (Steps 1..max_new_tokens)
    # Feeds strictly 1 token per step (num_input_tokens = 1)
    # -------------------------------------------------------------------------
    curr_tok_tensor = torch.tensor([[next_tok]], device=device)

    for step in range(1, max_new_tokens):
        if not ignore_eos and next_tok in stop_ids:
            break

        start_pos = prompt_len + step - 1
        total_context_len = start_pos + 1  # Total tokens in KV cache history

        # Sampling condition: profile every N steps and the final step
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
# 7. Main Entrypoint & CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Llama-3.2-1B KV-Cache inference with fine-grained PyTorch Profiling.")
    parser.add_argument("--prompt", type=str, default="Write a comprehensive guide on quantum computing principles.", help="Input prompt")
    parser.add_argument("--model_path", type=str, default="unsloth/Llama-3.2-1B-Instruct", help="HuggingFace model ID or local directory")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Tokens to generate")
    parser.add_argument("--profile_interval", type=int, default=25, help="Profiling sampling interval (default: 25)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--ignore_eos", action="store_true", default=False, help="Ignore EOS token to generate fixed length")
    args = parser.parse_args()

    print(f"[*] Running on device: {args.device} ({torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'})")

    # 1. Initialize custom Model Architecture
    model_args = ModelArgs.llama_3_2_1b()
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = ProfiledTransformer(model_args).to(device=args.device, dtype=dtype)
    model.eval()

    # 2. Load Safetensors Weights
    load_hf_safetensors(model, model_path_or_repo=args.model_path, device=args.device, dtype=dtype)

    # 3. Load Tokenizer
    tok_path = resolve_safetensors_path(args.model_path)
    tok_dir = os.path.dirname(tok_path) if os.path.exists(tok_path) else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=os.path.exists(tok_dir))

    # 4. Model Warmup (eliminates one-time CUDA driver/context overhead)
    if args.device == "cuda":
        print("[*] Performing 1-step model warmup...")
        with torch.no_grad():
            dummy = torch.tensor([[1]], device=args.device)
            dummy_cache = KVCache(model.args.n_layers, 1, 16, model.args.n_kv_heads, model.args.head_dim, dtype=dtype, device=args.device)
            _ = model(dummy, start_pos=0, kv_cache=dummy_cache)
            torch.cuda.synchronize()

    # 5. Autoregressive Generation with Sampled Profiling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "profile_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Starting profiled generation (max_new_tokens={args.max_new_tokens}, profile_interval={args.profile_interval})...")
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

    # 6. Render Terminal Summary and Save Dashboard Metrics
    if token_records:
        sampled_records = [r for r in token_records if r.get("is_sampled", True) and r.get("breakdown")]
        print(f"\n[*] Rendering dashboard for {len(token_records)} tokens ({len(sampled_records)} sampled checkpoints)...")
        render_terminal_dashboard(token_records, prompt=args.prompt)

        timeline_records = {r["step"]: r["timeline"] for r in token_records if "timeline" in r and r["timeline"]}
        json_file = os.path.join(output_dir, "token_metrics.json")
        save_json_metrics(token_records, prompt=args.prompt, output_file=json_file, timeline_records=timeline_records)

        rel_viz = os.path.relpath(os.path.join(script_dir, "profile_visualizer.py"), os.getcwd())
        rel_json = os.path.relpath(json_file, os.getcwd())
        print(f"\n[✓] Profiling complete. Metrics saved to: {os.path.abspath(json_file)}")
        print(f"[*] To generate the HTML dashboard, run:")
        print(f"    python {rel_viz} --json {rel_json}")


if __name__ == "__main__":
    main()
