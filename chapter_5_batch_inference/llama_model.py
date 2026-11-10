"""
Native PyTorch Llama-3.2-1B-Instruct with Static KV Cache for Chapter 5 Batch Inference.

Intentionally uses pure eager PyTorch operations (bfloat16, no custom Triton/CUDA kernels)
so batching optimizations (Sequential FCFS -> Static Batching -> Continuous Batching ->
Chunked Prefill -> PagedAttention) can be evaluated and profiled cleanly from first principles.

Fine-grained `torch.profiler.record_function` scopes can be toggled via `model.set_profile_ops(True)`
during deep hardware/operator profiling and disabled (`False`) during serving timing runs so
Python profiler context overhead never distorts queueing or latency measurements.
"""

import glob
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

_NULL_CTX = nullcontext()


def _op_scope(enabled: bool, name: str):
    return torch.profiler.record_function(name) if enabled else _NULL_CTX


# -----------------------------------------------------------------------------
# 1. Model Configuration & Primitives
# -----------------------------------------------------------------------------


@dataclass
class ModelArgs:
    """Hyperparameters for Llama-3.2-1B-Instruct."""

    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128256
    hidden_dim: int = 8192
    head_dim: int = 64
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_seq_len: int = 8192
    tie_word_embeddings: bool = True
    rope_scaling: Optional[dict] = None

    @classmethod
    def llama_3_2_1b(cls, max_seq_len: int = 8192) -> "ModelArgs":
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
            max_seq_len=max_seq_len,
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
    """Native PyTorch Root Mean Square Layer Normalization."""

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
    """Precomputes RoPE cos/sin tables with Llama-3 wavelength scaling."""
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
    """Broadcasts unrepeated KV heads (8) across GQA query groups (32)."""
    if n_rep == 1:
        return x
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seqlen, n_kv_heads * n_rep, head_dim)
    )


# -----------------------------------------------------------------------------
# 2. Static Contiguous KV Cache Buffer
# -----------------------------------------------------------------------------


class KVCache:
    """
    Pre-allocated static GPU buffer for Key-Value caching across all layers.
    Shape per buffer: (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim).
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
        self.n_layers = n_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

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
            raise ValueError(
                f"Context length ({end_pos}) exceeds pre-allocated KVCache max_seq_len ({self.max_seq_len})."
            )
        self.k[layer_idx, :bsz, start_pos:end_pos] = xk
        self.v[layer_idx, :bsz, start_pos:end_pos] = xv
        return self.k[layer_idx, :bsz, :end_pos], self.v[layer_idx, :bsz, :end_pos]

    def bytes_per_token(self) -> int:
        """KV bytes consumed per cached token across all layers (32 KB in BF16 for Llama-3.2-1B)."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.k.element_size()

    def get_allocated_mb(self) -> float:
        total_bytes = (self.k.numel() + self.v.numel()) * self.k.element_size()
        return total_bytes / (1024.0 * 1024.0)


# -----------------------------------------------------------------------------
# 3. Native PyTorch Transformer Modules
# -----------------------------------------------------------------------------


class NativeAttention(nn.Module):
    """Grouped-Query Attention in pure PyTorch with optional record_function scopes."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.profile_ops: bool = False

        self.q_proj = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_idx: int = 0,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, num_input_tokens, _ = x.shape
        prof = self.profile_ops

        with _op_scope(prof, "Q_Linear"):
            xq = self.q_proj(x).view(bsz, num_input_tokens, self.n_heads, self.head_dim)
        with _op_scope(prof, "K_Linear"):
            xk = self.k_proj(x).view(bsz, num_input_tokens, self.n_kv_heads, self.head_dim)
        with _op_scope(prof, "V_Linear"):
            xv = self.v_proj(x).view(bsz, num_input_tokens, self.n_kv_heads, self.head_dim)

        with _op_scope(prof, "RoPE"):
            xq = apply_rotary_emb(xq, cos, sin)
            xk = apply_rotary_emb(xk, cos, sin)

        if kv_cache is not None:
            with _op_scope(prof, "KV_Cache_Update"):
                xk, xv = kv_cache.update(layer_idx, start_pos, xk, xv)

        total_context_len = xk.shape[1]

        with _op_scope(prof, "Attn_Compute"):
            xk = repeat_kv(xk, self.n_rep)
            xv = repeat_kv(xv, self.n_rep)

            xq = xq.transpose(1, 2)  # (bsz, n_heads, num_input_tokens, head_dim)
            xk = xk.transpose(1, 2)  # (bsz, n_heads, total_context_len, head_dim)
            xv = xv.transpose(1, 2)  # (bsz, n_heads, total_context_len, head_dim)

            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if attn_mask is not None:
                scores = scores + attn_mask
            elif num_input_tokens > 1:
                mask = torch.full(
                    (num_input_tokens, total_context_len),
                    float("-inf"),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                mask = torch.triu(mask, diagonal=start_pos + 1)
                scores = scores + mask

            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)
            output = torch.matmul(probs, xv)
            output = output.transpose(1, 2).contiguous().view(bsz, num_input_tokens, -1)

        with _op_scope(prof, "O_Linear"):
            out = self.o_proj(output)
        return out


class NativeFeedForward(nn.Module):
    """SwiGLU MLP in pure PyTorch with optional record_function scopes."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.profile_ops: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prof = self.profile_ops
        with _op_scope(prof, "FFN_Gate_Up_Linear"):
            g = self.gate_proj(x)
            u = self.up_proj(x)
        with _op_scope(prof, "FFN_SiLU_Mul"):
            act = F.silu(g) * u
        with _op_scope(prof, "FFN_Down_Linear"):
            out = self.down_proj(act)
        return out


class NativeTransformerBlock(nn.Module):
    """Single Transformer block (RMSNorm + GQA + RMSNorm + SwiGLU)."""

    def __init__(self, layer_idx: int, args: ModelArgs):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.self_attn = NativeAttention(args)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = NativeFeedForward(args.dim, args.hidden_dim)
        self.profile_ops: bool = False

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prof = self.profile_ops
        with _op_scope(prof, "RMSNorm_Attn"):
            norm_x1 = self.input_layernorm(x)
        x = x + self.self_attn(
            norm_x1,
            cos,
            sin,
            layer_idx=self.layer_idx,
            start_pos=start_pos,
            kv_cache=kv_cache,
            attn_mask=attn_mask,
        )
        with _op_scope(prof, "RMSNorm_FFN"):
            norm_x2 = self.post_attention_layernorm(x)
        x = x + self.mlp(norm_x2)
        return x


class NativeLlamaTransformer(nn.Module):
    """Native PyTorch Llama-3.2-1B model with KV Cache and toggleable op profiling."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.profile_ops: bool = False
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([NativeTransformerBlock(i, args) for i in range(args.n_layers)])
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

    def set_profile_ops(self, enabled: bool) -> None:
        """Enables or disables fine-grained torch.profiler.record_function annotations."""
        self.profile_ops = enabled
        for block in self.layers:
            block.profile_ops = enabled
            block.self_attn.profile_ops = enabled
            block.mlp.profile_ops = enabled

    def get_weight_memory_mb(self) -> float:
        """Returns unique model parameter footprint in MiB (accounting for tied embeddings)."""
        seen_ptrs = set()
        total_bytes = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                total_bytes += p.numel() * p.element_size()
        return total_bytes / (1024.0 * 1024.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, num_input_tokens = input_ids.shape
        prof = self.profile_ops

        with _op_scope(prof, "Embedding"):
            h = self.embed_tokens(input_ids)

        if position_ids is not None:
            # Supports per-sequence RoPE positions (bsz, seqlen) for future padded/continuous batching
            cos = self.cos_cached[position_ids].unsqueeze(2)
            sin = self.sin_cached[position_ids].unsqueeze(2)
        else:
            cos = self.cos_cached[start_pos : start_pos + num_input_tokens].unsqueeze(0).unsqueeze(2)
            sin = self.sin_cached[start_pos : start_pos + num_input_tokens].unsqueeze(0).unsqueeze(2)

        for layer in self.layers:
            h = layer(h, cos, sin, start_pos=start_pos, kv_cache=kv_cache, attn_mask=attn_mask)

        with _op_scope(prof, "RMSNorm_Final"):
            h = self.norm(h)

        with _op_scope(prof, "LM_Head"):
            logits = self.lm_head(h[:, [-1], :])  # (bsz, 1, vocab_size)
        return logits


# -----------------------------------------------------------------------------
# 4. Checkpoint, Tokenizer & Sampling Utilities
# -----------------------------------------------------------------------------


def resolve_safetensors_path(model_path_or_repo: str = "unsloth/Llama-3.2-1B-Instruct") -> str:
    if os.path.isfile(model_path_or_repo) and model_path_or_repo.endswith(".safetensors"):
        return model_path_or_repo
    if os.path.isdir(model_path_or_repo):
        cand = os.path.join(model_path_or_repo, "model.safetensors")
        if os.path.exists(cand):
            return cand

    hf_hub_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(hf_hub_cache):
        for repo_sub in [
            "models--unsloth--Llama-3.2-1B-Instruct",
            "models--meta-llama--Llama-3.2-1B-Instruct",
        ]:
            matches = glob.glob(os.path.join(hf_hub_cache, repo_sub, "snapshots", "*", "model.safetensors"))
            if matches:
                return matches[0]

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN", None)
    try:
        return hf_hub_download(repo_id=model_path_or_repo, filename="model.safetensors", token=token)
    except Exception:
        return hf_hub_download(repo_id="unsloth/Llama-3.2-1B-Instruct", filename="model.safetensors")


def load_model_and_tokenizer(
    model_path_or_repo: str = "unsloth/Llama-3.2-1B-Instruct",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = 8192,
) -> Tuple[NativeLlamaTransformer, AutoTokenizer]:
    """Loads NativeLlamaTransformer and HuggingFace tokenizer from local cache or Hub."""
    model_args = ModelArgs.llama_3_2_1b(max_seq_len=max_seq_len)
    model = NativeLlamaTransformer(model_args).to(device=device, dtype=dtype)
    model.eval()

    safetensors_file = resolve_safetensors_path(model_path_or_repo)
    raw_state = safetensors.torch.load_file(safetensors_file)
    custom_state = {
        (k[len("model.") :] if k.startswith("model.") else k): v.to(dtype=dtype, device=device)
        for k, v in raw_state.items()
    }
    model.load_state_dict(custom_state, strict=False)

    tok_dir = os.path.dirname(safetensors_file) if os.path.exists(safetensors_file) else model_path_or_repo
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=os.path.exists(tok_dir))
    return model, tokenizer


def get_stop_token_ids(tokenizer: AutoTokenizer) -> Set[int]:
    stop_ids: Set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
    return stop_ids


def sample_next_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    profile_ops: bool = False,
) -> torch.Tensor:
    """
    Samples next token IDs for a batch of logits of shape (bsz, 1, vocab_size).
    Returns tensor of shape (bsz, 1) on the same device.
    """
    with _op_scope(profile_ops, "Sampling"):
        last_logits = logits[:, -1, :]
        if temperature == 0.0:
            return torch.argmax(last_logits, dim=-1, keepdim=True)

        scaled = last_logits / temperature
        if top_k is not None:
            v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
            scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
        probs = F.softmax(scaled, dim=-1, dtype=torch.float32)
        return torch.multinomial(probs, num_samples=1)

