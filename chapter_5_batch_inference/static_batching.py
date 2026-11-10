"""
Standalone Static (Request-Level) Padded Batching Engine for Chapter 5.

Completely self-contained copy of the pure-PyTorch Llama-3.2-1B-Instruct model,
static batched KV cache, and request-level batch scheduler (no shared model state
with the B=1 baseline or Continuous Batching modules).

Core Systems Mechanics of Static Batching:
1. Request-Level Batching: Pops up to `max_batch_size` arrived requests from the FIFO queue
   and executes them together as a fixed batch until *every* sequence in the batch finishes.
2. Left-Padded Batched Prefill: Prompts of lengths [p_0, ..., p_{b-1}] are left-padded to
   `P_max = max(p_i)` so all final prompt tokens align at index `P_max - 1`, using per-sequence
   RoPE `position_ids` and a causal + left-pad attention mask.
   -> Penalty 1: Wasted prefill compute on `sum(P_max - p_i)` pad tokens.
3. Lockstep Batched Decode: Steps `1 .. M_max - 1` run batched GEMMs of shape `(b, 1, dim)`,
   amortizing the 2.36 GiB HBM weight read across `b` requests!
   -> Penalty 2 (Zombie Slots & Straggler Blocking): Sequences that finish early (`m_i < M_max`)
      cannot release their batch slot until the longest straggler (`M_max`) completes, wasting
      decode compute on zombie slots and blocking newly arrived requests in the waiting queue.
"""

import glob
import math
import os
import statistics
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    from .profiler_metrics import RequestExecutionRecord, analyze_chrome_trace
    from .workloads import InferenceRequest
except ImportError:
    from profiler_metrics import RequestExecutionRecord, analyze_chrome_trace
    from workloads import InferenceRequest

_NULL_CTX = nullcontext()


def _op_scope(enabled: bool, name: str):
    return torch.profiler.record_function(name) if enabled else _NULL_CTX


# -----------------------------------------------------------------------------
# 1. Standalone Native PyTorch Llama-3.2-1B Architecture (Static Batching Copy)
# -----------------------------------------------------------------------------


@dataclass
class StaticModelArgs:
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
    def llama_3_2_1b(cls, max_seq_len: int = 8192) -> "StaticModelArgs":
        return cls(
            max_seq_len=max_seq_len,
            rope_scaling={
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )


class StaticRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed.to(input_dtype)


def _precompute_rope_freqs(
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seqlen, n_kv_heads * n_rep, head_dim)
    )


class StaticBatchKVCache:
    """Contiguous `(n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)` KV buffer."""

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
            raise ValueError(f"Padded context length ({end_pos}) exceeds max_seq_len ({self.max_seq_len}).")
        self.k[layer_idx, :bsz, start_pos:end_pos] = xk
        self.v[layer_idx, :bsz, start_pos:end_pos] = xv
        return self.k[layer_idx, :bsz, :end_pos], self.v[layer_idx, :bsz, :end_pos]

    def bytes_per_token(self) -> int:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.k.element_size()

    def get_allocated_mb(self) -> float:
        total_bytes = (self.k.numel() + self.v.numel()) * self.k.element_size()
        return total_bytes / (1024.0 * 1024.0)


class StaticBatchAttention(nn.Module):
    def __init__(self, args: StaticModelArgs):
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
        layer_idx: int,
        start_pos: int,
        kv_cache: StaticBatchKVCache,
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
            xq = _apply_rotary_emb(xq, cos, sin)
            xk = _apply_rotary_emb(xk, cos, sin)

        with _op_scope(prof, "KV_Cache_Update"):
            xk, xv = kv_cache.update(layer_idx, start_pos, xk, xv)

        with _op_scope(prof, "Attn_Compute"):
            xk = _repeat_kv(xk, self.n_rep)
            xv = _repeat_kv(xv, self.n_rep)

            xq = xq.transpose(1, 2)  # (bsz, n_heads, q_len, head_dim)
            xk = xk.transpose(1, 2)  # (bsz, n_heads, kv_len, head_dim)
            xv = xv.transpose(1, 2)  # (bsz, n_heads, kv_len, head_dim)

            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores + attn_mask

            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)
            output = torch.matmul(probs, xv)
            output = output.transpose(1, 2).contiguous().view(bsz, num_input_tokens, -1)

        with _op_scope(prof, "O_Linear"):
            return self.o_proj(output)


class StaticBatchFeedForward(nn.Module):
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
            return self.down_proj(act)


class StaticBatchTransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, args: StaticModelArgs):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = StaticRMSNorm(args.dim, eps=args.norm_eps)
        self.self_attn = StaticBatchAttention(args)
        self.post_attention_layernorm = StaticRMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = StaticBatchFeedForward(args.dim, args.hidden_dim)
        self.profile_ops: bool = False

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int,
        kv_cache: StaticBatchKVCache,
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


class StaticBatchLlamaTransformer(nn.Module):
    def __init__(self, args: StaticModelArgs):
        super().__init__()
        self.args = args
        self.profile_ops: bool = False
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([StaticBatchTransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = StaticRMSNorm(args.dim, eps=args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        if args.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = _precompute_rope_freqs(
            dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            theta=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def set_profile_ops(self, enabled: bool) -> None:
        self.profile_ops = enabled
        for block in self.layers:
            block.profile_ops = enabled
            block.self_attn.profile_ops = enabled
            block.mlp.profile_ops = enabled

    def get_weight_memory_mb(self) -> float:
        seen = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel() * p.element_size()
        return total / (1024.0 * 1024.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start_pos: int,
        kv_cache: StaticBatchKVCache,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prof = self.profile_ops
        with _op_scope(prof, "Embedding"):
            h = self.embed_tokens(input_ids)

        cos = self.cos_cached[position_ids].unsqueeze(2)  # (bsz, seqlen, 1, head_dim)
        sin = self.sin_cached[position_ids].unsqueeze(2)

        for layer in self.layers:
            h = layer(h, cos, sin, start_pos=start_pos, kv_cache=kv_cache, attn_mask=attn_mask)

        with _op_scope(prof, "RMSNorm_Final"):
            h = self.norm(h)

        with _op_scope(prof, "LM_Head"):
            logits = self.lm_head(h[:, [-1], :])  # (bsz, 1, vocab_size)
        return logits


def _resolve_safetensors(model_path_or_repo: str = "unsloth/Llama-3.2-1B-Instruct") -> str:
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

    return hf_hub_download(repo_id=model_path_or_repo, filename="model.safetensors")


def load_static_batch_model(
    model_path_or_repo: str = "unsloth/Llama-3.2-1B-Instruct",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = 8192,
) -> Tuple[StaticBatchLlamaTransformer, AutoTokenizer]:
    args = StaticModelArgs.llama_3_2_1b(max_seq_len=max_seq_len)
    model = StaticBatchLlamaTransformer(args).to(device=device, dtype=dtype)
    model.eval()

    st_file = _resolve_safetensors(model_path_or_repo)
    raw = safetensors.torch.load_file(st_file)
    mapped = {
        (k[len("model.") :] if k.startswith("model.") else k): v.to(dtype=dtype, device=device)
        for k, v in raw.items()
    }
    model.load_state_dict(mapped, strict=False)

    tok_dir = os.path.dirname(st_file) if os.path.exists(st_file) else model_path_or_repo
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=os.path.exists(tok_dir))
    return model, tokenizer


def _get_stop_ids(tokenizer: AutoTokenizer) -> Set[int]:
    stop_ids: Set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
    return stop_ids


def _sample_batch(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    profile_ops: bool = False,
) -> torch.Tensor:
    with _op_scope(profile_ops, "Sampling"):
        last = logits[:, -1, :]
        if temperature == 0.0:
            return torch.argmax(last, dim=-1, keepdim=True)
        scaled = last / temperature
        if top_k is not None:
            v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
            scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
        probs = F.softmax(scaled, dim=-1, dtype=torch.float32)
        return torch.multinomial(probs, num_samples=1)


# -----------------------------------------------------------------------------
# 2. Static Padded Batching Engine
# -----------------------------------------------------------------------------


@dataclass
class StaticBatchRunOutput:
    records: List[RequestExecutionRecord]
    wall_clock_exec_s: float
    gpu_idle_time_s: float
    peak_vram_allocated_mb: float
    peak_vram_reserved_mb: float
    batch_diagnostics: Dict[str, Any] = field(default_factory=dict)


class StaticBatchEngine:
    """
    Request-level Static Batching Engine (`max_batch_size = B`).

    When the GPU finishes a batch at `sim_clock_s`, pops up to `max_batch_size` currently
    queued requests, left-pads their prompts to `P_max`, and decodes in lockstep until
    every request in the batch completes (`M_max`).
    """

    def __init__(
        self,
        model: StaticBatchLlamaTransformer,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.stop_ids = _get_stop_ids(tokenizer)
        self.pad_token_id = tokenizer.eos_token_id or 0
        self.name = f"static_batch_b{max_batch_size}"

        self.kv_cache = StaticBatchKVCache(
            n_layers=model.args.n_layers,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_kv_heads=model.args.n_kv_heads,
            head_dim=model.args.head_dim,
            dtype=self.dtype,
            device=device,
        )

    def get_kv_allocated_mb(self) -> float:
        return self.kv_cache.get_allocated_mb()

    def get_kv_bytes_per_token(self) -> int:
        return self.kv_cache.bytes_per_token()

    def _build_prefill_tensors(
        self, batch: List[InferenceRequest]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int], int]:
        bsz = len(batch)
        p_max = max(r.prompt_len for r in batch)
        pad_lens = [p_max - r.prompt_len for r in batch]

        input_ids = torch.full((bsz, p_max), self.pad_token_id, dtype=torch.long, device=self.device)
        pos_ids = torch.zeros((bsz, p_max), dtype=torch.long, device=self.device)
        base_cols = torch.arange(p_max, dtype=torch.long, device=self.device)

        for i, req in enumerate(batch):
            pl = pad_lens[i]
            input_ids[i, pl:] = torch.tensor(req.prompt_token_ids, dtype=torch.long, device=self.device)
            pos_ids[i] = (base_cols - pl).clamp(min=0)

        if p_max == 1 and all(pl == 0 for pl in pad_lens):
            attn_mask = None
        else:
            q_idx = base_cols[None, None, :, None]
            k_idx = base_cols[None, None, None, :]
            pad_t = torch.tensor(pad_lens, dtype=torch.long, device=self.device)[:, None, None, None]
            # Real queries attend causally to non-pad keys; pad queries attend to self diagonal to avoid NaN
            allow = ((k_idx <= q_idx) & (k_idx >= pad_t)) | ((q_idx < pad_t) & (k_idx == q_idx))
            attn_mask = torch.where(allow, 0.0, float("-inf")).to(dtype=self.dtype)

        return input_ids, pos_ids, attn_mask, pad_lens, p_max

    @torch.inference_mode()
    def warmup(self, include_profiler: bool = True) -> None:
        self.model.set_profile_ops(False)
        for b in sorted({1, min(4, self.max_batch_size), self.max_batch_size}):
            p_len = 128
            ids = torch.ones((b, p_len), dtype=torch.long, device=self.device)
            pos = torch.arange(p_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(b, p_len)
            q_idx = torch.arange(p_len, device=self.device)[None, None, :, None]
            k_idx = torch.arange(p_len, device=self.device)[None, None, None, :]
            mask = torch.where(k_idx <= q_idx, 0.0, float("-inf")).to(self.dtype)
            logits = self.model(ids, pos, start_pos=0, kv_cache=self.kv_cache, attn_mask=mask)
            nxt = _sample_batch(logits)
            _ = nxt.squeeze(-1).tolist()
            dec_pos = torch.full((b, 1), p_len, dtype=torch.long, device=self.device)
            logits_d = self.model(nxt, dec_pos, start_pos=p_len, kv_cache=self.kv_cache, attn_mask=None)
            _ = _sample_batch(logits_d).squeeze(-1).tolist()

        if self.device == "cuda":
            torch.cuda.synchronize()

        if include_profiler and self.device == "cuda":
            self.model.set_profile_ops(True)
            b = self.max_batch_size
            ids = torch.ones((b, 32), dtype=torch.long, device=self.device)
            pos = torch.arange(32, dtype=torch.long, device=self.device).unsqueeze(0).expand(b, 32)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            ):
                logits = self.model(ids, pos, start_pos=0, kv_cache=self.kv_cache)
                _ = _sample_batch(logits, profile_ops=True).squeeze(-1).tolist()
            torch.cuda.synchronize()
            self.model.set_profile_ops(False)

    @torch.inference_mode()
    def serve_requests(
        self,
        requests: List[InferenceRequest],
        real_time: bool = False,
        ignore_eos: bool = True,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
    ) -> StaticBatchRunOutput:
        self.model.set_profile_ops(False)
        sorted_reqs = sorted(requests, key=lambda r: (r.arrival_time_s, r.request_id))
        n_total = len(sorted_reqs)

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        ready_queue: Deque[InferenceRequest] = deque()
        records: List[RequestExecutionRecord] = []
        next_idx = 0
        sim_clock_s = 0.0
        gpu_idle_time_s = 0.0

        # Static batching efficiency counters
        num_batches_run = 0
        batch_sizes_used: List[int] = []
        total_prefill_real_tokens = 0
        total_prefill_padded_tokens = 0
        total_decode_active_slot_steps = 0
        total_decode_zombie_slot_steps = 0

        wall_t0 = time.perf_counter()

        def wall_now_s() -> float:
            return time.perf_counter() - wall_t0

        while len(records) < n_total:
            if not real_time:
                while next_idx < n_total and sorted_reqs[next_idx].arrival_time_s <= sim_clock_s:
                    ready_queue.append(sorted_reqs[next_idx])
                    next_idx += 1

                if not ready_queue and next_idx < n_total:
                    next_arr = sorted_reqs[next_idx].arrival_time_s
                    if next_arr > sim_clock_s:
                        gpu_idle_time_s += next_arr - sim_clock_s
                        sim_clock_s = next_arr
                    while next_idx < n_total and sorted_reqs[next_idx].arrival_time_s <= sim_clock_s:
                        ready_queue.append(sorted_reqs[next_idx])
                        next_idx += 1
            else:
                curr_now = wall_now_s()
                while next_idx < n_total and sorted_reqs[next_idx].arrival_time_s <= curr_now:
                    ready_queue.append(sorted_reqs[next_idx])
                    next_idx += 1

                if not ready_queue and next_idx < n_total:
                    target_arr = sorted_reqs[next_idx].arrival_time_s
                    idle_start = wall_now_s()
                    rem = target_arr - idle_start
                    if rem > 0.002:
                        time.sleep(rem - 0.001)
                    while wall_now_s() < target_arr:
                        pass
                    curr_now = wall_now_s()
                    gpu_idle_time_s += max(0.0, curr_now - idle_start)
                    while next_idx < n_total and sorted_reqs[next_idx].arrival_time_s <= curr_now:
                        ready_queue.append(sorted_reqs[next_idx])
                        next_idx += 1

            # Pop up to `max_batch_size` currently queued requests
            batch: List[InferenceRequest] = []
            while ready_queue and len(batch) < self.max_batch_size:
                batch.append(ready_queue.popleft())

            bsz = len(batch)
            num_batches_run += 1
            batch_sizes_used.append(bsz)

            if self.device == "cuda":
                torch.cuda.synchronize()
            dispatch_time_s = wall_now_s() if real_time else sim_clock_s

            # --- Phase 1: Left-Padded Batched Prefill ---
            t_pre0 = time.perf_counter()
            input_ids, pos_ids, pre_mask, pad_lens, p_max = self._build_prefill_tensors(batch)
            logits = self.model(input_ids, pos_ids, start_pos=0, kv_cache=self.kv_cache, attn_mask=pre_mask)
            next_tok = _sample_batch(logits, temperature=temperature, top_k=top_k)
            tok_ids = next_tok.squeeze(-1).tolist()  # Synchronizes CUDA stream
            t_pre1 = time.perf_counter()

            if real_time:
                first_tok_time_s = wall_now_s()
            else:
                sim_clock_s += t_pre1 - t_pre0
                first_tok_time_s = sim_clock_s

            total_prefill_real_tokens += sum(r.prompt_len for r in batch)
            total_prefill_padded_tokens += bsz * p_max

            token_times_per_req: List[List[float]] = [[first_tok_time_s] for _ in range(bsz)]
            gen_ids_per_req: List[List[int]] = [[int(tok_ids[i])] for i in range(bsz)]
            finish_times: List[float] = [first_tok_time_s for _ in range(bsz)]
            finished: List[bool] = [
                (batch[i].target_output_len <= 1) or (not ignore_eos and int(tok_ids[i]) in self.stop_ids)
                for i in range(bsz)
            ]

            # --- Phase 2: Lockstep Batched Decode until Straggler Completes ---
            m_max = max(r.target_output_len for r in batch)
            has_left_pad = any(pl > 0 for pl in pad_lens)
            pad_t = (
                torch.tensor(pad_lens, dtype=torch.long, device=self.device)[:, None, None, None]
                if has_left_pad
                else None
            )
            p_lens_t = torch.tensor([r.prompt_len for r in batch], dtype=torch.long, device=self.device).unsqueeze(1)

            step = 1
            while step < m_max and not all(finished):
                t_d0 = time.perf_counter()
                start_pos = p_max + step - 1
                total_ctx = p_max + step
                dec_pos_ids = p_lens_t + (step - 1)  # Exact per-sequence RoPE position!

                if has_left_pad:
                    k_idx = torch.arange(total_ctx, dtype=torch.long, device=self.device)[None, None, None, :]
                    dec_mask = torch.where(k_idx >= pad_t, 0.0, float("-inf")).to(dtype=self.dtype)
                else:
                    dec_mask = None

                logits = self.model(
                    next_tok,
                    dec_pos_ids,
                    start_pos=start_pos,
                    kv_cache=self.kv_cache,
                    attn_mask=dec_mask,
                )
                next_tok = _sample_batch(logits, temperature=temperature, top_k=top_k)
                tok_ids = next_tok.squeeze(-1).tolist()
                t_d1 = time.perf_counter()

                if real_time:
                    step_time_s = wall_now_s()
                else:
                    sim_clock_s += t_d1 - t_d0
                    step_time_s = sim_clock_s

                for i in range(bsz):
                    if finished[i]:
                        total_decode_zombie_slot_steps += 1
                    else:
                        total_decode_active_slot_steps += 1
                        tid = int(tok_ids[i])
                        token_times_per_req[i].append(step_time_s)
                        gen_ids_per_req[i].append(tid)
                        if len(gen_ids_per_req[i]) >= batch[i].target_output_len or (
                            not ignore_eos and tid in self.stop_ids
                        ):
                            finished[i] = True
                            finish_times[i] = step_time_s

                step += 1

            for i, req in enumerate(batch):
                records.append(
                    RequestExecutionRecord(
                        request_id=req.request_id,
                        source_id=req.source_id,
                        prompt_len=req.prompt_len,
                        target_output_len=req.target_output_len,
                        output_len=len(gen_ids_per_req[i]),
                        arrival_time_s=req.arrival_time_s,
                        dispatch_time_s=dispatch_time_s,
                        first_token_time_s=first_tok_time_s,
                        finish_time_s=finish_times[i],
                        token_times_s=token_times_per_req[i],
                        generated_token_ids=gen_ids_per_req[i],
                    )
                )

        wall_exec_s = time.perf_counter() - wall_t0
        if self.device == "cuda":
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            peak_res_mb = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
        else:
            peak_alloc_mb, peak_res_mb = 0.0, 0.0

        records.sort(key=lambda r: r.request_id)

        pad_waste_pct = (
            (1.0 - total_prefill_real_tokens / max(1, total_prefill_padded_tokens)) * 100.0
        )
        total_slot_steps = total_decode_active_slot_steps + total_decode_zombie_slot_steps
        zombie_waste_pct = (
            (total_decode_zombie_slot_steps / max(1, total_slot_steps)) * 100.0
        )

        diagnostics = {
            "max_batch_size": self.max_batch_size,
            "num_batches_executed": num_batches_run,
            "avg_batch_size_dispatched": round(statistics.mean(batch_sizes_used), 2) if batch_sizes_used else 0.0,
            "prefill_real_tokens": total_prefill_real_tokens,
            "prefill_padded_tokens": total_prefill_padded_tokens,
            "prefill_padding_waste_pct": round(pad_waste_pct, 1),
            "decode_active_slot_steps": total_decode_active_slot_steps,
            "decode_zombie_slot_steps": total_decode_zombie_slot_steps,
            "decode_zombie_slot_waste_pct": round(zombie_waste_pct, 1),
            "effective_active_batch_during_decode": round(
                (total_decode_active_slot_steps / max(1, total_slot_steps)) * statistics.mean(batch_sizes_used), 2
            )
            if batch_sizes_used
            else 0.0,
        }

        return StaticBatchRunOutput(
            records=records,
            wall_clock_exec_s=wall_exec_s,
            gpu_idle_time_s=gpu_idle_time_s,
            peak_vram_allocated_mb=peak_alloc_mb,
            peak_vram_reserved_mb=peak_res_mb,
            batch_diagnostics=diagnostics,
        )

    @torch.inference_mode()
    def profile_representative_phases(
        self,
        prefill_seq_len: int = 128,
        decode_context_len: int = 256,
        traces_dir: str = "chapter_5_batch_inference/profile_results/traces",
        native_trials: int = 5,
    ) -> Dict[str, Any]:
        """Profiles batched Prefill (`B, prefill_seq_len`) and batched Decode (`B, 1`)."""
        os.makedirs(traces_dir, exist_ok=True)
        b = self.max_batch_size
        prefill_seq_len = min(max(8, prefill_seq_len), self.max_seq_len - 16)
        decode_context_len = min(max(prefill_seq_len + 1, decode_context_len), self.max_seq_len - 4)

        prompt_ids = torch.ones((b, prefill_seq_len), dtype=torch.long, device=self.device)
        pre_pos = torch.arange(prefill_seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(b, -1)
        q_idx = torch.arange(prefill_seq_len, device=self.device)[None, None, :, None]
        k_idx = torch.arange(prefill_seq_len, device=self.device)[None, None, None, :]
        pre_mask = torch.where(k_idx <= q_idx, 0.0, float("-inf")).to(self.dtype)

        decode_tok = torch.ones((b, 1), dtype=torch.long, device=self.device)
        dec_pos = torch.full((b, 1), decode_context_len - 1, dtype=torch.long, device=self.device)
        seed_ids = torch.ones((b, decode_context_len - 1), dtype=torch.long, device=self.device)
        seed_pos = torch.arange(decode_context_len - 1, dtype=torch.long, device=self.device).unsqueeze(0).expand(b, -1)

        self.model.set_profile_ops(False)
        prefill_walls: List[float] = []
        for _ in range(native_trials):
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = self.model(prompt_ids, pre_pos, start_pos=0, kv_cache=self.kv_cache, attn_mask=pre_mask)
            _ = _sample_batch(logits).squeeze(-1).tolist()
            prefill_walls.append((time.perf_counter() - t0) * 1000.0)
        native_pre_ms = statistics.median(prefill_walls)

        _ = self.model(seed_ids, seed_pos, start_pos=0, kv_cache=self.kv_cache)
        if self.device == "cuda":
            torch.cuda.synchronize()

        decode_walls: List[float] = []
        for _ in range(native_trials):
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = self.model(decode_tok, dec_pos, start_pos=decode_context_len - 1, kv_cache=self.kv_cache)
            _ = _sample_batch(logits).squeeze(-1).tolist()
            decode_walls.append((time.perf_counter() - t0) * 1000.0)
        native_dec_ms = statistics.median(decode_walls)

        self.model.set_profile_ops(True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        pre_trace = os.path.join(traces_dir, f"static_b{b}_prefill_trace.json")
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof_pre:
            with torch.profiler.record_function("step_prefill"):
                logits = self.model(prompt_ids, pre_pos, start_pos=0, kv_cache=self.kv_cache, attn_mask=pre_mask)
                _ = _sample_batch(logits, profile_ops=True).squeeze(-1).tolist()
        if self.device == "cuda":
            torch.cuda.synchronize()
        prof_pre.export_chrome_trace(pre_trace)

        self.model.set_profile_ops(False)
        _ = self.model(seed_ids, seed_pos, start_pos=0, kv_cache=self.kv_cache)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.model.set_profile_ops(True)

        dec_trace = os.path.join(traces_dir, f"static_b{b}_decode_trace.json")
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof_dec:
            with torch.profiler.record_function("step_decode"):
                logits = self.model(decode_tok, dec_pos, start_pos=decode_context_len - 1, kv_cache=self.kv_cache)
                _ = _sample_batch(logits, profile_ops=True).squeeze(-1).tolist()
        if self.device == "cuda":
            torch.cuda.synchronize()
        prof_dec.export_chrome_trace(dec_trace)

        self.model.set_profile_ops(False)

        return {
            "batch_size": b,
            "prefill_seq_len": prefill_seq_len,
            "decode_context_len": decode_context_len,
            "prefill": analyze_chrome_trace(pre_trace, "step_prefill", native_wall_ms=native_pre_ms),
            "decode": analyze_chrome_trace(dec_trace, "step_decode", native_wall_ms=native_dec_ms),
        }

