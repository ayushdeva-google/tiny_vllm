"""
Standalone Continuous (Iteration-Level + Selective) Batching Engine for Chapter 5.

Completely self-contained copy of the pure-PyTorch Llama-3.2-1B-Instruct model,
slot-managed KV pool (`SlotKVCache`), and Orca-style iteration-level scheduler.

Core Systems Mechanics (Orca, OSDI '22):
1. Iteration-Level Scheduling:
   - At *every* forward-pass boundary, any slot whose request just hit `target_output_len`
     (or EOS) immediately frees its `slot_id` and retires — zero zombie decode steps!
   - Any newly arrived requests waiting in `ready_queue` immediately claim free `slot_id`s
     and join the very next forward pass without waiting for in-flight stragglers to finish!
2. Selective Ragged Token Packing (Zero Prefill Padding):
   - All token-independent layers (`Embedding`, `RMSNorm`, `Q/K/V/O_Linear`, `RoPE`,
     `SwiGLU FFN`, `LM_Head`) pack all newly admitted prefill tokens and all active decode
     tokens into a single unpadded `(1, T_total, dim)` tensor — zero wasted pad FLOPs and
     a single shared 2.36 GiB HBM weight read per layer!
   - Only `SelectiveAttention` branches by sequence history: unpadded causal attention for
     newly admitted prefills + a single batched masked attention over active decode slots.
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
# 1. Standalone Model & Slot-Indexed KV Pool (Continuous Batching Copy)
# -----------------------------------------------------------------------------


@dataclass
class ContModelArgs:
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
    def llama_3_2_1b(cls, max_seq_len: int = 8192) -> "ContModelArgs":
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


class ContRMSNorm(nn.Module):
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


class SlotKVCache:
    """
    Slot-addressed GPU KV Cache Pool of shape `(n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)`.
    Independent requests dynamically claim and release individual `slot_id` rows on any iteration.
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

    def write_prefill_slot(
        self,
        layer_idx: int,
        slot_id: int,
        xk_seq: torch.Tensor,
        xv_seq: torch.Tensor,
    ) -> None:
        """Writes unpadded prefill K/V of shape `(p_len, n_kv_heads, head_dim)` into `slot_id`."""
        p_len = xk_seq.shape[0]
        if p_len > self.max_seq_len:
            raise ValueError(f"Prompt length ({p_len}) exceeds max_seq_len ({self.max_seq_len}).")
        self.k[layer_idx, slot_id, :p_len] = xk_seq
        self.v[layer_idx, slot_id, :p_len] = xv_seq

    def update_and_gather_decode(
        self,
        layer_idx: int,
        slot_ids: torch.Tensor,
        write_positions: torch.Tensor,
        max_ctx_len: int,
        xk_dec: torch.Tensor,
        xv_dec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized scatter-write of 1 new K/V token per active decode slot, followed by
        batched gather of `[0 : max_ctx_len]` across `slot_ids`.
        """
        if max_ctx_len > self.max_seq_len:
            raise ValueError(f"Decode context ({max_ctx_len}) exceeds max_seq_len ({self.max_seq_len}).")
        self.k[layer_idx, slot_ids, write_positions] = xk_dec
        self.v[layer_idx, slot_ids, write_positions] = xv_dec
        return (
            self.k[layer_idx, slot_ids, :max_ctx_len],
            self.v[layer_idx, slot_ids, :max_ctx_len],
        )

    def bytes_per_token(self) -> int:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.k.element_size()

    def get_allocated_mb(self) -> float:
        total_bytes = (self.k.numel() + self.v.numel()) * self.k.element_size()
        return total_bytes / (1024.0 * 1024.0)


@dataclass
class StepBatchDescriptor:
    """
    Precomputed per-iteration metadata shared across all 16 Transformer layers:
    - `prefill_specs`: list of `(slot_id, tok_start, tok_end, causal_mask)` for newly admitted prompts.
    - `dec_slot_ids`: 1D LongTensor of active decoding slot indices `(b_dec,)`.
    - `dec_write_pos`: 1D LongTensor of KV write positions `seq_len_j - 1` `(b_dec,)`.
    - `dec_max_ctx`: int `max(seq_len_j)` across active decode slots.
    - `dec_attn_mask`: `(b_dec, 1, 1, dec_max_ctx)` masking keys beyond each slot's `seq_len_j`.
    - `logit_token_indices`: 1D LongTensor of positions in `[0 .. T_total - 1]` to feed into `LM_Head`.
    """

    total_tokens: int
    prefill_tokens: int
    prefill_specs: List[Tuple[int, int, int, Optional[torch.Tensor]]]
    dec_slot_ids: Optional[torch.Tensor]
    dec_write_pos: Optional[torch.Tensor]
    dec_max_ctx: int
    dec_attn_mask: Optional[torch.Tensor]
    logit_token_indices: torch.Tensor


class SelectiveAttention(nn.Module):
    """Orca-style Selective Attention operating on packed `(1, T_total, dim)` activations."""

    def __init__(self, args: ContModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
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
        kv_cache: SlotKVCache,
        desc: StepBatchDescriptor,
    ) -> torch.Tensor:
        # x shape: (1, T_total, dim) — unpadded packed tokens across prefill + decode!
        _, total_toks, _ = x.shape
        prof = self.profile_ops

        with _op_scope(prof, "Q_Linear"):
            xq = self.q_proj(x).view(1, total_toks, self.n_heads, self.head_dim)
        with _op_scope(prof, "K_Linear"):
            xk = self.k_proj(x).view(1, total_toks, self.n_kv_heads, self.head_dim)
        with _op_scope(prof, "V_Linear"):
            xv = self.v_proj(x).view(1, total_toks, self.n_kv_heads, self.head_dim)

        with _op_scope(prof, "RoPE"):
            xq = _apply_rotary_emb(xq, cos, sin)
            xk = _apply_rotary_emb(xk, cos, sin)

        attn_parts: List[torch.Tensor] = []

        # 1. Newly admitted Prefill sequences (exact unpadded length per sequence)
        for slot_id, s_idx, e_idx, causal_mask in desc.prefill_specs:
            p_len = e_idx - s_idx
            xq_p = xq[:, s_idx:e_idx]  # (1, p_len, n_heads, head_dim)
            xk_p = xk[:, s_idx:e_idx]  # (1, p_len, n_kv_heads, head_dim)
            xv_p = xv[:, s_idx:e_idx]

            with _op_scope(prof, "KV_Cache_Update"):
                kv_cache.write_prefill_slot(layer_idx, slot_id, xk_p[0], xv_p[0])

            with _op_scope(prof, "Attn_Compute"):
                xk_rep = _repeat_kv(xk_p, self.n_rep).transpose(1, 2)  # (1, n_heads, p_len, head_dim)
                xv_rep = _repeat_kv(xv_p, self.n_rep).transpose(1, 2)
                xq_t = xq_p.transpose(1, 2)

                scores = torch.matmul(xq_t, xk_rep.transpose(-2, -1)) * self.scale
                if causal_mask is not None:
                    scores = scores + causal_mask
                probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)
                out_p = torch.matmul(probs, xv_rep).transpose(1, 2).contiguous().view(1, p_len, -1)
                attn_parts.append(out_p)

        # 2. Active Decode slots (batched into a single matmul across all active slots)
        if desc.dec_slot_ids is not None:
            t_pre = desc.prefill_tokens
            b_dec = desc.dec_slot_ids.shape[0]
            xq_d = xq[0, t_pre:].unsqueeze(1)  # (b_dec, 1, n_heads, head_dim)
            xk_d = xk[0, t_pre:]               # (b_dec, n_kv_heads, head_dim)
            xv_d = xv[0, t_pre:]

            with _op_scope(prof, "KV_Cache_Update"):
                k_hist, v_hist = kv_cache.update_and_gather_decode(
                    layer_idx=layer_idx,
                    slot_ids=desc.dec_slot_ids,
                    write_positions=desc.dec_write_pos,
                    max_ctx_len=desc.dec_max_ctx,
                    xk_dec=xk_d,
                    xv_dec=xv_d,
                )

            with _op_scope(prof, "Attn_Compute"):
                k_rep = _repeat_kv(k_hist, self.n_rep).transpose(1, 2)  # (b_dec, n_heads, max_ctx, head_dim)
                v_rep = _repeat_kv(v_hist, self.n_rep).transpose(1, 2)
                q_t = xq_d.transpose(1, 2)                              # (b_dec, n_heads, 1, head_dim)

                scores = torch.matmul(q_t, k_rep.transpose(-2, -1)) * self.scale
                if desc.dec_attn_mask is not None:
                    scores = scores + desc.dec_attn_mask
                probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)
                out_d = torch.matmul(probs, v_rep).transpose(1, 2).contiguous().view(1, b_dec, -1)
                attn_parts.append(out_d)

        packed_out = attn_parts[0] if len(attn_parts) == 1 else torch.cat(attn_parts, dim=1)

        with _op_scope(prof, "O_Linear"):
            return self.o_proj(packed_out)


class ContFeedForward(nn.Module):
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


class ContTransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, args: ContModelArgs):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = ContRMSNorm(args.dim, eps=args.norm_eps)
        self.self_attn = SelectiveAttention(args)
        self.post_attention_layernorm = ContRMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = ContFeedForward(args.dim, args.hidden_dim)
        self.profile_ops: bool = False

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: SlotKVCache,
        desc: StepBatchDescriptor,
    ) -> torch.Tensor:
        prof = self.profile_ops
        with _op_scope(prof, "RMSNorm_Attn"):
            norm_x1 = self.input_layernorm(x)
        x = x + self.self_attn(norm_x1, cos, sin, layer_idx=self.layer_idx, kv_cache=kv_cache, desc=desc)
        with _op_scope(prof, "RMSNorm_FFN"):
            norm_x2 = self.post_attention_layernorm(x)
        x = x + self.mlp(norm_x2)
        return x


class ContinuousBatchLlamaTransformer(nn.Module):
    def __init__(self, args: ContModelArgs):
        super().__init__()
        self.args = args
        self.profile_ops: bool = False
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([ContTransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = ContRMSNorm(args.dim, eps=args.norm_eps)
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
        flat_input_ids: torch.Tensor,
        flat_position_ids: torch.Tensor,
        kv_cache: SlotKVCache,
        desc: StepBatchDescriptor,
    ) -> torch.Tensor:
        prof = self.profile_ops
        with _op_scope(prof, "Embedding"):
            h = self.embed_tokens(flat_input_ids)  # (1, T_total, dim)

        cos = self.cos_cached[flat_position_ids].unsqueeze(2)  # (1, T_total, 1, head_dim)
        sin = self.sin_cached[flat_position_ids].unsqueeze(2)

        for layer in self.layers:
            h = layer(h, cos, sin, kv_cache=kv_cache, desc=desc)

        # Slice ONLY the sequence-final positions before RMSNorm_Final & LM_Head!
        h_emit = h[0, desc.logit_token_indices, :].unsqueeze(1)  # (num_seqs, 1, dim)

        with _op_scope(prof, "RMSNorm_Final"):
            h_emit = self.norm(h_emit)

        with _op_scope(prof, "LM_Head"):
            logits = self.lm_head(h_emit)  # (num_seqs, 1, vocab_size)
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


def load_continuous_batch_model(
    model_path_or_repo: str = "unsloth/Llama-3.2-1B-Instruct",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = 8192,
) -> Tuple[ContinuousBatchLlamaTransformer, AutoTokenizer]:
    args = ContModelArgs.llama_3_2_1b(max_seq_len=max_seq_len)
    model = ContinuousBatchLlamaTransformer(args).to(device=device, dtype=dtype)
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
# 2. Iteration-Level Continuous Batching Scheduler & Engine
# -----------------------------------------------------------------------------


@dataclass
class _ActiveSequenceState:
    req: InferenceRequest
    slot_id: int
    dispatch_time_s: float
    first_token_time_s: float
    seq_len: int  # Total valid tokens currently stored in KV slot (prompt + generated so far)
    next_input_tok: int
    token_times_s: List[float] = field(default_factory=list)
    generated_ids: List[int] = field(default_factory=list)


@dataclass
class ContinuousBatchRunOutput:
    records: List[RequestExecutionRecord]
    wall_clock_exec_s: float
    gpu_idle_time_s: float
    peak_vram_allocated_mb: float
    peak_vram_reserved_mb: float
    scheduler_diagnostics: Dict[str, Any] = field(default_factory=dict)


class ContinuousBatchEngine:
    """
    Orca-style Iteration-Level Continuous Batching Engine (`max_batch_size = B`).

    At every single forward-pass step:
    1. Checks `sim_clock_s` (or wall clock) to admit newly arrived requests into `ready_queue`.
    2. Assigns waiting requests from `ready_queue` into any free KV slots (`free_slots`).
    3. Executes ONE fused selective-batch step combining any newly admitted prefills (unpadded)
       and all currently active decode sequences into a single `(1, T_total, dim)` forward pass!
    4. Immediately retires any sequence that reaches `target_output_len` (or EOS), freeing its
       KV slot for the very next step (~0% zombie waste, ~0% prefill padding waste).
    """

    def __init__(
        self,
        model: ContinuousBatchLlamaTransformer,
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
        self.name = f"continuous_batch_b{max_batch_size}"

        self.kv_cache = SlotKVCache(
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

    def _prepare_iteration_step(
        self,
        new_prefills: List[Tuple[InferenceRequest, int]],
        active_decodes: List[_ActiveSequenceState],
    ) -> Tuple[torch.Tensor, torch.Tensor, StepBatchDescriptor]:
        tok_chunks: List[List[int]] = []
        pos_chunks: List[List[int]] = []
        logit_indices: List[int] = []
        prefill_specs: List[Tuple[int, int, int, Optional[torch.Tensor]]] = []

        cursor = 0
        for req, slot_id in new_prefills:
            p_len = req.prompt_len
            tok_chunks.append(req.prompt_token_ids)
            pos_chunks.append(list(range(p_len)))
            s_idx = cursor
            e_idx = cursor + p_len
            cursor = e_idx
            logit_indices.append(e_idx - 1)

            if p_len > 1:
                c_mask = torch.triu(
                    torch.full((p_len, p_len), float("-inf"), device=self.device, dtype=self.dtype),
                    diagonal=1,
                )
            else:
                c_mask = None
            prefill_specs.append((slot_id, s_idx, e_idx, c_mask))

        prefill_tokens = cursor

        if active_decodes:
            dec_toks = [st.next_input_tok for st in active_decodes]
            dec_pos = [st.seq_len for st in active_decodes]  # 0-based index of new token!
            tok_chunks.append(dec_toks)
            pos_chunks.append(dec_pos)

            b_dec = len(active_decodes)
            for j in range(b_dec):
                logit_indices.append(prefill_tokens + j)

            dec_slot_ids = torch.tensor([st.slot_id for st in active_decodes], dtype=torch.long, device=self.device)
            dec_write_pos = torch.tensor(dec_pos, dtype=torch.long, device=self.device)
            new_seq_lens = [st.seq_len + 1 for st in active_decodes]
            max_ctx = max(new_seq_lens)
            min_ctx = min(new_seq_lens)

            if max_ctx == min_ctx:
                dec_attn_mask = None
            else:
                seq_t = torch.tensor(new_seq_lens, dtype=torch.long, device=self.device)[:, None, None, None]
                k_idx = torch.arange(max_ctx, dtype=torch.long, device=self.device)[None, None, None, :]
                dec_attn_mask = torch.where(k_idx < seq_t, 0.0, float("-inf")).to(dtype=self.dtype)

            total_tokens = prefill_tokens + b_dec
        else:
            dec_slot_ids = None
            dec_write_pos = None
            max_ctx = 0
            dec_attn_mask = None
            total_tokens = prefill_tokens

        flat_tok_list = [t for ch in tok_chunks for t in ch]
        flat_pos_list = [p for ch in pos_chunks for p in ch]

        flat_input_ids = torch.tensor([flat_tok_list], dtype=torch.long, device=self.device)
        flat_pos_ids = torch.tensor([flat_pos_list], dtype=torch.long, device=self.device)
        logit_idx_t = torch.tensor(logit_indices, dtype=torch.long, device=self.device)

        desc = StepBatchDescriptor(
            total_tokens=total_tokens,
            prefill_tokens=prefill_tokens,
            prefill_specs=prefill_specs,
            dec_slot_ids=dec_slot_ids,
            dec_write_pos=dec_write_pos,
            dec_max_ctx=max_ctx,
            dec_attn_mask=dec_attn_mask,
            logit_token_indices=logit_idx_t,
        )
        return flat_input_ids, flat_pos_ids, desc

    @torch.inference_mode()
    def warmup(self, include_profiler: bool = True) -> None:
        self.model.set_profile_ops(False)
        b = self.max_batch_size
        dummy_reqs = [
            InferenceRequest(i, f"w_{i}", "warmup", [1] * 64, 64, 4, 0.0)
            for i in range(b)
        ]
        # Warmup prefill + batched decode
        f_ids, f_pos, desc = self._prepare_iteration_step([(r, i) for i, r in enumerate(dummy_reqs)], [])
        logits = self.model(f_ids, f_pos, kv_cache=self.kv_cache, desc=desc)
        _ = _sample_batch(logits).squeeze(-1).tolist()

        active = [
            _ActiveSequenceState(r, i, 0.0, 0.0, seq_len=64, next_input_tok=1)
            for i, r in enumerate(dummy_reqs)
        ]
        f_ids_d, f_pos_d, desc_d = self._prepare_iteration_step([], active)
        logits_d = self.model(f_ids_d, f_pos_d, kv_cache=self.kv_cache, desc=desc_d)
        _ = _sample_batch(logits_d).squeeze(-1).tolist()

        if self.device == "cuda":
            torch.cuda.synchronize()

        if include_profiler and self.device == "cuda":
            self.model.set_profile_ops(True)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            ):
                logits_p = self.model(f_ids_d, f_pos_d, kv_cache=self.kv_cache, desc=desc_d)
                _ = _sample_batch(logits_p, profile_ops=True).squeeze(-1).tolist()
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
    ) -> ContinuousBatchRunOutput:
        self.model.set_profile_ops(False)
        sorted_reqs = sorted(requests, key=lambda r: (r.arrival_time_s, r.request_id))
        n_total = len(sorted_reqs)

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        ready_queue: Deque[InferenceRequest] = deque()
        free_slots: Deque[int] = deque(range(self.max_batch_size))
        active_seqs: List[_ActiveSequenceState] = []
        records: List[RequestExecutionRecord] = []

        next_idx = 0
        sim_clock_s = 0.0
        gpu_idle_time_s = 0.0

        # Scheduler diagnostics
        total_iterations = 0
        mixed_prefill_decode_iters = 0
        active_batch_sizes: List[int] = []
        total_active_slot_steps = 0

        wall_t0 = time.perf_counter()

        def wall_now_s() -> float:
            return time.perf_counter() - wall_t0

        while len(records) < n_total:
            # 1. Check arrivals at current step boundary
            if not real_time:
                while next_idx < n_total and sorted_reqs[next_idx].arrival_time_s <= sim_clock_s:
                    ready_queue.append(sorted_reqs[next_idx])
                    next_idx += 1

                if not active_seqs and not ready_queue and next_idx < n_total:
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

                if not active_seqs and not ready_queue and next_idx < n_total:
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

            # 2. Immediately fill any free KV slots with waiting requests!
            new_prefills: List[Tuple[InferenceRequest, int]] = []
            while ready_queue and free_slots:
                req = ready_queue.popleft()
                slot_id = free_slots.popleft()
                new_prefills.append((req, slot_id))

            n_pre = len(new_prefills)
            n_dec = len(active_seqs)
            total_iterations += 1
            if n_pre > 0 and n_dec > 0:
                mixed_prefill_decode_iters += 1
            active_batch_sizes.append(n_pre + n_dec)
            total_active_slot_steps += n_dec

            if self.device == "cuda" and n_pre > 0:
                torch.cuda.synchronize()
            step_start_clock_s = wall_now_s() if real_time else sim_clock_s

            # 3. Execute ONE unified selective-batch forward pass over (new_prefills + active_seqs)
            t0 = time.perf_counter()
            flat_ids, flat_pos, desc = self._prepare_iteration_step(new_prefills, active_seqs)
            logits = self.model(flat_ids, flat_pos, kv_cache=self.kv_cache, desc=desc)
            sampled = _sample_batch(logits, temperature=temperature, top_k=top_k)
            tok_ids = sampled.squeeze(-1).tolist()  # Synchronizes stream on DtoH transfer
            t1 = time.perf_counter()

            if real_time:
                step_end_clock_s = wall_now_s()
            else:
                sim_clock_s += t1 - t0
                step_end_clock_s = sim_clock_s

            # 4. Update existing decode sequences & immediately retire completed ones!
            surviving_seqs: List[_ActiveSequenceState] = []
            for j, st in enumerate(active_seqs):
                tid = int(tok_ids[n_pre + j])
                st.seq_len += 1
                st.next_input_tok = tid
                st.token_times_s.append(step_end_clock_s)
                st.generated_ids.append(tid)

                is_done = (len(st.generated_ids) >= st.req.target_output_len) or (
                    not ignore_eos and tid in self.stop_ids
                )
                if is_done:
                    free_slots.append(st.slot_id)
                    records.append(
                        RequestExecutionRecord(
                            request_id=st.req.request_id,
                            source_id=st.req.source_id,
                            prompt_len=st.req.prompt_len,
                            target_output_len=st.req.target_output_len,
                            output_len=len(st.generated_ids),
                            arrival_time_s=st.req.arrival_time_s,
                            dispatch_time_s=st.dispatch_time_s,
                            first_token_time_s=st.first_token_time_s,
                            finish_time_s=step_end_clock_s,
                            token_times_s=st.token_times_s,
                            generated_token_ids=st.generated_ids,
                        )
                    )
                else:
                    surviving_seqs.append(st)

            # 5. Transition newly prefilled sequences into active decode pool (or retire if 1-token budget)
            for i, (req, slot_id) in enumerate(new_prefills):
                tid = int(tok_ids[i])
                is_done = (req.target_output_len <= 1) or (not ignore_eos and tid in self.stop_ids)
                if is_done:
                    free_slots.append(slot_id)
                    records.append(
                        RequestExecutionRecord(
                            request_id=req.request_id,
                            source_id=req.source_id,
                            prompt_len=req.prompt_len,
                            target_output_len=req.target_output_len,
                            output_len=1,
                            arrival_time_s=req.arrival_time_s,
                            dispatch_time_s=step_start_clock_s,
                            first_token_time_s=step_end_clock_s,
                            finish_time_s=step_end_clock_s,
                            token_times_s=[step_end_clock_s],
                            generated_token_ids=[tid],
                        )
                    )
                else:
                    surviving_seqs.append(
                        _ActiveSequenceState(
                            req=req,
                            slot_id=slot_id,
                            dispatch_time_s=step_start_clock_s,
                            first_token_time_s=step_end_clock_s,
                            seq_len=req.prompt_len,
                            next_input_tok=tid,
                            token_times_s=[step_end_clock_s],
                            generated_ids=[tid],
                        )
                    )

            active_seqs = surviving_seqs

        wall_exec_s = time.perf_counter() - wall_t0
        if self.device == "cuda":
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            peak_res_mb = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
        else:
            peak_alloc_mb, peak_res_mb = 0.0, 0.0

        records.sort(key=lambda r: r.request_id)

        diagnostics = {
            "max_batch_size": self.max_batch_size,
            "total_scheduler_iterations": total_iterations,
            "mixed_prefill_decode_iterations": mixed_prefill_decode_iters,
            "avg_active_batch_size": round(statistics.mean(active_batch_sizes), 2) if active_batch_sizes else 0.0,
            "peak_active_batch_size": max(active_batch_sizes, default=0),
            "prefill_padding_waste_pct": 0.0,
            "decode_zombie_slot_waste_pct": 0.0,
            "total_active_decode_slot_steps": total_active_slot_steps,
        }

        return ContinuousBatchRunOutput(
            records=records,
            wall_clock_exec_s=wall_exec_s,
            gpu_idle_time_s=gpu_idle_time_s,
            peak_vram_allocated_mb=peak_alloc_mb,
            peak_vram_reserved_mb=peak_res_mb,
            scheduler_diagnostics=diagnostics,
        )

    @torch.inference_mode()
    def profile_representative_phases(
        self,
        prefill_seq_len: int = 128,
        decode_context_len: int = 256,
        traces_dir: str = "chapter_5_batch_inference/profile_results/traces",
        native_trials: int = 5,
    ) -> Dict[str, Any]:
        """Profiles packed unpadded Prefill and full-slot Batched Decode (`B` slots)."""
        os.makedirs(traces_dir, exist_ok=True)
        b = self.max_batch_size
        prefill_seq_len = min(max(8, prefill_seq_len), self.max_seq_len - 16)
        decode_context_len = min(max(prefill_seq_len + 1, decode_context_len), self.max_seq_len - 4)

        pre_reqs = [
            InferenceRequest(i, f"p_{i}", "prof", [1] * prefill_seq_len, prefill_seq_len, 4, 0.0)
            for i in range(b)
        ]
        seed_reqs = [
            InferenceRequest(i, f"s_{i}", "prof", [1] * (decode_context_len - 1), decode_context_len - 1, 4, 0.0)
            for i in range(b)
        ]

        self.model.set_profile_ops(False)
        pre_walls: List[float] = []
        for _ in range(native_trials):
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            f_ids, f_pos, desc = self._prepare_iteration_step([(r, i) for i, r in enumerate(pre_reqs)], [])
            logits = self.model(f_ids, f_pos, kv_cache=self.kv_cache, desc=desc)
            _ = _sample_batch(logits).squeeze(-1).tolist()
            pre_walls.append((time.perf_counter() - t0) * 1000.0)
        native_pre_ms = statistics.median(pre_walls)

        # Seed slots up to decode_context_len - 1
        f_seed, p_seed, d_seed = self._prepare_iteration_step([(r, i) for i, r in enumerate(seed_reqs)], [])
        _ = self.model(f_seed, p_seed, kv_cache=self.kv_cache, desc=d_seed)
        if self.device == "cuda":
            torch.cuda.synchronize()

        active_dec = [
            _ActiveSequenceState(r, i, 0.0, 0.0, seq_len=decode_context_len - 1, next_input_tok=1)
            for i, r in enumerate(seed_reqs)
        ]
        dec_walls: List[float] = []
        for _ in range(native_trials):
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            f_ids_d, f_pos_d, desc_d = self._prepare_iteration_step([], active_dec)
            logits_d = self.model(f_ids_d, f_pos_d, kv_cache=self.kv_cache, desc=desc_d)
            _ = _sample_batch(logits_d).squeeze(-1).tolist()
            dec_walls.append((time.perf_counter() - t0) * 1000.0)
        native_dec_ms = statistics.median(dec_walls)

        self.model.set_profile_ops(True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        pre_trace = os.path.join(traces_dir, f"continuous_b{b}_prefill_trace.json")
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof_pre:
            with torch.profiler.record_function("step_prefill"):
                f_ids, f_pos, desc = self._prepare_iteration_step([(r, i) for i, r in enumerate(pre_reqs)], [])
                logits = self.model(f_ids, f_pos, kv_cache=self.kv_cache, desc=desc)
                _ = _sample_batch(logits, profile_ops=True).squeeze(-1).tolist()
        if self.device == "cuda":
            torch.cuda.synchronize()
        prof_pre.export_chrome_trace(pre_trace)

        self.model.set_profile_ops(False)
        _ = self.model(f_seed, p_seed, kv_cache=self.kv_cache, desc=d_seed)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.model.set_profile_ops(True)

        dec_trace = os.path.join(traces_dir, f"continuous_b{b}_decode_trace.json")
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof_dec:
            with torch.profiler.record_function("step_decode"):
                f_ids_d, f_pos_d, desc_d = self._prepare_iteration_step([], active_dec)
                logits_d = self.model(f_ids_d, f_pos_d, kv_cache=self.kv_cache, desc=desc_d)
                _ = _sample_batch(logits_d, profile_ops=True).squeeze(-1).tolist()
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
