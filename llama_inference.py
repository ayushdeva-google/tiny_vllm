"""
Minimal, standalone inference script for meta-llama/Llama-3.2-1B-Instruct from scratch.

Features:
- Pure PyTorch & Python implementation inspired by Andrej Karpathy's llama2.c style.
- Custom Transformer architecture: RMSNorm, RoPE (Llama-3 frequency scaling), GQA, SwiGLU.
- Direct safetensors weight loading from HuggingFace checkpoint format.
- Naive autoregressive generation loop without KV cache (recomputes full sequence attention every step).
- Minimalist single-file design suitable for profiling with PyTorch Profiler and NVIDIA Nsight Systems (nsys).
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from transformers import AutoTokenizer


# -----------------------------------------------------------------------------
# 1. Model Configuration & Architecture
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
    # Llama 3 / 3.2 RoPE frequency scaling parameters
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
    """Root Mean Square Layer Normalization (RMSNorm)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability, then cast back to input dtype
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
    Precompute cosine and sine frequency tables for Rotary Position Embeddings (RoPE).
    Includes exact support for Llama 3/3.2 wavelength-based frequency scaling.
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

    # Compute outer product over position indices: (max_seq_len, dim // 2)
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)

    # Duplicate frequencies for HuggingFace RoPE layout: (max_seq_len, dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies HuggingFace-compatible Rotary Position Embedding to tensor x."""
    # x shape: (bsz, seqlen, num_heads, head_dim)
    # cos/sin shape broadcasted: (1, seqlen, 1, head_dim)
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


class Attention(nn.Module):
    """Multi-Head Grouped-Query Attention (GQA) layer."""
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # 1. Project Q, K, V
        xq = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # 2. Apply RoPE to Query and Key
        xq = apply_rotary_emb(xq, cos, sin)
        xk = apply_rotary_emb(xk, cos, sin)

        # 3. Repeat KV heads for GQA
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 4. Transpose to (bsz, n_heads, seqlen, head_dim) for batch matrix multiplication
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 5. Scaled dot-product attention
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 6. Apply causal mask (prevent attending to future tokens)
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=scores.device, dtype=scores.dtype)
            mask = torch.triu(mask, diagonal=1)
            scores = scores + mask

        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)
        output = torch.matmul(probs, xv)  # (bsz, n_heads, seqlen, head_dim)

        # 7. Reshape back to (bsz, seqlen, dim) and project output
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single Transformer block containing RMSNorm, GQA Attention, and SwiGLU MLP."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.self_attn = Attention(args)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = FeedForward(args.dim, args.hidden_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connection for attention
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        # Pre-norm residual connection for feed-forward
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Transformer(nn.Module):
    """Top-level Llama-3.2 Causal Transformer architecture."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Weight tying: Llama-3.2-1B shares lm_head weights with token embeddings
        if args.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE frequency tables
        cos, sin = precompute_rope_freqs(
            dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            theta=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor, return_all_logits: bool = False) -> torch.Tensor:
        """
        Vanilla forward pass over input_ids: (bsz, seqlen).
        If return_all_logits is True: returns logits for all tokens (bsz, seqlen, vocab_size).
        Otherwise returns logits for the last token position: (bsz, 1, vocab_size).
        """
        bsz, seqlen = input_ids.shape
        h = self.embed_tokens(input_ids)

        # Slice precomputed RoPE tables for current sequence length and unsqueeze for broadcasting
        cos = self.cos_cached[:seqlen].unsqueeze(0).unsqueeze(2)  # (1, seqlen, 1, head_dim)
        sin = self.sin_cached[:seqlen].unsqueeze(0).unsqueeze(2)

        for layer in self.layers:
            h = layer(h, cos, sin)

        h = self.norm(h)

        # Compute logits for all positions or only the final position
        if return_all_logits:
            logits = self.lm_head(h)  # shape: (bsz, seqlen, vocab_size)
        else:
            logits = self.lm_head(h[:, [-1], :])  # shape: (bsz, 1, vocab_size)
        return logits


# -----------------------------------------------------------------------------
# 2. Safetensors Weight Loading
# -----------------------------------------------------------------------------

def load_hf_safetensors(
    model: Transformer,
    model_path_or_repo: str = "meta-llama/Llama-3.2-1B-Instruct",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Transformer:
    """
    Loads weights directly from HuggingFace safetensors format into custom Transformer model.
    Accepts either a local path or a HuggingFace repo ID.
    """
    safetensors_file = None

    # Case 1: Direct file path
    if os.path.isfile(model_path_or_repo) and model_path_or_repo.endswith(".safetensors"):
        safetensors_file = model_path_or_repo
    # Case 2: Local directory containing model.safetensors
    elif os.path.isdir(model_path_or_repo):
        candidate = os.path.join(model_path_or_repo, "model.safetensors")
        if os.path.exists(candidate):
            safetensors_file = candidate
    # Case 3: HuggingFace hub repo ID
    if safetensors_file is None:
        try:
            from huggingface_hub import hf_hub_download
            print(f"[*] Downloading weights from HuggingFace Hub: {model_path_or_repo}...")
            token = os.environ.get("HF_TOKEN", None)
            try:
                safetensors_file = hf_hub_download(
                    repo_id=model_path_or_repo,
                    filename="model.safetensors",
                    token=token,
                )
            except Exception as e:
                # If meta-llama is gated and no token provided, try unsloth public mirror
                if "meta-llama" in model_path_or_repo and token is None:
                    fallback_repo = "unsloth/Llama-3.2-1B-Instruct"
                    print(f"[!] Access to '{model_path_or_repo}' requires HF_TOKEN. Falling back to '{fallback_repo}'...")
                    safetensors_file = hf_hub_download(
                        repo_id=fallback_repo,
                        filename="model.safetensors",
                    )
                else:
                    raise e
        except ImportError:
            raise RuntimeError("huggingface_hub is required to download models from repo IDs. Run: pip install huggingface_hub")

    print(f"[*] Loading weights from: {safetensors_file}")
    raw_state_dict = safetensors.torch.load_file(safetensors_file)

    # Map HuggingFace tensor names to our custom module parameter names
    # e.g., 'model.layers.0.self_attn.q_proj.weight' -> 'layers.0.self_attn.q_proj.weight'
    #       'model.embed_tokens.weight' -> 'embed_tokens.weight'
    #       'model.norm.weight' -> 'norm.weight'
    custom_state_dict = {}
    for key, tensor in raw_state_dict.items():
        mapped_key = key
        if mapped_key.startswith("model."):
            mapped_key = mapped_key[len("model."):]
        custom_state_dict[mapped_key] = tensor.to(dtype=dtype, device=device)

    # Load parameters into custom model
    missing, unexpected = model.load_state_dict(custom_state_dict, strict=False)

    # Note: 'lm_head.weight' is missing if weight tying is enabled, which is expected
    if missing:
        if missing == ["lm_head.weight"] and model.args.tie_word_embeddings:
            print("[*] Weight tying confirmed: lm_head shares weights with embed_tokens.")
        else:
            print(f"[!] Warning: Missing keys: {missing}")
    if unexpected:
        print(f"[!] Warning: Unexpected keys: {unexpected}")

    print("[*] Weights successfully mapped and loaded.")
    return model


# -----------------------------------------------------------------------------
# 3. Naive Autoregressive Generation Loop
# -----------------------------------------------------------------------------

@torch.inference_mode()
def generate(
    model: Transformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    device: str = "cuda",
    use_chat_template: bool = True,
) -> str:
    """
    Naive autoregressive token generation loop.
    Recomputes attention over the entire sequence history on every step (no KV cache).
    """
    # 1. Format prompt
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = prompt

    # 2. Tokenize prompt into input_ids tensor
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    print(f"\n--- Prompt --- \n{prompt}")
    print(f"\n--- Model Response (streaming tokens) ---")

    # Stop tokens for Llama-3.2-Instruct
    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    for special_tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(special_tok)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            stop_token_ids.add(tok_id)

    curr_ids = input_ids
    generated_token_ids = []

    # 3. Generation loop
    for step in range(max_new_tokens):
        # Forward pass: recomputes all token activations from scratch every step
        logits = model(curr_ids)  # (1, 1, vocab_size)
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

        # 4. Sampling / Greedy selection
        if temperature == 0.0:
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_token_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        token_val = next_token_id.item()
        if token_val in stop_token_ids:
            break

        generated_token_ids.append(token_val)

        # 5. Append token to running sequence
        curr_ids = torch.cat([curr_ids, next_token_id], dim=-1)

        # Stream decoded token
        token_str = tokenizer.decode([token_val], clean_up_tokenization_spaces=False)
        print(token_str, end="", flush=True)

    print("\n-----------------------------------------")
    return tokenizer.decode(generated_token_ids, clean_up_tokenization_spaces=False)


# -----------------------------------------------------------------------------
# 4. Main Entrypoint & CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Minimal standalone Llama-3.2-1B-Instruct inference loop from scratch.")
    parser.add_argument("--prompt", type=str, default="Explain why the sky is blue in 2 sentences.", help="Input text prompt")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID or local directory")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy decoding)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--profile", action="store_true", help="Profile generation using PyTorch Profiler")
    args = parser.parse_args()

    print(f"[*] Running on device: {args.device} ({torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'})")

    # 1. Initialize custom Model Architecture
    model_args = ModelArgs.llama_3_2_1b()
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = Transformer(model_args).to(device=args.device, dtype=dtype)
    model.eval()

    # 2. Load Safetensors Weights directly
    load_hf_safetensors(model, model_path_or_repo=args.model_path, device=args.device, dtype=dtype)

    # 3. Load HuggingFace Tokenizer (used only for encoding/decoding text)
    tok_repo = args.model_path if os.path.exists(args.model_path) else ("unsloth/Llama-3.2-1B-Instruct" if "meta-llama" in args.model_path and "HF_TOKEN" not in os.environ else args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_repo)

    # 4. Autoregressive Generation
    if args.profile and args.device == "cuda":
        print("[*] Profiling generation with PyTorch Profiler...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            generate(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=args.device,
            )
        print("\n--- Profiler Table (Top CUDA Time Operations) ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    else:
        generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )


if __name__ == "__main__":
    main()
