"""
Llama-3.2-1B inference with fine-grained PyTorch Profiler instrumentation.

Provides deep operator time decomposition:
- Embedding
- RMSNorm_Attn (first RMSNorm in block)
- QKV_Linear (Q, K, V linear projections)
- RoPE (rotary position embedding rotation)
- Attn_Compute (GQA repeat, attention dot products, mask, softmax, PV)
- O_Linear (attention output projection)
- RMSNorm_FFN (second RMSNorm in block)
- FFN_Gate_Up_Linear (gate_proj & up_proj)
- FFN_SiLU_Mul (SiLU activation & elementwise multiply)
- FFN_Down_Linear (down_proj)
- RMSNorm_Final (final normalization before LM_HEAD)
- LM_Head (vocab projection)
- Sampling (argmax / multinomial + GPU-to-CPU sync)
- Tokenizer_Decode (string detokenization)

Supports sampled profiling (e.g. step 0, every 100 steps, and last step)
to profile up to 2048+ tokens with zero memory bloat.
"""

import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from transformers import AutoTokenizer

from llama_inference import (
    ModelArgs,
    RMSNorm,
    precompute_rope_freqs,
    apply_rotary_emb,
    repeat_kv,
    load_hf_safetensors,
)
from profile_visualizer import (
    extract_single_token_metric,
    extract_timeline_from_trace,
    render_terminal_dashboard,
    generate_html_dashboard,
    save_json_metrics,
)


# -----------------------------------------------------------------------------
# 1. Fine-Grained Profiled Architecture
# -----------------------------------------------------------------------------

class ProfiledAttention(nn.Module):
    """
    Multi-Head Grouped-Query Attention with fine-grained sub-operator profiling:
    - QKV_Linear
    - RoPE
    - Attn_Compute (GQA repeat, QK^T, mask, softmax, PV)
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # 1. Linear Projections
        with torch.profiler.record_function("QKV_Linear"):
            with torch.profiler.record_function("Q_Linear"):
                xq = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)
            with torch.profiler.record_function("K_Linear"):
                xk = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
            with torch.profiler.record_function("V_Linear"):
                xv = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # 2. RoPE
        with torch.profiler.record_function("RoPE"):
            xq = apply_rotary_emb(xq, cos, sin)
            xk = apply_rotary_emb(xk, cos, sin)

        # 3. Attention Compute (GQA repeat, QK^T, mask, softmax, PV)
        with torch.profiler.record_function("Attn_Compute"):
            xk = repeat_kv(xk, self.n_rep)
            xv = repeat_kv(xv, self.n_rep)

            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)

            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if seqlen > 1:
                mask = torch.full((seqlen, seqlen), float("-inf"), device=scores.device, dtype=scores.dtype)
                mask = torch.triu(mask, diagonal=1)
                scores = scores + mask

            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)
            output = torch.matmul(probs, xv)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 4. Output Projection
        with torch.profiler.record_function("O_Linear"):
            out = self.o_proj(output)
        return out


class ProfiledFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network with fine-grained sub-operator profiling:
    - FFN_Gate_Up_Linear
    - FFN_SiLU_Mul
    - FFN_Down_Linear
    """
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
    """
    Transformer block tracking RMSNorm_Attn, ProfiledAttention,
    RMSNorm_FFN, and ProfiledFeedForward.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.self_attn = ProfiledAttention(args)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = ProfiledFeedForward(args.dim, args.hidden_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connection for attention
        with torch.profiler.record_function("RMSNorm_Attn"):
            norm_x1 = self.input_layernorm(x)
        attn_out = self.self_attn(norm_x1, cos, sin)
        x = x + attn_out

        # Pre-norm residual connection for feed-forward
        with torch.profiler.record_function("RMSNorm_FFN"):
            norm_x2 = self.post_attention_layernorm(x)
        ffn_out = self.mlp(norm_x2)
        x = x + ffn_out
        return x


class ProfiledTransformer(nn.Module):
    """
    Top-level Transformer tracking Embedding, Layers, RMSNorm_Final, and LM_Head.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([ProfiledTransformerBlock(args) for _ in range(args.n_layers)])
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        with torch.profiler.record_function("Embedding"):
            h = self.embed_tokens(input_ids)

        cos = self.cos_cached[:seqlen].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seqlen].unsqueeze(0).unsqueeze(2)

        for layer in self.layers:
            h = layer(h, cos, sin)

        with torch.profiler.record_function("RMSNorm_Final"):
            h = self.norm(h)

        with torch.profiler.record_function("LM_Head"):
            logits = self.lm_head(h[:, [-1], :])
        return logits


# -----------------------------------------------------------------------------
# 2. Sampled Autoregressive Generation Loop
# -----------------------------------------------------------------------------

@torch.inference_mode()
def generate_profiled(
    model: ProfiledTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    profile: bool = True,
    profile_interval: int = 100,
    profile_output_dir: str = "profile_results",
    ignore_eos: bool = False,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    device: str = "cuda",
    use_chat_template: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generates tokens with targeted sampled profiling:
    Profiles step 0, every `profile_interval` steps (e.g. 100, 200, ...), and the final step.
    Un-sampled steps run natively with 0 profiler overhead.
    """
    # 1. Format prompt
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    print(f"\n--- Prompt --- \n{prompt}")
    print(f"\n--- Model Response (streaming tokens) ---")

    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    for special_tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(special_tok)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            stop_token_ids.add(tok_id)

    curr_ids = input_ids
    generated_token_ids = []
    token_records: List[Dict[str, Any]] = []

    for step in range(max_new_tokens):
        # Sampling condition: step 0, every profile_interval steps, or final step
        is_sample = profile and (
            (step == 0) or (step % profile_interval == 0) or (step == max_new_tokens - 1)
        )

        if device == "cuda":
            torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        if is_sample:
            # Profile only this isolated step
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as prof_step:
                with torch.profiler.record_function(f"token_{step}"):
                    logits = model(curr_ids)
                    next_token_logits = logits[:, -1, :]

                    with torch.profiler.record_function("Sampling"):
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

            if device == "cuda":
                torch.cuda.synchronize()
            t_step_end = time.perf_counter()
            native_latency_ms = (t_step_end - t_step_start) * 1000.0

            # Detokenize
            token_str = tokenizer.decode([token_val], clean_up_tokenization_spaces=False)
            print(token_str, end="", flush=True)

            # Export trace for timeline visualization
            traces_dir = os.path.join(profile_output_dir, "traces")
            os.makedirs(traces_dir, exist_ok=True)
            trace_file = os.path.join(traces_dir, f"step_{step}_trace.json")
            prof_step.export_chrome_trace(trace_file)

            # Extract fine-grained metric and timeline for this step
            step_record = extract_single_token_metric(prof_step, step, token_val, token_str)
            seq_len = curr_ids.shape[-1]
            step_timeline = extract_timeline_from_trace(
                trace_path=trace_file,
                step_idx=step,
                token_id=token_val,
                token_text=token_str,
                seq_len=seq_len,
            )
            step_record["timeline"] = step_timeline
            if "three_metrics" in step_timeline:
                step_record["three_metrics"] = step_timeline["three_metrics"]
                step_record["total_latency_ms"] = step_timeline["three_metrics"]["total_latency_ms"]
            else:
                step_record["total_latency_ms"] = round(native_latency_ms, 2)
            step_record["is_sampled"] = True
            token_records.append(step_record)
        else:
            # Un-profiled native execution
            logits = model(curr_ids)
            next_token_logits = logits[:, -1, :]

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
            if device == "cuda":
                torch.cuda.synchronize()
            t_step_end = time.perf_counter()
            native_latency_ms = (t_step_end - t_step_start) * 1000.0

            token_str = tokenizer.decode([token_val], clean_up_tokenization_spaces=False)
            print(token_str, end="", flush=True)

            step_record = {
                "step": step,
                "token_id": token_val,
                "token_text": token_str,
                "total_latency_ms": round(native_latency_ms, 2),
                "is_sampled": False,
                "breakdown": None,
                "three_metrics": None,
                "timeline": None,
            }
            token_records.append(step_record)

        if not ignore_eos and token_val in stop_token_ids:
            # If early stop and last step wasn't profiled, we still captured up to this point
            break

        generated_token_ids.append(token_val)
        curr_ids = torch.cat([curr_ids, next_token_id], dim=-1)

    print("\n-----------------------------------------")
    full_response = tokenizer.decode(generated_token_ids, clean_up_tokenization_spaces=False)
    return full_response, token_records


# -----------------------------------------------------------------------------
# 3. Main Entrypoint & CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Llama-3.2-1B fine-grained sampled profiling.")
    parser.add_argument("--prompt", type=str, default="Write a comprehensive guide on quantum computing principles.", help="Input text prompt")
    parser.add_argument("--model_path", type=str, default="unsloth/Llama-3.2-1B-Instruct", help="HuggingFace model ID or local directory")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum number of tokens to generate (e.g. 500, 2048)")
    parser.add_argument("--profile_interval", type=int, default=100, help="Sample profiling interval (e.g. every 100 steps + step 0 and last step)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy decoding)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--profile", action="store_true", default=True, help="Enable fine-grained sampled profiling")
    parser.add_argument("--no_profile", dest="profile", action="store_false", help="Disable profiling")
    parser.add_argument("--profile_output_dir", type=str, default="profile_results", help="Directory for profile outputs")
    parser.add_argument("--warmup", action="store_true", default=True, help="Run 1-step warmup before profiling")
    parser.add_argument("--ignore_eos", action="store_true", default=False, help="Ignore EOS/EOT tokens to guarantee generating up to max_new_tokens for benchmarking")
    args = parser.parse_args()

    print(f"[*] Running on device: {args.device} ({torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'})")

    # 1. Model Architecture with Fine-Grained Instrumentation
    model_args = ModelArgs.llama_3_2_1b()
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = ProfiledTransformer(model_args).to(device=args.device, dtype=dtype)
    model.eval()

    # 2. Load Safetensors Weights
    load_hf_safetensors(model, model_path_or_repo=args.model_path, device=args.device, dtype=dtype)

    # 3. Load Tokenizer
    tok_repo = args.model_path if os.path.exists(args.model_path) else ("unsloth/Llama-3.2-1B-Instruct" if "meta-llama" in args.model_path and "HF_TOKEN" not in os.environ else args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_repo)

    # 4. Warmup
    if args.profile and args.warmup and args.device == "cuda":
        print("[*] Performing 1-step model warmup to eliminate CUDA context overhead...")
        with torch.no_grad():
            dummy = torch.tensor([[1, 2]], device=args.device)
            _ = model(dummy)
            torch.cuda.synchronize()

    # 5. Generation with Sampled Profiling
    os.makedirs(args.profile_output_dir, exist_ok=True)
    print(f"[*] Starting generation (max_new_tokens={args.max_new_tokens}, profile_interval={args.profile_interval})...")

    response, token_records = generate_profiled(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        profile=args.profile,
        profile_interval=args.profile_interval,
        profile_output_dir=args.profile_output_dir,
        ignore_eos=args.ignore_eos,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )

    if args.profile and token_records:
        sampled_records = [r for r in token_records if r.get("is_sampled", True) and r.get("breakdown")]
        print(f"\n[*] Rendering dashboard for {len(token_records)} tokens ({len(sampled_records)} sampled checkpoints)...")
        render_terminal_dashboard(token_records, prompt=args.prompt)

        timeline_records = {
            r["step"]: r["timeline"] for r in token_records if "timeline" in r and r["timeline"]
        }

        html_file = os.path.join(args.profile_output_dir, "profile_dashboard.html")
        generate_html_dashboard(token_records, prompt=args.prompt, output_file=html_file, timeline_records=timeline_records)

        json_file = os.path.join(args.profile_output_dir, "token_metrics.json")
        save_json_metrics(token_records, prompt=args.prompt, output_file=json_file, timeline_records=timeline_records)

        print(f"\n[✓] Fine-grained sampled profiling outputs saved to: {os.path.abspath(args.profile_output_dir)}")


if __name__ == "__main__":
    main()
