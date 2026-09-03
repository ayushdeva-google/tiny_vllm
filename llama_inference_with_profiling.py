"""
Llama-3.2-1B inference with PyTorch Profiler instrumentation.

Extends the baseline architecture with fine-grained torch.profiler.record_function
annotations to measure:
- Per-output-token latency (TPOT)
- Op-time composition per token (RMSNorm, Attention, FFN, LM Head, Sampling, Tokenizer)

Visualizations and metrics are handled via profile_visualizer.py.
"""

import os
import math
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
    Attention,
    FeedForward,
    precompute_rope_freqs,
    load_hf_safetensors,
)
from profile_visualizer import (
    extract_token_metrics,
    render_terminal_dashboard,
    generate_html_dashboard,
    save_json_metrics,
)


# -----------------------------------------------------------------------------
# 1. Instrumented Transformer Architecture
# -----------------------------------------------------------------------------

class ProfiledTransformerBlock(nn.Module):
    """
    Transformer block instrumented with torch.profiler.record_function
    to capture RMSNorm, Attention, and FFN times.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.self_attn = Attention(args)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = FeedForward(args.dim, args.hidden_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connection for attention
        with torch.profiler.record_function("RMSNorm"):
            norm_x1 = self.input_layernorm(x)
        with torch.profiler.record_function("Attention"):
            attn_out = self.self_attn(norm_x1, cos, sin)
        x = x + attn_out

        # Pre-norm residual connection for feed-forward
        with torch.profiler.record_function("RMSNorm"):
            norm_x2 = self.post_attention_layernorm(x)
        with torch.profiler.record_function("FFN"):
            ffn_out = self.mlp(norm_x2)
        x = x + ffn_out
        return x


class ProfiledTransformer(nn.Module):
    """
    Top-level Transformer instrumented with Embedding, RMSNorm, and LM_Head markers.
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

        with torch.profiler.record_function("RMSNorm"):
            h = self.norm(h)

        with torch.profiler.record_function("LM_Head"):
            logits = self.lm_head(h[:, [-1], :])
        return logits


# -----------------------------------------------------------------------------
# 2. Instrumented Autoregressive Generation Loop
# -----------------------------------------------------------------------------

from contextlib import nullcontext

@torch.inference_mode()
def generate_profiled(
    model: ProfiledTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    max_profile_tokens: int = 20,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    device: str = "cuda",
    use_chat_template: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Instrumented generation loop that demarcates every token decode step
    and records token-level metadata.
    """
    # 1. Format prompt
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = prompt

    # 2. Tokenize prompt into input_ids tensor
    with torch.profiler.record_function("Tokenizer_Encode"):
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
    token_details: List[Dict[str, Any]] = []

    # 3. Generation loop with per-token profiling scopes
    for step in range(max_new_tokens):
        should_profile = step < max_profile_tokens
        token_scope = torch.profiler.record_function(f"token_{step}") if should_profile else nullcontext()
        with token_scope:
            # Model forward
            logits = model(curr_ids)
            next_token_logits = logits[:, -1, :]

            # Sampling
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

                token_val = next_token_id.item()  # GPU-CPU synchronization

            if token_val in stop_token_ids:
                break

            generated_token_ids.append(token_val)
            curr_ids = torch.cat([curr_ids, next_token_id], dim=-1)

            # Tokenizer decode
            with torch.profiler.record_function("Tokenizer_Decode"):
                token_str = tokenizer.decode([token_val], clean_up_tokenization_spaces=False)
                print(token_str, end="", flush=True)

            token_details.append({
                "step": step,
                "id": token_val,
                "text": token_str,
            })

    print("\n-----------------------------------------")
    full_response = tokenizer.decode(generated_token_ids, clean_up_tokenization_spaces=False)
    return full_response, token_details


# -----------------------------------------------------------------------------
# 3. Main Entrypoint & CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Llama-3.2-1B inference with PyTorch Profiler and Op Breakdown.")
    parser.add_argument("--prompt", type=str, default="Explain why the sky is blue in 2 sentences.", help="Input text prompt")
    parser.add_argument("--model_path", type=str, default="unsloth/Llama-3.2-1B-Instruct", help="HuggingFace model ID or local directory")
    parser.add_argument("--max_new_tokens", type=int, default=15, help="Maximum number of tokens to generate")
    parser.add_argument("--max_profile_tokens", type=int, default=20, help="Safety cap: maximum number of tokens to profile to prevent memory explosion")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy decoding)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--profile", action="store_true", default=True, help="Enable PyTorch Profiler (default: True)")
    parser.add_argument("--no_profile", dest="profile", action="store_false", help="Disable PyTorch Profiler")
    parser.add_argument("--profile_output_dir", type=str, default="profile_results", help="Directory for profile outputs")
    parser.add_argument("--save_trace", action="store_true", default=False, help="Export Perfetto/Chrome JSON trace (default: False to save disk/RAM)")
    parser.add_argument("--warmup", action="store_true", default=True, help="Run 1-step warmup before profiling to avoid initial CUDA alloc spike")
    args = parser.parse_args()

    print(f"[*] Running on device: {args.device} ({torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'})")

    # 1. Initialize custom Model Architecture
    model_args = ModelArgs.llama_3_2_1b()
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = ProfiledTransformer(model_args).to(device=args.device, dtype=dtype)
    model.eval()

    # 2. Load Safetensors Weights directly
    load_hf_safetensors(model, model_path_or_repo=args.model_path, device=args.device, dtype=dtype)

    # 3. Load HuggingFace Tokenizer
    tok_repo = args.model_path if os.path.exists(args.model_path) else ("unsloth/Llama-3.2-1B-Instruct" if "meta-llama" in args.model_path and "HF_TOKEN" not in os.environ else args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_repo)

    # 4. Optional Warmup
    if args.profile and args.warmup and args.device == "cuda":
        print("[*] Performing 1-step model warmup to eliminate CUDA initialization overhead...")
        with torch.no_grad():
            dummy = torch.tensor([[1, 2]], device=args.device)
            _ = model(dummy)
            torch.cuda.synchronize()

    # 5. Generation (Profiled vs Normal)
    if args.profile and args.device == "cuda":
        os.makedirs(args.profile_output_dir, exist_ok=True)
        print(f"[*] Profiling generation with PyTorch Profiler...")

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            response, token_details = generate_profiled(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                max_profile_tokens=args.max_profile_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=args.device,
            )

        torch.cuda.synchronize()

        # Extract structured metrics
        print("\n[*] Parsing profiler events and aggregating per-token op times...")
        token_records = extract_token_metrics(prof, token_details=token_details)

        # 1. Render Rich Terminal Dashboard
        render_terminal_dashboard(token_records, prompt=args.prompt)

        # 2. Export Standalone HTML Dashboard
        html_file = os.path.join(args.profile_output_dir, "profile_dashboard.html")
        generate_html_dashboard(token_records, prompt=args.prompt, output_file=html_file)

        # 3. Export JSON Metrics
        json_file = os.path.join(args.profile_output_dir, "token_metrics.json")
        save_json_metrics(token_records, prompt=args.prompt, output_file=json_file)

        # 4. Export Chrome Trace
        if args.save_trace:
            trace_file = os.path.join(args.profile_output_dir, "llama_trace.json")
            prof.export_chrome_trace(trace_file)
            print(f"[*] Perfetto / Chrome trace exported at: {trace_file}")

        print(f"\n[✓] All profiling outputs saved to: {os.path.abspath(args.profile_output_dir)}")
    else:
        generate_profiled(
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

