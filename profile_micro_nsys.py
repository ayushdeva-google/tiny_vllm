"""
Micro-benchmark inference script instrumented with NVTX and cudaProfilerApi.
Profiles an isolated single decode step to capture exact hardware execution
with Nsight Systems (nsys) without trace bloat or memory overhead.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from llama_inference import ModelArgs, load_hf_safetensors
from llama_inference_with_profiling import ProfiledTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Explain the principle of superposition in quantum mechanics.")
    parser.add_argument("--warmup_tokens", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="profile_results/nsys_micro")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[*] Loading LLaMA-3.2-1B-Instruct model and weights...")
    model_args = ModelArgs.llama_3_2_1b()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = ProfiledTransformer(model_args).to(device=device, dtype=dtype)
    load_hf_safetensors(model, device=device, dtype=dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    curr_ids = inputs["input_ids"].to(device)

    print(f"[*] Prompt length: {curr_ids.shape[-1]} tokens")
    print(f"[*] Starting warm-up ({args.warmup_tokens} tokens)...")

    # Warmup phase (prefill + warmup decode steps)
    with torch.no_grad():
        for _ in range(args.warmup_tokens):
            logits = model(curr_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)
        torch.cuda.synchronize()

    print("[*] Warm-up complete. Preparing targeted isolated decode token profile...")

    # Targeted Isolated Profile Step
    torch.cuda.synchronize()

    # 1. Start Nsight Systems Capture Range via CUDA Profiler API
    torch.cuda.cudart().cudaProfilerStart()

    # 2. Emit NVTX ranges for Nsys
    with torch.autograd.profiler.emit_nvtx(record_shapes=True):
        with torch.no_grad():
            logits = model(curr_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            token_val = next_token.item()

    torch.cuda.synchronize()

    # 3. Stop Nsight Systems Capture Range
    torch.cuda.cudart().cudaProfilerStop()

    print(f"[✓] Isolated single-token profile complete!")
    print(f"[✓] Token generated: {repr(tokenizer.decode([token_val]))}")


if __name__ == "__main__":
    main()
