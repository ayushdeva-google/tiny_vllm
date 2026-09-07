#!/usr/bin/env python3
"""
Lightweight Quality Evaluation Suite for tiny_vllm (Llama-3.2-1B-Instruct).

Designed to run in ~1-2 minutes to evaluate the quality impact of latency optimizations
(quantization, KV-cache, kernel fusions, torch.compile, etc.).

Three Pillars:
1. WikiText-2 Perplexity (PPL): Continuous next-token prediction loss over 5 chunks of 1024 tokens.
   Hyper-sensitive to quantization drift, precision loss, and scaling errors.
2. Greedy Parity vs HuggingFace: Token-by-token exact match & logit cosine similarity across
   10 diverse prompts. Exercises the decode loop, KV-cache, and EOS termination.
3. ARC-Easy Benchmark (100 Qs): Standard multiple-choice reasoning accuracy (Baseline: ~80%).
   Tests if lossy optimizations (e.g. FP8/INT4) actually degrade model intelligence.
"""

import os
import time
import math
import json
import argparse
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from llama_inference import Transformer, ModelArgs, load_hf_safetensors


# -----------------------------------------------------------------------------
# Curated Prompts for Parity Evaluation
# -----------------------------------------------------------------------------

PARITY_PROMPTS = [
    "Write a Python function to check if a string is a palindrome.",
    "If a car travels at 60 mph for 2.5 hours, how far does it travel? Explain step by step.",
    "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?",
    "Rewrite this SQL query with an INNER JOIN: SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);",
    "Describe quantum computing to a 10-year-old in two simple sentences.",
    "What is the capital of Australia, and what is its approximate population?",
    'Extract name and age into JSON from: "Alice Smith is a 28-year-old data scientist living in Boston."',
    "Explain the core concept of Newton's third law of motion.",
    "List five common fruits in reverse alphabetical order.",
    "Give the exact Linux find command to search for files larger than 100MB in the current directory.",
]


def resolve_data_path(rel_path: str) -> str:
    """Resolves data path whether running from repo root or chapter_1/."""
    if os.path.exists(rel_path):
        return rel_path
    parent_candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", rel_path)
    if os.path.exists(parent_candidate):
        return os.path.abspath(parent_candidate)
    return rel_path


# -----------------------------------------------------------------------------
# Pillar 1: WikiText-2 Perplexity (PPL)
# -----------------------------------------------------------------------------

def evaluate_wikitext_ppl(
    model: Transformer,
    tokenizer: AutoTokenizer,
    data_path: str = "data/wikitext_2_test.txt",
    num_chunks: int = 5,
    chunk_size: int = 1024,
    device: str = "cuda",
) -> Tuple[float, float, float]:
    """
    Computes teacher-forced perplexity over fixed-length chunks of WikiText-2.
    Returns: (perplexity, avg_cross_entropy_loss, elapsed_time)
    """
    start_time = time.perf_counter()
    data_path = resolve_data_path(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"WikiText test file not found at {data_path}. Please download it first.")

    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokens = tokenizer.encode(raw_text, return_tensors="pt")[0]
    total_tokens_needed = num_chunks * chunk_size
    if len(tokens) < total_tokens_needed:
        raise ValueError(f"Text too short ({len(tokens)} tokens) for {num_chunks}x{chunk_size} tokens.")

    losses = []
    with torch.inference_mode():
        for i in range(num_chunks):
            chunk = tokens[i * chunk_size : (i + 1) * chunk_size].unsqueeze(0).to(device)
            # Forward pass over full sequence
            logits = model(chunk, return_all_logits=True)  # (1, chunk_size, vocab_size)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    elapsed = time.perf_counter() - start_time
    return ppl, avg_loss, elapsed


# -----------------------------------------------------------------------------
# Pillar 2: Greedy Parity vs HuggingFace Reference
# -----------------------------------------------------------------------------

def evaluate_greedy_parity(
    custom_model: Transformer,
    hf_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 32,
    device: str = "cuda",
) -> Tuple[Dict, float]:
    """
    Compares step-by-step greedy autoregressive generation and prefill logits
    between custom_model and HuggingFace AutoModelForCausalLM.
    """
    start_time = time.perf_counter()
    results = []
    total_tokens_generated = 0
    total_tokens_matched = 0
    total_prompts_matched = 0

    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    for special_tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(special_tok)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            stop_token_ids.add(tok_id)

    with torch.inference_mode():
        for idx, prompt in enumerate(prompts):
            # Format prompt using instruct chat template
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]

            # 1. Compare step 0 logits
            custom_step0 = custom_model(input_ids, return_all_logits=False)[:, -1, :]
            hf_step0 = hf_model(input_ids).logits[:, -1, :]
            
            max_logit_diff = torch.max(torch.abs(custom_step0 - hf_step0)).item()
            cos_sim = F.cosine_similarity(custom_step0.float(), hf_step0.float(), dim=-1).item()

            # 2. Autoregressive greedy generation on Custom Model
            curr_custom = input_ids.clone()
            custom_tokens = []
            for _ in range(max_new_tokens):
                logits = custom_model(curr_custom, return_all_logits=False)[:, -1, :]
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                tok_val = next_tok.item()
                custom_tokens.append(tok_val)
                if tok_val in stop_token_ids:
                    break
                curr_custom = torch.cat([curr_custom, next_tok], dim=-1)

            # 3. Autoregressive greedy generation on HuggingFace Model
            curr_hf = input_ids.clone()
            hf_tokens = []
            for _ in range(max_new_tokens):
                logits = hf_model(curr_hf).logits[:, -1, :]
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                tok_val = next_tok.item()
                hf_tokens.append(tok_val)
                if tok_val in stop_token_ids:
                    break
                curr_hf = torch.cat([curr_hf, next_tok], dim=-1)

            # 4. Compare token sequence
            comp_len = min(len(custom_tokens), len(hf_tokens))
            matches = sum(1 for a, b in zip(custom_tokens[:comp_len], hf_tokens[:comp_len]) if a == b)
            max_len = max(len(custom_tokens), len(hf_tokens))
            
            is_exact_match = (custom_tokens == hf_tokens)
            if is_exact_match:
                total_prompts_matched += 1

            total_tokens_matched += matches
            total_tokens_generated += max_len

            match_pct = (matches / max_len * 100.0) if max_len > 0 else 100.0
            results.append({
                "prompt": prompt,
                "cos_sim": cos_sim,
                "max_diff": max_logit_diff,
                "tokens_matched": matches,
                "total_tokens": max_len,
                "match_pct": match_pct,
                "exact_match": is_exact_match,
            })

    elapsed = time.perf_counter() - start_time
    overall_token_match = (total_tokens_matched / total_tokens_generated * 100.0) if total_tokens_generated > 0 else 100.0

    summary = {
        "results": results,
        "total_prompts": len(prompts),
        "prompts_exact_matched": total_prompts_matched,
        "token_match_pct": overall_token_match,
        "avg_cos_sim": sum(r["cos_sim"] for r in results) / len(results),
        "max_logit_diff": max(r["max_diff"] for r in results),
    }
    return summary, elapsed


# -----------------------------------------------------------------------------
# Pillar 3: ARC-Easy Benchmark (Standard lm-eval style continuation logprob)
# -----------------------------------------------------------------------------

def evaluate_arc_easy(
    model: Transformer,
    tokenizer: AutoTokenizer,
    data_path: str = "data/arc_easy_150.json",
    limit: int = 100,
    device: str = "cuda",
) -> Tuple[float, int, int, float]:
    """
    Evaluates ARC-Easy using standard length-normalized continuation log-likelihood.
    Returns: (accuracy, num_correct, total_evaluated, elapsed_time)
    """
    start_time = time.perf_counter()
    data_path = resolve_data_path(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"ARC-Easy test file not found at {data_path}.")

    with open(data_path, "r", encoding="utf-8") as f:
        items = json.load(f)[:limit]

    correct = 0
    with torch.inference_mode():
        for item in items:
            q = item["question"]
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]
            
            prompt_prefix = f"Question: {q}\nAnswer:"
            prefix_ids = tokenizer.encode(prompt_prefix, add_special_tokens=True)
            
            choice_logprobs = {}
            for l, text in zip(labels, texts):
                full_text = f"{prompt_prefix} {text}"
                full_ids = tokenizer.encode(full_text, add_special_tokens=True)
                target_ids = full_ids[len(prefix_ids):]
                if len(target_ids) == 0:
                    continue

                input_tensor = torch.tensor([full_ids], device=device)
                logits = model(input_tensor, return_all_logits=True)  # (1, seqlen, vocab_size)
                log_probs = F.log_softmax(logits[0, :-1], dim=-1)     # (seqlen-1, vocab_size)

                start_idx = len(prefix_ids) - 1
                target_logprobs = [
                    log_probs[start_idx + pos, tok].item()
                    for pos, tok in enumerate(target_ids)
                ]
                
                # Length-normalized log-likelihood (standard lm-eval metric)
                choice_logprobs[l] = sum(target_logprobs) / len(target_logprobs)

            if choice_logprobs:
                pred = max(choice_logprobs, key=choice_logprobs.get)
                if pred == item["answerKey"]:
                    correct += 1

    total = len(items)
    accuracy = (correct / total) if total > 0 else 0.0
    elapsed = time.perf_counter() - start_time
    return accuracy, correct, total, elapsed


# -----------------------------------------------------------------------------
# Main CLI & Runner
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lightweight Quality Eval Suite for tiny_vllm.")
    parser.add_argument("--model_path", type=str, default="unsloth/Llama-3.2-1B-Instruct",
                        help="HuggingFace model ID or local directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--wt_chunks", type=int, default=5,
                        help="Number of 1024-token chunks for WikiText-2 PPL")
    parser.add_argument("--arc_limit", type=int, default=100,
                        help="Number of ARC-Easy questions to evaluate (default: 100)")
    parser.add_argument("--parity_tokens", type=int, default=32,
                        help="Tokens to generate per prompt for greedy parity check")
    parser.add_argument("--skip_ppl", action="store_true", help="Skip WikiText-2 PPL check")
    parser.add_argument("--skip_parity", action="store_true", help="Skip HuggingFace Parity check")
    parser.add_argument("--skip_arc", action="store_true", help="Skip ARC-Easy check")
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("      TINY_VLLM LIGHTWEIGHT QUALITY EVALUATION SUITE")
    print("=" * 70)
    print(f"[*] Target Model  : {args.model_path}")
    print(f"[*] Compute Device: {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"[*] Precision     : {dtype}")
    print("=" * 70)

    # 1. Load Tokenizer & Custom Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_args = ModelArgs.llama_3_2_1b()
    custom_model = Transformer(model_args).to(device=args.device, dtype=dtype)
    load_hf_safetensors(custom_model, args.model_path, device=args.device, dtype=dtype)
    custom_model.eval()

    total_suite_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Pillar 1: WikiText-2 PPL
    # -------------------------------------------------------------------------
    ppl_score = None
    ppl_time = 0.0
    if not args.skip_ppl:
        print("\n" + "-" * 70)
        print(" [1/3] Running WikiText-2 Perplexity (PPL)...")
        print("-" * 70)
        ppl_score, loss_score, ppl_time = evaluate_wikitext_ppl(
            custom_model, tokenizer, num_chunks=args.wt_chunks, chunk_size=1024, device=args.device
        )
        print(f"  ✓ Evaluated {args.wt_chunks} chunks of 1024 tokens ({args.wt_chunks * 1024:,} total tokens)")
        print(f"  ✓ Cross-Entropy Loss: {loss_score:.4f}")
        print(f"  ✓ Perplexity (PPL)  : {ppl_score:.2f} (Instruct Baseline: ~15.7)")
        print(f"  ⏱ Time taken        : {ppl_time:.2f}s")
    else:
        print("\n[-] Skipping WikiText-2 PPL.")

    # -------------------------------------------------------------------------
    # Pillar 2: Greedy Parity vs HuggingFace Reference
    # -------------------------------------------------------------------------
    parity_summary = None
    parity_time = 0.0
    if not args.skip_parity:
        print("\n" + "-" * 70)
        print(" [2/3] Running Greedy Parity vs HuggingFace Reference...")
        print("-" * 70)
        print("  [*] Loading HuggingFace reference model...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
        ).to(args.device)
        hf_model.eval()

        parity_summary, parity_time = evaluate_greedy_parity(
            custom_model, hf_model, tokenizer,
            prompts=PARITY_PROMPTS,
            max_new_tokens=args.parity_tokens,
            device=args.device,
        )

        print(f"  ✓ Evaluated {parity_summary['total_prompts']} diverse test prompts ({args.parity_tokens} tokens each)")
        print(f"  ✓ Logit Cosine Similarity : {parity_summary['avg_cos_sim']:.6f} (Target: > 0.999)")
        print(f"  ✓ Max Logit Difference    : {parity_summary['max_logit_diff']:.5f}")
        print(f"  ✓ Token Match Agreement   : {parity_summary['token_match_pct']:.1f}%")
        print(f"  ✓ Prompts 100% Identical  : {parity_summary['prompts_exact_matched']} / {parity_summary['total_prompts']}")
        print(f"  ⏱ Time taken              : {parity_time:.2f}s")

        # Free HF model memory
        del hf_model
        torch.cuda.empty_cache()
    else:
        print("\n[-] Skipping Greedy Parity.")

    # -------------------------------------------------------------------------
    # Pillar 3: ARC-Easy Benchmark
    # -------------------------------------------------------------------------
    arc_acc = None
    arc_time = 0.0
    if not args.skip_arc:
        print("\n" + "-" * 70)
        print(f" [3/3] Running ARC-Easy Benchmark (first {args.arc_limit} questions)...")
        print("-" * 70)
        arc_acc, arc_correct, arc_total, arc_time = evaluate_arc_easy(
            custom_model, tokenizer, limit=args.arc_limit, device=args.device
        )
        print(f"  ✓ Correct Answers : {arc_correct} / {arc_total}")
        print(f"  ✓ Accuracy        : {arc_acc * 100.0:.2f}% (0-shot Baseline: 65.0%)")
        print(f"  ⏱ Time taken      : {arc_time:.2f}s")
    else:
        print("\n[-] Skipping ARC-Easy.")

    total_suite_time = time.perf_counter() - total_suite_start

    # -------------------------------------------------------------------------
    # Final Scorecard Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                     EVALUATION SCORECARD")
    print("=" * 70)
    print(f" {'Metric':<30} | {'Score':<18} | {'Status / Guideline'}")
    print("-" * 70)
    
    if ppl_score is not None:
        status = "EXCELLENT" if ppl_score < 16.5 else ("DEGRADED" if ppl_score > 18.0 else "FAIR")
        print(f" {'WikiText-2 Perplexity (PPL)':<30} | {ppl_score:>6.2f}            | {status} (Baseline: ~15.7)")
    
    if parity_summary is not None:
        token_acc = parity_summary['token_match_pct']
        status = "PERFECT/LOSSLESS" if token_acc >= 95.0 else ("ACCEPTABLE (Lossy)" if token_acc >= 80.0 else "REGRESSION")
        print(f" {'HF Token Agreement':<30} | {token_acc:>5.1f}%            | {status}")
        print(f" {'HF Step-0 Logit CosSim':<30} | {parity_summary['avg_cos_sim']:>8.6f}         | Target: > 0.999")

    if arc_acc is not None:
        pct = arc_acc * 100.0
        status = "HEALTHY" if pct >= 62.0 else "DEGRADED"
        print(f" {'ARC-Easy Accuracy':<30} | {pct:>5.1f}%            | {status} (0-shot Baseline: 65.0%)")

    print("-" * 70)
    print(f" Total Eval Duration: {total_suite_time:.2f} seconds ({total_suite_time / 60.0:.1f} minutes)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
