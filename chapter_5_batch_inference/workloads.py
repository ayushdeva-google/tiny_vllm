"""
Workload & Incoming Request Arrival Modeling for Chapter 5 Batch Inference.

Follows standard LLM serving benchmark conventions (vLLM `bench serve`, SGLang,
MLPerf Inference Server, and Sarathi-Serve):
1. Real-world conversational length skew via cached ShareGPT (`data/sharegpt_200.json`).
2. Controlled synthetic presets built from real English token spans (`data/wikitext_2_test.txt`):
   - `synthetic-uniform`: Tight length distribution (isolates pure GEMV -> GEMM weight reuse).
   - `synthetic-skewed`: Bimodal chat + RAG traffic (exposes HoL blocking, padding waste & stragglers).
   - `synthetic-prefill-heavy`: Long prompts, short decodes (compute-bound prefill stress test).
   - `synthetic-decode-heavy`: Short prompts, long decodes (HBM bandwidth-bound decode stress test).
3. Open-loop arrival process:
   - Online serving: Seeded Poisson process (Exp(lambda)) or generalized Gamma(burstiness) arrivals.
   - Offline throughput: Burst mode (`request_rate = inf`, all requests arrive at t = 0.0s).
"""

import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

WORKLOAD_CHOICES = [
    "sharegpt",
    "synthetic-uniform",
    "synthetic-skewed",
    "synthetic-prefill-heavy",
    "synthetic-decode-heavy",
]


@dataclass
class InferenceRequest:
    """Single incoming inference request in a serving benchmark trace."""

    request_id: int
    source_id: str
    workload: str
    prompt_token_ids: List[int]
    prompt_len: int
    target_output_len: int
    arrival_time_s: float = 0.0


def _resolve_repo_data_path(rel_path: str) -> str:
    if os.path.exists(rel_path):
        return os.path.abspath(rel_path)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_candidate = os.path.abspath(os.path.join(here, "..", rel_path))
    if os.path.exists(repo_candidate):
        return repo_candidate
    raise FileNotFoundError(f"Could not find dataset file at '{rel_path}' or '{repo_candidate}'.")


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    weight = rank - lo
    return float(sorted_vals[lo] * (1.0 - weight) + sorted_vals[hi] * weight)


# -----------------------------------------------------------------------------
# 1. ShareGPT Workload Loader
# -----------------------------------------------------------------------------


def load_sharegpt_requests(
    tokenizer: AutoTokenizer,
    num_requests: int = 32,
    seed: int = 42,
    data_path: str = "data/sharegpt_200.json",
    max_prompt_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
) -> List[InferenceRequest]:
    """
    Loads real conversational prompts & target output token budgets from cached ShareGPT.
    Cycles deterministically if `num_requests` exceeds cached pool size (200).
    """
    resolved = _resolve_repo_data_path(data_path)
    with open(resolved, "r", encoding="utf-8") as f:
        records = json.load(f)

    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)

    requests: List[InferenceRequest] = []
    idx = 0
    while len(requests) < num_requests:
        rec = pool[idx % len(pool)]
        idx += 1

        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": rec["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = rec["prompt"]

        token_ids = tokenizer.encode(formatted, add_special_tokens=False)
        if max_prompt_tokens is not None and len(token_ids) > max_prompt_tokens:
            token_ids = token_ids[:max_prompt_tokens]

        out_len = int(rec["output_len"])
        if max_output_tokens is not None:
            out_len = min(out_len, max_output_tokens)
        out_len = max(1, out_len)

        req_id = len(requests)
        requests.append(
            InferenceRequest(
                request_id=req_id,
                source_id=str(rec.get("id", f"sg_{req_id}")),
                workload="sharegpt",
                prompt_token_ids=token_ids,
                prompt_len=len(token_ids),
                target_output_len=out_len,
            )
        )

    return requests


# -----------------------------------------------------------------------------
# 2. Synthetic Presets (Real WikiText-2 Token Spans)
# -----------------------------------------------------------------------------


def _load_wikitext_token_corpus(tokenizer: AutoTokenizer, data_path: str = "data/wikitext_2_test.txt") -> List[int]:
    resolved = _resolve_repo_data_path(data_path)
    with open(resolved, "r", encoding="utf-8") as f:
        text = f.read()
    # Encode initial ~200k chars (~50k tokens) for fast (<50ms) realistic span slicing
    tokens = tokenizer.encode(text[:200_000], add_special_tokens=False)
    if len(tokens) < 4096:
        raise ValueError("WikiText corpus too short to sample synthetic prompts.")
    return tokens


def build_synthetic_requests(
    tokenizer: AutoTokenizer,
    preset: str = "synthetic-skewed",
    num_requests: int = 32,
    seed: int = 42,
    synth_prompt_len: int = 128,
    synth_output_len: int = 64,
    max_prompt_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
) -> List[InferenceRequest]:
    """
    Constructs controlled synthetic request traces using real English token spans from WikiText-2.
    """
    corpus = _load_wikitext_token_corpus(tokenizer)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    rng = random.Random(seed)

    requests: List[InferenceRequest] = []
    for req_id in range(num_requests):
        if preset == "synthetic-uniform":
            p_jit = max(4, synth_prompt_len // 8)
            o_jit = max(2, synth_output_len // 8)
            p_len = rng.randint(max(8, synth_prompt_len - p_jit), synth_prompt_len + p_jit)
            o_len = rng.randint(max(4, synth_output_len - o_jit), synth_output_len + o_jit)
        elif preset == "synthetic-skewed":
            # 75% short interactive chat, 25% long RAG/document reasoning
            if rng.random() < 0.75:
                p_len = rng.randint(48, 128)
                o_len = rng.randint(24, 56)
            else:
                p_len = rng.randint(384, 768)
                o_len = rng.randint(128, 224)
        elif preset == "synthetic-prefill-heavy":
            p_len = rng.randint(512, 1024)
            o_len = rng.randint(16, 32)
        elif preset == "synthetic-decode-heavy":
            p_len = rng.randint(32, 64)
            o_len = rng.randint(128, 256)
        else:
            raise ValueError(f"Unknown synthetic preset: {preset}")

        if max_prompt_tokens is not None:
            p_len = min(p_len, max_prompt_tokens)
        if max_output_tokens is not None:
            o_len = min(o_len, max_output_tokens)

        body_len = max(1, p_len - len(bos))
        start_idx = rng.randint(0, len(corpus) - body_len - 1)
        prompt_ids = bos + corpus[start_idx : start_idx + body_len]

        requests.append(
            InferenceRequest(
                request_id=req_id,
                source_id=f"{preset}_{req_id}",
                workload=preset,
                prompt_token_ids=prompt_ids,
                prompt_len=len(prompt_ids),
                target_output_len=max(1, o_len),
            )
        )

    return requests


# -----------------------------------------------------------------------------
# 3. Open-Loop Arrival Process (Poisson / Gamma / Offline Burst)
# -----------------------------------------------------------------------------


def assign_arrival_schedule(
    requests: List[InferenceRequest],
    request_rate: float,
    burstiness: float = 1.0,
    seed: int = 42,
) -> List[InferenceRequest]:
    """
    Assigns deterministic arrival timestamps `arrival_time_s` (relative to t=0):
    - `request_rate == inf`: Offline/burst mode, all requests arrive at t = 0.0s.
    - `request_rate < inf`: Open-loop Gamma/Poisson arrival process (vLLM `bench serve` standard):
        * `burstiness == 1.0` -> Exact Poisson process (inter-arrival ~ Exponential(request_rate)).
        * `0 < burstiness < 1.0` -> More bursty/clustered arrivals than Poisson.
        * `burstiness > 1.0` -> More regular/periodic arrivals than Poisson.
    """
    if request_rate <= 0:
        raise ValueError(f"request_rate must be > 0 or inf, got {request_rate}")
    if burstiness <= 0:
        raise ValueError(f"burstiness must be > 0, got {burstiness}")

    if math.isinf(request_rate):
        for req in requests:
            req.arrival_time_s = 0.0
        return requests

    rng = random.Random(seed + 10_007)
    theta = 1.0 / (request_rate * burstiness)
    t_curr = 0.0
    for i, req in enumerate(requests):
        if i == 0:
            req.arrival_time_s = 0.0
        else:
            delta_t = rng.gammavariate(alpha=burstiness, beta=theta)
            t_curr += delta_t
            req.arrival_time_s = t_curr
    return requests


def build_workload(
    tokenizer: AutoTokenizer,
    workload: str = "sharegpt",
    num_requests: int = 32,
    request_rate: float = 1.0,
    burstiness: float = 1.0,
    seed: int = 42,
    synth_prompt_len: int = 128,
    synth_output_len: int = 64,
    max_prompt_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
) -> List[InferenceRequest]:
    """Unified builder returning requests with deterministic tokens and arrival timestamps."""
    if workload == "sharegpt":
        reqs = load_sharegpt_requests(
            tokenizer=tokenizer,
            num_requests=num_requests,
            seed=seed,
            max_prompt_tokens=max_prompt_tokens,
            max_output_tokens=max_output_tokens,
        )
    elif workload in WORKLOAD_CHOICES:
        reqs = build_synthetic_requests(
            tokenizer=tokenizer,
            preset=workload,
            num_requests=num_requests,
            seed=seed,
            synth_prompt_len=synth_prompt_len,
            synth_output_len=synth_output_len,
            max_prompt_tokens=max_prompt_tokens,
            max_output_tokens=max_output_tokens,
        )
    else:
        raise ValueError(f"Unsupported workload '{workload}'. Choose from {WORKLOAD_CHOICES}.")

    return assign_arrival_schedule(reqs, request_rate=request_rate, burstiness=burstiness, seed=seed)


def summarize_workload(requests: List[InferenceRequest]) -> Dict[str, Any]:
    """Returns summary distribution statistics of prompt & target output lengths."""
    p_lens = sorted(r.prompt_len for r in requests)
    o_lens = sorted(r.target_output_len for r in requests)
    p_mean = statistics.mean(p_lens) if p_lens else 0.0
    o_mean = statistics.mean(o_lens) if o_lens else 0.0
    p_std = statistics.pstdev(p_lens) if len(p_lens) > 1 else 0.0
    o_std = statistics.pstdev(o_lens) if len(o_lens) > 1 else 0.0

    return {
        "num_requests": len(requests),
        "workload": requests[0].workload if requests else "empty",
        "total_prompt_tokens": sum(p_lens),
        "total_target_output_tokens": sum(o_lens),
        "arrival_span_s": round(requests[-1].arrival_time_s - requests[0].arrival_time_s, 4) if requests else 0.0,
        "prompt_len": {
            "min": p_lens[0],
            "mean": round(p_mean, 1),
            "std": round(p_std, 1),
            "cv": round(p_std / p_mean, 2) if p_mean > 0 else 0.0,
            "p50": round(_percentile(p_lens, 50), 1),
            "p90": round(_percentile(p_lens, 90), 1),
            "p99": round(_percentile(p_lens, 99), 1),
            "max": p_lens[-1],
        },
        "target_output_len": {
            "min": o_lens[0],
            "mean": round(o_mean, 1),
            "std": round(o_std, 1),
            "cv": round(o_std / o_mean, 2) if o_mean > 0 else 0.0,
            "p50": round(_percentile(o_lens, 50), 1),
            "p90": round(_percentile(o_lens, 90), 1),
            "p99": round(_percentile(o_lens, 99), 1),
            "max": o_lens[-1],
        },
    }

