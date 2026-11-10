#!/usr/bin/env python3
"""
Chapter 5 · Batch Inference Evaluation & Profiling Pipeline.

Models incoming concurrent inference requests (seeded open-loop Poisson/Gamma arrivals
or offline burst mode) over realistic variable-length workloads (`sharegpt` and synthetic
length presets), executes them on a pluggable serving engine (starting with the `B=1`
Sequential FCFS baseline in pure PyTorch + KV Cache), and tracks industry-standard
serving, queueing, memory, and hardware operator metrics.

Examples:
  # 1. Single online arrival rate on ShareGPT + PyTorch hardware profiling (~30s on L4)
  .venv/bin/python chapter_5_batch_inference/benchmark_serving.py \
      --workload sharegpt --num-requests 24 --request-rate 0.35

  # 2. Offline burst throughput benchmark on skewed synthetic traffic
  .venv/bin/python chapter_5_batch_inference/benchmark_serving.py \
      --workload synthetic-skewed --num-requests 32 --request-rate inf

  # 3. Full QPS sweep (Latency vs. Throughput knee curve across arrival rates)
  .venv/bin/python chapter_5_batch_inference/benchmark_serving.py \
      --workload synthetic-skewed --num-requests 24 --qps-sweep 0.15,0.35,0.6,1.0,inf
"""

import argparse
import math
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

import torch
from rich.console import Console

# Allow running both from repo root (`python chapter_5_batch_inference/benchmark_serving.py`)
# and from inside `chapter_5_batch_inference/`.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from engines import BaseServingEngine, SequentialFCFSEngine
from llama_model import load_model_and_tokenizer
from profiler_metrics import (
    compute_serving_summary,
    print_single_run_report,
    print_sweep_report,
    save_benchmark_json,
)
from workloads import WORKLOAD_CHOICES, build_workload, summarize_workload


def _parse_rate(val: str) -> float:
    v = val.strip().lower()
    if v in ("inf", "infinity", "burst"):
        return float("inf")
    rate = float(v)
    if rate <= 0:
        raise argparse.ArgumentTypeError(f"Arrival rate must be > 0 or 'inf', got '{val}'.")
    return rate


def build_engine(
    engine_name: str,
    model,
    tokenizer,
    max_seq_len: int,
    device: str,
) -> BaseServingEngine:
    if engine_name == "sequential_fcfs":
        return SequentialFCFSEngine(
            model=model,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            device=device,
        )
    raise ValueError(f"Unknown engine '{engine_name}'. Currently registered: ['sequential_fcfs'].")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Batch Inference Serving Benchmark & Hardware Profiler (Chapter 5)."
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="sequential_fcfs",
        choices=["sequential_fcfs"],
        help="Serving engine implementation (baseline: sequential_fcfs with batch_size=1).",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="sharegpt",
        choices=WORKLOAD_CHOICES,
        help="Incoming prompt/output length workload distribution (default: sharegpt).",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=24,
        help="Number of incoming requests per run (default: 24 for fast iteration on L4).",
    )
    parser.add_argument(
        "--request-rate",
        type=_parse_rate,
        default=0.35,
        help="Open-loop Poisson arrival rate in req/s, or 'inf' for offline burst (default: 0.35).",
    )
    parser.add_argument(
        "--qps-sweep",
        type=str,
        default=None,
        help="Optional comma-separated QPS rates to sweep (e.g. '0.15,0.35,0.6,1.0,inf').",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Gamma arrival shape factor: 1.0 = exact Poisson, <1.0 = burstier, >1.0 = periodic.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence capacity pre-allocated in static KVCache (default: 2048).",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=1024,
        help="Cap prompt tokens per request (0 = uncapped, default: 1024).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=128,
        help="Cap target output tokens per request for fast runs (0 = uncapped, default: 128).",
    )
    parser.add_argument(
        "--synth-prompt-len",
        type=int,
        default=128,
        help="Mean prompt tokens for `synthetic-uniform` preset (default: 128).",
    )
    parser.add_argument(
        "--synth-output-len",
        type=int,
        default=64,
        help="Mean output tokens for `synthetic-uniform` preset (default: 64).",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Replay arrivals in real wall-clock time (sleep on idle gaps) instead of event-driven clock.",
    )
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        default=False,
        help="Stop early on EOS tokens instead of generating exact target_output_len (default: ignore_eos=True).",
    )
    parser.add_argument(
        "--slo-ttft-ms",
        type=float,
        default=2000.0,
        help="TTFT Service Level Objective threshold in ms for Goodput (default: 2000.0).",
    )
    parser.add_argument(
        "--slo-tpot-ms",
        type=float,
        default=50.0,
        help="TPOT Service Level Objective threshold in ms/tok for Goodput (default: 50.0).",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        default=False,
        help="Skip isolated PyTorch Profiler Prefill/Decode hardware trace decomposition.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic RNG seed for workload sampling and Poisson arrival schedule (default: 42).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID or local safetensors directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device (cuda or cpu).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON path (defaults to chapter_5_batch_inference/results/<engine>_<workload>.json).",
    )
    parser.add_argument(
        "--include-per-request",
        action="store_true",
        default=True,
        help="Save per-request latency & token ID records in JSON output (default: True).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    term_cols = shutil.get_terminal_size((136, 24)).columns
    console = Console(width=max(136, term_cols))
    torch.manual_seed(args.seed)

    max_prompt_cap: Optional[int] = args.max_prompt_tokens if args.max_prompt_tokens > 0 else None
    max_output_cap: Optional[int] = args.max_output_tokens if args.max_output_tokens > 0 else None
    ignore_eos = not args.respect_eos
    clock_mode = "real-time" if args.real_time else "event-driven"

    rates: List[float] = (
        [_parse_rate(r) for r in args.qps_sweep.split(",") if r.strip()]
        if args.qps_sweep
        else [args.request_rate]
    )

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    gpu_desc = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
    console.print(f"[*] Initializing Native PyTorch Llama-3.2-1B on [bold]{args.device}[/bold] ({gpu_desc}, {dtype})...")

    model, tokenizer = load_model_and_tokenizer(
        model_path_or_repo=args.model_path,
        device=args.device,
        dtype=dtype,
        max_seq_len=max(8192, args.max_seq_len),
    )
    model_weights_mb = model.get_weight_memory_mb()

    engine = build_engine(
        engine_name=args.engine,
        model=model,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )

    console.print("[*] Warming up model kernels, KV cache & profiler buffers...")
    engine.warmup(include_profiler=(not args.no_profile))

    run_summaries: List[Dict[str, Any]] = []
    per_rate_requests: Dict[str, List[Dict[str, Any]]] = {}

    for rate in rates:
        rate_label = "inf" if math.isinf(rate) else f"{rate:g}"
        requests = build_workload(
            tokenizer=tokenizer,
            workload=args.workload,
            num_requests=args.num_requests,
            request_rate=rate,
            burstiness=args.burstiness,
            seed=args.seed,
            synth_prompt_len=args.synth_prompt_len,
            synth_output_len=args.synth_output_len,
            max_prompt_tokens=max_prompt_cap,
            max_output_tokens=max_output_cap,
        )
        wl_summary = summarize_workload(requests)

        # Ensure pre-allocated static KVCache can fit the longest sequence in this workload
        max_needed_seq = wl_summary["prompt_len"]["max"] + wl_summary["target_output_len"]["max"]
        if max_needed_seq > args.max_seq_len:
            raise ValueError(
                f"Longest request ({max_needed_seq} tokens) exceeds --max-seq-len ({args.max_seq_len})."
            )

        console.print(
            f"[*] Running [bold cyan]{engine.name}[/bold cyan] | workload=[bold]{args.workload}[/bold] "
            f"(N={args.num_requests}, in={wl_summary['total_prompt_tokens']} tok, "
            f"out={wl_summary['total_target_output_tokens']} tok) | QPS=[bold]{rate_label}[/bold] ({clock_mode})..."
        )

        run_out = engine.serve_requests(
            requests=requests,
            real_time=args.real_time,
            ignore_eos=ignore_eos,
        )

        summary = compute_serving_summary(
            records=run_out.records,
            engine_name=engine.name,
            workload_summary=wl_summary,
            request_rate=rate,
            burstiness=args.burstiness,
            clock_mode=clock_mode,
            wall_clock_exec_s=run_out.wall_clock_exec_s,
            gpu_idle_time_s=run_out.gpu_idle_time_s,
            model_weights_mb=model_weights_mb,
            kv_cache_allocated_mb=engine.get_kv_allocated_mb(),
            kv_bytes_per_token=engine.get_kv_bytes_per_token(),
            peak_vram_allocated_mb=run_out.peak_vram_allocated_mb,
            peak_vram_reserved_mb=run_out.peak_vram_reserved_mb,
            slo_ttft_ms=args.slo_ttft_ms,
            slo_tpot_ms=args.slo_tpot_ms,
        )
        run_summaries.append(summary)
        if args.include_per_request:
            per_rate_requests[rate_label] = [r.to_dict() for r in run_out.records]

    # Run isolated PyTorch Profiler hardware decomposition on representative workload lengths
    phase_profile: Optional[Dict[str, Any]] = None
    if not args.no_profile and isinstance(engine, SequentialFCFSEngine):
        rep_wl = run_summaries[0]["workload"]
        rep_prefill_len = int(round(rep_wl["prompt_len"]["p50"]))
        rep_decode_ctx = int(round(rep_wl["prompt_len"]["p50"] + rep_wl["target_output_len"]["p50"] / 2.0))
        traces_dir = os.path.join(HERE, "profile_results", "traces")
        console.print(
            f"[*] Running isolated PyTorch Profiler hardware pass "
            f"(prefill_len={rep_prefill_len}, decode_ctx={rep_decode_ctx})..."
        )
        phase_profile = engine.profile_representative_phases(
            prefill_seq_len=rep_prefill_len,
            decode_context_len=rep_decode_ctx,
            traces_dir=traces_dir,
        )

    # Render terminal reports
    if len(run_summaries) == 1:
        print_single_run_report(run_summaries[0], phase_profile=phase_profile, console=console)
    else:
        # Print detailed breakdown for the median/last rate + full QPS knee table
        print_single_run_report(run_summaries[-1], phase_profile=phase_profile, console=console)
        print_sweep_report(run_summaries, console=console)

    # Export JSON results
    default_name = (
        f"{engine.name}_{args.workload}_sweep.json"
        if len(run_summaries) > 1
        else f"{engine.name}_{args.workload}_qps_{run_summaries[0]['request_rate_qps']}.json"
    )
    out_path = args.output_json or os.path.join(HERE, "results", default_name)

    payload: Dict[str, Any] = {
        "config": {
            "engine": engine.name,
            "workload": args.workload,
            "num_requests": args.num_requests,
            "rates": ["inf" if math.isinf(r) else r for r in rates],
            "burstiness": args.burstiness,
            "clock_mode": clock_mode,
            "ignore_eos": ignore_eos,
            "max_seq_len": args.max_seq_len,
            "max_prompt_tokens": max_prompt_cap,
            "max_output_tokens": max_output_cap,
            "seed": args.seed,
        },
        "summaries": run_summaries,
        "hardware_phase_profile": phase_profile,
    }
    if args.include_per_request:
        payload["per_request_records"] = per_rate_requests

    save_benchmark_json(payload, out_path)
    console.print(f"\n[bold green][✓][/bold green] Saved structured benchmark & profiling metrics to: [underline]{out_path}[/underline]")
    if phase_profile:
        console.print(
            f"[bold green][✓][/bold green] Saved Chrome traces to: "
            f"[underline]{phase_profile['prefill']['trace_path']}[/underline] & "
            f"[underline]{phase_profile['decode']['trace_path']}[/underline]"
        )


if __name__ == "__main__":
    main()

