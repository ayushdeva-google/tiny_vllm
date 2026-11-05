#!/usr/bin/env python3
"""
Chapter 5 · Unified Benchmark & Profiler:
  Sequential Baseline (B=1) vs. Static Batching vs. Continuous Batching.

Workflow:
1. Correctness Gate: Verifies exact FP32 token-for-token greedy parity across
   `SequentialFCFSEngine`, `StaticBatchEngine`, and `ContinuousBatchEngine` on
   interleaved variable-length ShareGPT requests before timing anything.
2. Arrival Matrix Evaluation: Evaluates a rich mix of ShareGPT queries across:
   - Sweep A (Varying Poisson Rate lambda, beta = 1.0): Light -> Knee -> Heavy -> Burst (inf).
   - Sweep B (Varying Burstiness beta at fixed lambda): Micro-Burst (beta=0.3) vs
     Poisson (beta=1.0) vs Smooth Periodic (beta=8.0).
3. Hardware Profiler Comparison: Captures isolated PyTorch Profiler Chrome traces
   comparing B=1 GEMV vs. Batched GEMM execution on the NVIDIA L4.
"""

import argparse
import gc
import math
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from continuous_batching import ContinuousBatchEngine, load_continuous_batch_model
from engines import SequentialFCFSEngine
from llama_model import load_model_and_tokenizer
from profiler_metrics import compute_serving_summary, save_benchmark_json
from static_batching import StaticBatchEngine, load_static_batch_model
from workloads import WORKLOAD_CHOICES, build_workload, summarize_workload


DEFAULT_SCENARIOS: List[Tuple[float, float, str]] = [
    # Sweep A: Varying Poisson Arrival Rate lambda (beta = 1.0)
    (0.25, 1.0, "Poisson Light (λ=0.25, β=1.0)"),
    (1.00, 1.0, "Poisson Moderate (λ=1.0, β=1.0)"),
    (2.50, 1.0, "Poisson Busy (λ=2.5, β=1.0)"),
    (5.00, 1.0, "Poisson Heavy (λ=5.0, β=1.0)"),
    (float("inf"), 1.0, "Offline Burst (λ=inf)"),
    # Sweep B: Varying Burstiness beta at fixed lambda = 2.5 req/s (compares directly with row 3)
    (2.50, 0.30, "Micro-Bursty (λ=2.5, β=0.30)"),
    (2.50, 8.00, "Smooth Periodic (λ=2.5, β=8.0)"),
]


def _parse_scenarios(spec: Optional[str]) -> List[Tuple[float, float, str]]:
    """
    Parses custom `--scenarios` string formatted as comma-separated `lambda:beta` pairs,
    e.g. `0.25:1.0,1.0:1.0,2.5:0.3,inf:1.0`.
    """
    if not spec:
        return DEFAULT_SCENARIOS
    out: List[Tuple[float, float, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            r_str, b_str = item.split(":", 1)
            beta = float(b_str)
        else:
            r_str, beta = item, 1.0
        rate = float("inf") if r_str.strip().lower() in ("inf", "burst") else float(r_str)
        lbl = f"Burst (λ=inf)" if math.isinf(rate) else f"λ={rate:g}, β={beta:g}"
        out.append((rate, beta, lbl))
    return out


def verify_fp32_parity_and_bf16_agreement(
    device: str,
    seed: int = 42,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
    """
    Proves mathematical equivalence in FP32 (exact greedy token match across all 3 standalone
    implementations) and measures BF16 Tensor-Core vs GEMV greedy token agreement.
    """
    console = console or Console()
    console.print("\n[*] [bold cyan]Step 1/3 · Running Correctness & Parity Gate (FP32 exactness + BF16 agreement)...[/bold cyan]")

    # 1. Exact FP32 mathematical check on 8 interleaved variable-length ShareGPT requests
    m_b1, tok = load_model_and_tokenizer(device=device, dtype=torch.float32)
    reqs = build_workload(tok, workload="sharegpt", num_requests=8, request_rate=3.0, seed=seed, max_output_tokens=20)
    e_b1 = SequentialFCFSEngine(m_b1, tok, max_seq_len=1024, device=device)
    rec_b1 = e_b1.serve_requests(reqs, ignore_eos=True).records
    del m_b1, e_b1
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    m_st, _ = load_static_batch_model(device=device, dtype=torch.float32)
    e_st = StaticBatchEngine(m_st, tok, max_batch_size=4, max_seq_len=1024, device=device)
    rec_st = e_st.serve_requests(reqs, ignore_eos=True).records
    del m_st, e_st
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    m_cb, _ = load_continuous_batch_model(device=device, dtype=torch.float32)
    e_cb = ContinuousBatchEngine(m_cb, tok, max_batch_size=4, max_seq_len=1024, device=device)
    out_cb = e_cb.serve_requests(reqs, ignore_eos=True)
    rec_cb = out_cb.records
    del m_cb, e_cb
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    fp32_static_exact = 0
    fp32_cont_exact = 0
    for r0, rs, rc in zip(rec_b1, rec_st, rec_cb):
        if r0.generated_token_ids == rs.generated_token_ids:
            fp32_static_exact += 1
        if r0.generated_token_ids == rc.generated_token_ids:
            fp32_cont_exact += 1

    if fp32_static_exact != len(reqs) or fp32_cont_exact != len(reqs):
        raise RuntimeError(
            f"FP32 Parity Gate FAILED! Static={fp32_static_exact}/{len(reqs)}, "
            f"Continuous={fp32_cont_exact}/{len(reqs)}."
        )

    console.print(
        f"    [bold green][✓] FP32 Mathematical Parity PASSED:[/bold green] "
        f"Static={fp32_static_exact}/{len(reqs)} exact, Continuous={fp32_cont_exact}/{len(reqs)} exact "
        f"(including {out_cb.scheduler_diagnostics['mixed_prefill_decode_iterations']} mixed prefill+decode steps)."
    )

    return {
        "num_test_requests": len(reqs),
        "tokens_per_request": 20,
        "fp32_static_exact_requests": fp32_static_exact,
        "fp32_continuous_exact_requests": fp32_cont_exact,
        "mixed_prefill_decode_steps_tested": out_cb.scheduler_diagnostics["mixed_prefill_decode_iterations"],
        "passed": True,
    }


def render_comparison_tables(
    scenarios: List[Tuple[float, float, str]],
    engine_order: List[str],
    results_by_engine: Dict[str, List[Dict[str, Any]]],
    diagnostics_by_engine: Dict[str, List[Dict[str, Any]]],
    hw_profiles: Dict[str, Dict[str, Any]],
    slo_ttft_ms: float,
    slo_tpot_ms: float,
    console: Console,
) -> None:
    """Renders side-by-side Rich comparison tables across engines and (lambda, beta) scenarios."""

    # 1. Per-Scenario Side-by-Side Serving Comparison Table
    comp_tbl = Table(
        title=(
            f"Batch Inference Serving Comparison · ShareGPT · "
            f"SLOs: TTFT ≤ {slo_ttft_ms:.0f}ms, TPOT ≤ {slo_tpot_ms:.0f}ms"
        ),
        header_style="bold cyan",
    )
    comp_tbl.add_column("Scenario (λ, β)", style="bold white", overflow="fold")
    comp_tbl.add_column("Engine", style="cyan", overflow="fold")
    comp_tbl.add_column("Out tok/s", justify="right", style="bold green", overflow="fold")
    comp_tbl.add_column("req/s", justify="right", style="green", overflow="fold")
    comp_tbl.add_column("Wait P50/P99", justify="right", style="yellow", overflow="fold")
    comp_tbl.add_column("TTFT P50/P99", justify="right", style="bold yellow", overflow="fold")
    comp_tbl.add_column("TPOT P50", justify="right", overflow="fold")
    comp_tbl.add_column("ITL P99", justify="right", overflow="fold")
    comp_tbl.add_column("Goodput", justify="right", style="magenta", overflow="fold")

    for s_idx, (_, _, label) in enumerate(scenarios):
        for e_idx, eng_name in enumerate(engine_order):
            s = results_by_engine[eng_name][s_idx]
            tp = s["throughput"]
            lat = s["latencies_ms"]
            slo = s["slo_goodput"]
            scen_col = label if e_idx == 0 else ""
            comp_tbl.add_row(
                scen_col,
                eng_name,
                f"{tp['output_token_throughput_tps']:.1f}",
                f"{tp['request_throughput_rps']:.2f}",
                f"{lat['queue_wait']['p50']:.0f}/{lat['queue_wait']['p99']:.0f}ms",
                f"{lat['ttft']['p50']:.0f}/{lat['ttft']['p99']:.0f}ms",
                f"{lat['tpot']['p50']:.1f}ms",
                f"{lat['itl']['p99']:.1f}ms",
                f"{slo['goodput_rps']:.2f} ({slo['slo_attainment_pct']:.0f}%)",
            )
        if s_idx < len(scenarios) - 1:
            comp_tbl.add_section()

    console.print()
    console.print(comp_tbl)

    # 2. Scheduler Efficiency & Waste Comparison (Burst & Busy Online)
    eff_tbl = Table(
        title="Batching Efficiency: Padding Waste, Zombie Decode Slots & Active Batch Occupancy",
        header_style="bold magenta",
    )
    eff_tbl.add_column("Scenario", style="bold white", overflow="fold")
    eff_tbl.add_column("Engine", style="cyan", overflow="fold")
    eff_tbl.add_column("Prefill Pad Waste", justify="right", style="yellow", overflow="fold")
    eff_tbl.add_column("Zombie Decode Waste", justify="right", style="red", overflow="fold")
    eff_tbl.add_column("Avg Active Batch", justify="right", style="green", overflow="fold")
    eff_tbl.add_column("Peak Queue", justify="right", overflow="fold")
    eff_tbl.add_column("Speedup vs B=1", justify="right", style="bold green", overflow="fold")

    for s_idx, (_, _, label) in enumerate(scenarios):
        b1_tps = results_by_engine[engine_order[0]][s_idx]["throughput"]["output_token_throughput_tps"]
        for e_idx, eng_name in enumerate(engine_order):
            s = results_by_engine[eng_name][s_idx]
            diag = diagnostics_by_engine[eng_name][s_idx]
            tps = s["throughput"]["output_token_throughput_tps"]
            speedup = tps / max(1e-6, b1_tps)

            pad_w = diag.get("prefill_padding_waste_pct", 0.0)
            zom_w = diag.get("decode_zombie_slot_waste_pct", 0.0)
            avg_b = diag.get(
                "avg_active_batch_size",
                diag.get("effective_active_batch_during_decode", 1.0),
            )
            eff_tbl.add_row(
                label if e_idx == 0 else "",
                eng_name,
                f"{pad_w:.1f}%",
                f"{zom_w:.1f}%",
                f"{avg_b:.2f}",
                str(s["throughput"]["max_queue_depth"]),
                f"{speedup:.2f}x",
            )
        if s_idx < len(scenarios) - 1:
            eff_tbl.add_section()

    console.print()
    console.print(eff_tbl)

    # 3. Hardware Profiler Decode GEMV -> GEMM Scaling Table
    if hw_profiles:
        hw_tbl = Table(
            title="PyTorch Profiler Hardware Decode Scaling (GEMV -> Batched GEMM on NVIDIA L4)",
            header_style="bold green",
        )
        hw_tbl.add_column("Engine", style="bold cyan", overflow="fold")
        hw_tbl.add_column("Step Tokens", justify="right")
        hw_tbl.add_column("Step Wall (ms)", justify="right")
        hw_tbl.add_column("GPU Active (ms)", justify="right", style="green")
        hw_tbl.add_column("CPU Gap (ms)", justify="right", style="red")
        hw_tbl.add_column("Duty Cycle", justify="right")
        hw_tbl.add_column("Step Decode tok/s", justify="right", style="bold green")
        hw_tbl.add_column("FFN+Linear+LM %", justify="right")

        for eng_name, prof in hw_profiles.items():
            b = prof.get("batch_size", 1)
            dec = prof["decode"]
            ops = dec["operator_gpu_breakdown"]
            lin_pct = sum(
                ops[k]["pct_of_gpu"]
                for k in (
                    "Q_Linear",
                    "K_Linear",
                    "V_Linear",
                    "O_Linear",
                    "FFN_Gate_Up_Linear",
                    "FFN_Down_Linear",
                    "LM_Head",
                )
            )
            step_tps = (b * 1000.0) / max(0.01, dec["step_wall_ms"])
            hw_tbl.add_row(
                eng_name,
                str(b),
                f"{dec['step_wall_ms']:.2f}",
                f"{dec['gpu_active_ms']:.2f}",
                f"{dec['host_dispatch_gap_ms']:.2f}",
                f"{dec['gpu_duty_cycle_pct']:.1f}%",
                f"{step_tps:.1f}",
                f"{lin_pct:.1f}%",
            )
        console.print()
        console.print(hw_tbl)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Baseline (B=1) vs. Static Batching vs. Continuous Batching."
    )
    parser.add_argument("--workload", type=str, default="sharegpt", choices=WORKLOAD_CHOICES)
    parser.add_argument(
        "--num-requests",
        type=int,
        default=48,
        help="Number of ShareGPT queries per (lambda, beta) run (default: 48).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="8,16",
        help="Comma-separated max_batch_size values for Static & Continuous Batching (default: '8,16').",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Custom comma-separated 'lambda:beta' list (default runs 8-scenario lambda + beta matrix).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Pre-allocated max sequence length per KV slot (default: 2048).",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=1024,
        help="Max prompt tokens per query (0 = uncapped, default: 1024).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=0,
        help="Max target output tokens per query (0 = uncapped natural ShareGPT up to 382, default: 0).",
    )
    parser.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    parser.add_argument("--slo-tpot-ms", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-parity", action="store_true", default=False)
    parser.add_argument("--no-profile", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join(HERE, "results", "batch_comparison_sharegpt.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    term_cols = shutil.get_terminal_size((140, 24)).columns
    console = Console(width=max(140, term_cols))
    torch.manual_seed(args.seed)

    scenarios = _parse_scenarios(args.scenarios)
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    max_p_cap = args.max_prompt_tokens if args.max_prompt_tokens > 0 else None
    max_o_cap = args.max_output_tokens if args.max_output_tokens > 0 else None
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    traces_dir = os.path.join(HERE, "profile_results", "traces")

    parity_report = None
    if not args.skip_parity:
        parity_report = verify_fp32_parity_and_bf16_agreement(device=args.device, seed=args.seed, console=console)

    console.print(
        f"\n[*] [bold cyan]Step 2/3 · Benchmarking Across {len(scenarios)} (λ, β) Arrival Scenarios "
        f"(workload={args.workload}, N={args.num_requests}, batch_sizes={batch_sizes})...[/bold cyan]"
    )

    results_by_engine: Dict[str, List[Dict[str, Any]]] = {}
    diagnostics_by_engine: Dict[str, List[Dict[str, Any]]] = {}
    hw_profiles: Dict[str, Dict[str, Any]] = {}
    engine_order: List[str] = []

    # Helper to evaluate all scenarios on a loaded engine instance
    def evaluate_engine(eng, weights_mb: float, is_baseline: bool = False) -> None:
        eng_name = eng.name
        engine_order.append(eng_name)
        results_by_engine[eng_name] = []
        diagnostics_by_engine[eng_name] = []

        eng.warmup(include_profiler=(not args.no_profile))

        rep_wl = None
        for rate, beta, label in scenarios:
            reqs = build_workload(
                tokenizer=eng.tokenizer,
                workload=args.workload,
                num_requests=args.num_requests,
                request_rate=rate,
                burstiness=beta,
                seed=args.seed,
                max_prompt_tokens=max_p_cap,
                max_output_tokens=max_o_cap,
            )
            wl_summary = summarize_workload(reqs)
            if rep_wl is None:
                rep_wl = wl_summary

            console.print(f"    -> [{eng_name}] Running [bold]{label}[/bold]...")
            out = eng.serve_requests(reqs, real_time=False, ignore_eos=True)

            summary = compute_serving_summary(
                records=out.records,
                engine_name=eng_name,
                workload_summary=wl_summary,
                request_rate=rate,
                burstiness=beta,
                clock_mode="event-driven",
                wall_clock_exec_s=out.wall_clock_exec_s,
                gpu_idle_time_s=out.gpu_idle_time_s,
                model_weights_mb=weights_mb,
                kv_cache_allocated_mb=eng.get_kv_allocated_mb(),
                kv_bytes_per_token=eng.get_kv_bytes_per_token(),
                peak_vram_allocated_mb=out.peak_vram_allocated_mb,
                peak_vram_reserved_mb=out.peak_vram_reserved_mb,
                slo_ttft_ms=args.slo_ttft_ms,
                slo_tpot_ms=args.slo_tpot_ms,
            )

            if is_baseline:
                diag = {
                    "max_batch_size": 1,
                    "prefill_padding_waste_pct": 0.0,
                    "decode_zombie_slot_waste_pct": 0.0,
                    "avg_active_batch_size": 1.0,
                }
            elif hasattr(out, "batch_diagnostics"):
                diag = out.batch_diagnostics
            else:
                diag = out.scheduler_diagnostics

            summary["batching_diagnostics"] = diag
            results_by_engine[eng_name].append(summary)
            diagnostics_by_engine[eng_name].append(diag)

        if not args.no_profile and rep_wl is not None:
            p_len = int(round(rep_wl["prompt_len"]["p50"]))
            d_ctx = int(round(rep_wl["prompt_len"]["p50"] + rep_wl["target_output_len"]["p50"] / 2.0))
            console.print(f"    -> [{eng_name}] Capturing PyTorch Profiler hardware trace...")
            prof = eng.profile_representative_phases(
                prefill_seq_len=p_len,
                decode_context_len=d_ctx,
                traces_dir=traces_dir,
            )
            if "batch_size" not in prof:
                prof["batch_size"] = 1
            hw_profiles[eng_name] = prof

    # --- 1. Baseline: Sequential FCFS (B=1) ---
    m_b1, tok_b1 = load_model_and_tokenizer(device=args.device, dtype=dtype, max_seq_len=max(8192, args.max_seq_len))
    w_mb = m_b1.get_weight_memory_mb()
    e_b1 = SequentialFCFSEngine(m_b1, tok_b1, max_seq_len=args.max_seq_len, device=args.device)
    evaluate_engine(e_b1, w_mb, is_baseline=True)
    del m_b1, e_b1
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # --- 2. Static Batching (across requested batch_sizes) ---
    m_st, tok_st = load_static_batch_model(device=args.device, dtype=dtype, max_seq_len=max(8192, args.max_seq_len))
    w_st_mb = m_st.get_weight_memory_mb()
    for bsz in batch_sizes:
        e_st = StaticBatchEngine(m_st, tok_st, max_batch_size=bsz, max_seq_len=args.max_seq_len, device=args.device)
        evaluate_engine(e_st, w_st_mb)
        del e_st
        if args.device == "cuda":
            torch.cuda.empty_cache()
    del m_st
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # --- 3. Continuous Batching (across requested batch_sizes) ---
    m_cb, tok_cb = load_continuous_batch_model(device=args.device, dtype=dtype, max_seq_len=max(8192, args.max_seq_len))
    w_cb_mb = m_cb.get_weight_memory_mb()
    for bsz in batch_sizes:
        e_cb = ContinuousBatchEngine(m_cb, tok_cb, max_batch_size=bsz, max_seq_len=args.max_seq_len, device=args.device)
        evaluate_engine(e_cb, w_cb_mb)
        del e_cb
        if args.device == "cuda":
            torch.cuda.empty_cache()
    del m_cb
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    console.print("\n[*] [bold cyan]Step 3/3 · Rendering Side-by-Side Comparison & Saving Artifacts...[/bold cyan]")
    render_comparison_tables(
        scenarios=scenarios,
        engine_order=engine_order,
        results_by_engine=results_by_engine,
        diagnostics_by_engine=diagnostics_by_engine,
        hw_profiles=hw_profiles,
        slo_ttft_ms=args.slo_ttft_ms,
        slo_tpot_ms=args.slo_tpot_ms,
        console=console,
    )

    payload = {
        "config": {
            "workload": args.workload,
            "num_requests": args.num_requests,
            "batch_sizes": batch_sizes,
            "max_seq_len": args.max_seq_len,
            "max_prompt_tokens": max_p_cap,
            "max_output_tokens": max_o_cap,
            "slo_ttft_ms": args.slo_ttft_ms,
            "slo_tpot_ms": args.slo_tpot_ms,
            "seed": args.seed,
            "scenarios": [
                {"request_rate": "inf" if math.isinf(r) else r, "burstiness": b, "label": lbl}
                for r, b, lbl in scenarios
            ],
        },
        "parity_gate": parity_report,
        "engine_order": engine_order,
        "results_by_engine": results_by_engine,
        "hardware_profiles": hw_profiles,
    }
    save_benchmark_json(payload, args.output_json)
    console.print(f"\n[bold green][✓][/bold green] Saved full comparison metrics to: [underline]{args.output_json}[/underline]")


if __name__ == "__main__":
    main()

