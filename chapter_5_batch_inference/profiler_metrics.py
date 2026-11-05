"""
Serving Evaluation Metrics, Queueing Analysis & Hardware/Operator Profiler for Chapter 5.

Implements industry-standard LLM serving metrics (vLLM `bench serve`, GenAI-Perf,
MLPerf Inference Server, and Sarathi-Serve):
1. Request-Level Timing Decomposition:
   - Queue Waiting Time:   t_start - t_arrival
   - Pure Prefill Time:    t_first_token - t_start
   - TTFT (End-to-End):    t_first_token - t_arrival  (= Queue Wait + Pure Prefill)
   - Pure Decode Time:     t_finish - t_first_token
   - GPU Service Time:     t_finish - t_start         (= Pure Prefill + Pure Decode)
   - E2E Request Latency:  t_finish - t_arrival       (= Queue Wait + GPU Service Time)
   - TPOT:                 (t_finish - t_first_token) / (output_len - 1)
   - ITL:                  t_k - t_{k-1} across all generated decode tokens (captures stall spikes)
   - Normalized Latency:   e2e_latency_ms / output_len
2. System-Level Throughput, Queueing & SLO Goodput:
   - Request throughput (req/s), Output token throughput (tok/s), Total token throughput (tok/s)
   - Server utilization (rho = gpu_busy / makespan), GPU idle time, peak & time-weighted queue depth
   - SLO attainment % and Goodput (req/s meeting both TTFT and TPOT targets)
3. Zero-Distortion PyTorch Hardware & Sub-Operator Profiler:
   - Runs isolated `torch.profiler` traces on representative Prefill and Decode steps so CUPTI/Kineto
     trace collection never pollutes queueing timestamps during the serving benchmark run.
"""

import json
import math
import os
import statistics
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

OPERATOR_CATEGORIES = [
    "Embedding",
    "RMSNorm_Attn",
    "Q_Linear",
    "K_Linear",
    "V_Linear",
    "RoPE",
    "KV_Cache_Update",
    "Attn_Compute",
    "O_Linear",
    "RMSNorm_FFN",
    "FFN_Gate_Up_Linear",
    "FFN_SiLU_Mul",
    "FFN_Down_Linear",
    "RMSNorm_Final",
    "LM_Head",
    "Sampling",
]

# NVIDIA L4 specs: 300 GB/s theoretical peak HBM/GDDR6 memory bandwidth
L4_THEORETICAL_BW_GBPS = 300.0


@dataclass
class RequestExecutionRecord:
    """Raw timestamps and token outputs recorded for a single request."""

    request_id: int
    source_id: str
    prompt_len: int
    target_output_len: int
    output_len: int
    arrival_time_s: float
    dispatch_time_s: float
    first_token_time_s: float
    finish_time_s: float
    token_times_s: List[float] = field(default_factory=list)
    generated_token_ids: List[int] = field(default_factory=list)

    @property
    def queue_wait_ms(self) -> float:
        return max(0.0, (self.dispatch_time_s - self.arrival_time_s) * 1000.0)

    @property
    def prefill_ms(self) -> float:
        return max(0.0, (self.first_token_time_s - self.dispatch_time_s) * 1000.0)

    @property
    def ttft_ms(self) -> float:
        return max(0.0, (self.first_token_time_s - self.arrival_time_s) * 1000.0)

    @property
    def decode_ms(self) -> float:
        return max(0.0, (self.finish_time_s - self.first_token_time_s) * 1000.0)

    @property
    def service_ms(self) -> float:
        return max(0.0, (self.finish_time_s - self.dispatch_time_s) * 1000.0)

    @property
    def e2e_latency_ms(self) -> float:
        return max(0.0, (self.finish_time_s - self.arrival_time_s) * 1000.0)

    @property
    def tpot_ms(self) -> Optional[float]:
        if self.output_len <= 1:
            return None
        return self.decode_ms / float(self.output_len - 1)

    @property
    def itl_ms_list(self) -> List[float]:
        if len(self.token_times_s) <= 1:
            return []
        return [
            max(0.0, (self.token_times_s[i] - self.token_times_s[i - 1]) * 1000.0)
            for i in range(1, len(self.token_times_s))
        ]

    @property
    def norm_e2e_ms_per_tok(self) -> float:
        return self.e2e_latency_ms / float(max(1, self.output_len))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_id": self.source_id,
            "prompt_len": self.prompt_len,
            "target_output_len": self.target_output_len,
            "output_len": self.output_len,
            "arrival_time_s": round(self.arrival_time_s, 6),
            "dispatch_time_s": round(self.dispatch_time_s, 6),
            "first_token_time_s": round(self.first_token_time_s, 6),
            "finish_time_s": round(self.finish_time_s, 6),
            "queue_wait_ms": round(self.queue_wait_ms, 3),
            "prefill_ms": round(self.prefill_ms, 3),
            "ttft_ms": round(self.ttft_ms, 3),
            "decode_ms": round(self.decode_ms, 3),
            "service_ms": round(self.service_ms, 3),
            "e2e_latency_ms": round(self.e2e_latency_ms, 3),
            "tpot_ms": round(self.tpot_ms, 3) if self.tpot_ms is not None else None,
            "norm_e2e_ms_per_tok": round(self.norm_e2e_ms_per_tok, 3),
            "generated_token_ids": self.generated_token_ids,
        }


# -----------------------------------------------------------------------------
# 1. Distribution & Queueing Math Helpers
# -----------------------------------------------------------------------------


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    w = rank - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def summarize_distribution(values: List[float]) -> Dict[str, float]:
    """Computes standard serving latency statistics (Mean, Std, Min, P50, P90, P95, P99, Max)."""
    if not values:
        return {k: 0.0 for k in ("mean", "std", "min", "p50", "p90", "p95", "p99", "max")}
    s = sorted(values)
    return {
        "mean": round(statistics.mean(s), 2),
        "std": round(statistics.pstdev(s) if len(s) > 1 else 0.0, 2),
        "min": round(s[0], 2),
        "p50": round(_percentile(s, 50), 2),
        "p90": round(_percentile(s, 90), 2),
        "p95": round(_percentile(s, 95), 2),
        "p99": round(_percentile(s, 99), 2),
        "max": round(s[-1], 2),
    }


def compute_queue_depth_stats(
    records: List[RequestExecutionRecord],
    makespan_s: float,
) -> Tuple[int, float]:
    """
    Computes exact continuous-time waiting-queue depth metrics:
    - `max_queue_depth`: max_t Q(t), peak number of arrived requests waiting before dispatch.
    - `avg_queue_depth`: time-weighted average (1/T) * integral Q(t) dt = sum(wait_i) / T (Little's Law).
    """
    events: List[Tuple[float, int]] = []
    total_wait_s = 0.0
    for r in records:
        wait_s = max(0.0, r.dispatch_time_s - r.arrival_time_s)
        total_wait_s += wait_s
        if wait_s > 1e-9:
            events.append((r.arrival_time_s, +1))
            events.append((r.dispatch_time_s, -1))

    # Process departures (-1) before arrivals (+1) at identical timestamps
    events.sort(key=lambda x: (x[0], x[1]))
    curr_q = 0
    max_q = 0
    for _, delta in events:
        curr_q += delta
        if curr_q > max_q:
            max_q = curr_q

    avg_q = (total_wait_s / makespan_s) if makespan_s > 0 else 0.0
    return max_q, round(avg_q, 3)


def merge_intervals(intervals: List[Tuple[float, float]]) -> float:
    """Returns total union length of potentially overlapping [start, end] intervals."""
    if not intervals:
        return 0.0
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    cur_s, cur_e = sorted_iv[0]
    total = 0.0
    for s, e in sorted_iv[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += max(0.0, cur_e - cur_s)
            cur_s, cur_e = s, e
    total += max(0.0, cur_e - cur_s)
    return total


# -----------------------------------------------------------------------------
# 2. Aggregate Serving & Memory Summary Computation
# -----------------------------------------------------------------------------


def compute_serving_summary(
    records: List[RequestExecutionRecord],
    engine_name: str,
    workload_summary: Dict[str, Any],
    request_rate: float,
    burstiness: float,
    clock_mode: str,
    wall_clock_exec_s: float,
    gpu_idle_time_s: float,
    model_weights_mb: float,
    kv_cache_allocated_mb: float,
    kv_bytes_per_token: int,
    peak_vram_allocated_mb: float,
    peak_vram_reserved_mb: float,
    slo_ttft_ms: float = 2000.0,
    slo_tpot_ms: float = 50.0,
) -> Dict[str, Any]:
    """Aggregates per-request records into standard industry serving & hardware KPIs."""
    n_reqs = len(records)
    min_arrival = min((r.arrival_time_s for r in records), default=0.0)
    max_finish = max((r.finish_time_s for r in records), default=0.0)
    makespan_s = max(1e-6, max_finish - min_arrival)

    # Union of active [dispatch_time_s, finish_time_s] intervals works for both B=1 and batched engines
    gpu_busy_s = merge_intervals([(r.dispatch_time_s, r.finish_time_s) for r in records])
    utilization_pct = min(100.0, (gpu_busy_s / makespan_s) * 100.0)

    total_prompt_tokens = sum(r.prompt_len for r in records)
    total_output_tokens = sum(r.output_len for r in records)
    total_decode_steps = sum(max(0, r.output_len - 1) for r in records)
    total_tokens = total_prompt_tokens + total_output_tokens

    queue_waits = [r.queue_wait_ms for r in records]
    prefills = [r.prefill_ms for r in records]
    ttfts = [r.ttft_ms for r in records]
    decodes = [r.decode_ms for r in records]
    services = [r.service_ms for r in records]
    e2es = [r.e2e_latency_ms for r in records]
    norm_e2es = [r.norm_e2e_ms_per_tok for r in records]
    tpots = [r.tpot_ms for r in records if r.tpot_ms is not None]
    pooled_itls = [itl for r in records for itl in r.itl_ms_list]

    max_q_depth, avg_q_depth = compute_queue_depth_stats(records, makespan_s)

    # SLO Goodput: requests satisfying BOTH TTFT <= slo_ttft_ms AND TPOT <= slo_tpot_ms
    slo_met = 0
    for r in records:
        tpot_ok = (r.tpot_ms is None) or (r.tpot_ms <= slo_tpot_ms)
        if r.ttft_ms <= slo_ttft_ms and tpot_ok:
            slo_met += 1
    slo_attainment_pct = (slo_met / n_reqs * 100.0) if n_reqs > 0 else 0.0
    goodput_rps = slo_met / makespan_s

    # KV Cache peak actual usage vs static pre-allocation
    # Note: for B=1 sequential, max concurrent tokens in cache is max_i(prompt_len_i + output_len_i).
    max_seq_touched = max((r.prompt_len + r.output_len for r in records), default=0)
    kv_peak_used_mb = (max_seq_touched * kv_bytes_per_token) / (1024.0 * 1024.0)
    kv_util_pct = (kv_peak_used_mb / kv_cache_allocated_mb * 100.0) if kv_cache_allocated_mb > 0 else 0.0

    # Effective HBM memory bandwidth during decode (streaming weights + avg KV history per decode step)
    mean_itl_s = (statistics.mean(pooled_itls) / 1000.0) if pooled_itls else 0.0
    avg_ctx_tokens = (
        statistics.mean(r.prompt_len + (r.output_len / 2.0) for r in records) if records else 0.0
    )
    bytes_streamed_per_decode_step = (model_weights_mb * 1024.0 * 1024.0) + (avg_ctx_tokens * kv_bytes_per_token)
    eff_decode_bw_gbps = (
        (bytes_streamed_per_decode_step / 1e9) / mean_itl_s if mean_itl_s > 0 else 0.0
    )

    total_decode_time_s = sum(r.decode_ms for r in records) / 1000.0
    active_decode_tps = (total_decode_steps / total_decode_time_s) if total_decode_time_s > 0 else 0.0

    return {
        "engine": engine_name,
        "request_rate_qps": "inf" if math.isinf(request_rate) else request_rate,
        "burstiness": burstiness,
        "clock_mode": clock_mode,
        "workload": workload_summary,
        "throughput": {
            "num_completed_requests": n_reqs,
            "benchmark_makespan_s": round(makespan_s, 3),
            "wall_clock_exec_s": round(wall_clock_exec_s, 3),
            "gpu_busy_time_s": round(gpu_busy_s, 3),
            "gpu_idle_time_s": round(gpu_idle_time_s, 3),
            "server_utilization_pct": round(utilization_pct, 1),
            "request_throughput_rps": round(n_reqs / makespan_s, 3),
            "output_token_throughput_tps": round(total_output_tokens / makespan_s, 2),
            "total_token_throughput_tps": round(total_tokens / makespan_s, 2),
            "active_decode_tps": round(active_decode_tps, 2),
            "max_queue_depth": max_q_depth,
            "avg_queue_depth": avg_q_depth,
        },
        "slo_goodput": {
            "slo_ttft_ms": slo_ttft_ms,
            "slo_tpot_ms": slo_tpot_ms,
            "requests_meeting_slo": slo_met,
            "slo_attainment_pct": round(slo_attainment_pct, 1),
            "goodput_rps": round(goodput_rps, 3),
        },
        "latencies_ms": {
            "queue_wait": summarize_distribution(queue_waits),
            "prefill": summarize_distribution(prefills),
            "ttft": summarize_distribution(ttfts),
            "decode_total": summarize_distribution(decodes),
            "service_total": summarize_distribution(services),
            "tpot": summarize_distribution(tpots),
            "itl": summarize_distribution(pooled_itls),
            "e2e_latency": summarize_distribution(e2es),
            "norm_e2e_per_tok": summarize_distribution(norm_e2es),
        },
        "hardware_memory": {
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "model_weights_mb": round(model_weights_mb, 1),
            "kv_cache_allocated_mb": round(kv_cache_allocated_mb, 2),
            "kv_cache_peak_used_mb": round(kv_peak_used_mb, 2),
            "kv_cache_utilization_pct": round(kv_util_pct, 1),
            "max_seq_len_touched": max_seq_touched,
            "peak_vram_allocated_mb": round(peak_vram_allocated_mb, 1),
            "peak_vram_reserved_mb": round(peak_vram_reserved_mb, 1),
            "effective_decode_bw_gbps": round(eff_decode_bw_gbps, 1),
            "l4_theoretical_bw_pct": round((eff_decode_bw_gbps / L4_THEORETICAL_BW_GBPS) * 100.0, 1),
        },
    }


# -----------------------------------------------------------------------------
# 3. Chrome Trace Hardware & Sub-Operator Decomposition
# -----------------------------------------------------------------------------


def analyze_chrome_trace(
    trace_path: str,
    step_label: str,
    native_wall_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Parses a PyTorch Chrome trace JSON to extract:
    1. Exact union-merged GPU kernel active time vs. host CPU launch/dispatch starvation gap.
       When `native_wall_ms` (measured with profiler OFF) is supplied, uses it as the true
       wall-clock denominator so CUPTI instrumentation overhead never distorts GPU duty cycle %.
    2. Sub-operator CUDA kernel breakdown via `cuda_runtime` -> `kernel` correlation IDs.
    """
    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    events = trace_data.get("traceEvents", [])
    root_ev = next(
        (e for e in events if e.get("name") == step_label and "ts" in e and "dur" in e),
        None,
    )

    if root_ev is not None:
        base_ts = float(root_ev["ts"])
        cpu_wall_us = float(root_ev["dur"])
    else:
        valid_ts = [float(e["ts"]) for e in events if "ts" in e]
        base_ts = min(valid_ts) if valid_ts else 0.0
        cpu_wall_us = (max(valid_ts) - base_ts) if valid_ts else 1.0

    # Collect GPU execution intervals (kernels + memcpy/memset)
    gpu_events = [
        e
        for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "ts" in e and "dur" in e
    ]
    gpu_intervals = [(float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in gpu_events]
    gpu_active_us = merge_intervals(gpu_intervals)

    if gpu_intervals:
        max_gpu_end = max(e for _, e in gpu_intervals)
        step_wall_us = max(cpu_wall_us, max_gpu_end - base_ts)
    else:
        step_wall_us = cpu_wall_us

    profiled_wall_ms = max(0.001, step_wall_us / 1000.0)
    gpu_active_ms = gpu_active_us / 1000.0
    step_wall_ms = max(gpu_active_ms, native_wall_ms) if native_wall_ms is not None else profiled_wall_ms
    host_gap_ms = max(0.0, step_wall_ms - gpu_active_ms)
    duty_cycle_pct = min(100.0, (gpu_active_ms / step_wall_ms) * 100.0)

    # Map correlation ID -> CPU launch timestamp
    corr_to_launch_ts: Dict[int, float] = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "ts" in e:
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                corr_to_launch_ts[int(corr)] = float(e["ts"])

    # Collect user_annotation scopes matching our OPERATOR_CATEGORIES
    op_scopes: List[Tuple[float, float, str]] = []
    cat_set = set(OPERATOR_CATEGORIES)
    for e in events:
        name = e.get("name", "")
        if name in cat_set and "ts" in e and "dur" in e:
            s = float(e["ts"])
            op_scopes.append((s, s + float(e["dur"]), name))
    op_scopes.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    op_gpu_us: Dict[str, float] = OrderedDict((cat, 0.0) for cat in OPERATOR_CATEGORIES)
    unattributed_us = 0.0

    for kev in gpu_events:
        dur_us = float(kev["dur"])
        corr = kev.get("args", {}).get("correlation")
        launch_ts = corr_to_launch_ts.get(int(corr)) if corr is not None else None
        matched_op: Optional[str] = None
        if launch_ts is not None:
            # Innermost matching operator annotation containing launch_ts
            for s, e, name in op_scopes:
                if s > launch_ts:
                    break
                if s <= launch_ts <= e:
                    matched_op = name
        if matched_op is not None:
            op_gpu_us[matched_op] += dur_us
        else:
            unattributed_us += dur_us

    total_kernel_sum_ms = sum(op_gpu_us.values()) / 1000.0
    op_breakdown = {}
    denom_ms = max(1e-6, total_kernel_sum_ms)
    for cat, us in op_gpu_us.items():
        ms = us / 1000.0
        op_breakdown[cat] = {
            "gpu_ms": round(ms, 3),
            "pct_of_gpu": round((ms / denom_ms) * 100.0, 1),
        }

    return {
        "trace_path": trace_path,
        "step_wall_ms": round(step_wall_ms, 2),
        "profiled_wall_ms": round(profiled_wall_ms, 2),
        "gpu_active_ms": round(gpu_active_ms, 2),
        "host_dispatch_gap_ms": round(host_gap_ms, 2),
        "gpu_duty_cycle_pct": round(duty_cycle_pct, 1),
        "num_cuda_kernels": len([e for e in gpu_events if e.get("cat") == "kernel"]),
        "operator_gpu_breakdown": op_breakdown,
        "unattributed_gpu_ms": round(unattributed_us / 1000.0, 3),
    }


# -----------------------------------------------------------------------------
# 4. Rich Terminal Reporting & JSON Export
# -----------------------------------------------------------------------------


def _fmt_qps(qps: Any) -> str:
    if qps == "inf" or (isinstance(qps, float) and math.isinf(qps)):
        return "inf (Burst)"
    return f"{float(qps):.2f} req/s"


def print_single_run_report(
    summary: Dict[str, Any],
    phase_profile: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
) -> None:
    """Renders a clean, comprehensive Rich terminal report for a serving run."""
    console = console or Console()
    wl = summary["workload"]
    tp = summary["throughput"]
    slo = summary["slo_goodput"]
    lat = summary["latencies_ms"]
    hw = summary["hardware_memory"]

    header = (
        f"[bold cyan]Engine:[/bold cyan] {summary['engine']}   "
        f"[bold cyan]Workload:[/bold cyan] {wl['workload']} (N={wl['num_requests']})   "
        f"[bold cyan]Arrival Rate:[/bold cyan] {_fmt_qps(summary['request_rate_qps'])}   "
        f"[bold cyan]Clock:[/bold cyan] {summary['clock_mode']}\n"
        f"[dim]Prompts:[/dim] mean={wl['prompt_len']['mean']} tok (p50={wl['prompt_len']['p50']}, "
        f"p90={wl['prompt_len']['p90']}, max={wl['prompt_len']['max']}, CV={wl['prompt_len']['cv']})   "
        f"[dim]Outputs:[/dim] mean={wl['target_output_len']['mean']} tok (p50={wl['target_output_len']['p50']}, "
        f"p90={wl['target_output_len']['p90']}, max={wl['target_output_len']['max']}, CV={wl['target_output_len']['cv']})"
    )
    console.print()
    console.print(Panel(header, title="[bold white]Chapter 5 · Batch Inference Baseline Evaluation[/bold white]", border_style="cyan"))

    # 1. System Throughput, Queueing & Memory Summary Table
    kpi_tbl = Table(title="System Throughput, Queueing & Hardware Footprint", header_style="bold magenta", expand=True)
    kpi_tbl.add_column("Throughput & Queueing", style="white")
    kpi_tbl.add_column("Value", justify="right", style="bold green")
    kpi_tbl.add_column("Hardware, Memory & Goodput", style="white")
    kpi_tbl.add_column("Value", justify="right", style="bold yellow")

    kpi_tbl.add_row(
        "Request Throughput",
        f"{tp['request_throughput_rps']:.3f} req/s",
        "GPU Device",
        hw["gpu_name"],
    )
    kpi_tbl.add_row(
        "Output Token Throughput",
        f"{tp['output_token_throughput_tps']:.2f} tok/s",
        "Model Weights / Peak VRAM",
        f"{hw['model_weights_mb']:.0f} / {hw['peak_vram_allocated_mb']:.0f} MiB",
    )
    kpi_tbl.add_row(
        "Total Token Throughput (In+Out)",
        f"{tp['total_token_throughput_tps']:.2f} tok/s",
        "KV Cache (Peak Used / Alloc)",
        f"{hw['kv_cache_peak_used_mb']:.1f} / {hw['kv_cache_allocated_mb']:.1f} MiB ({hw['kv_cache_utilization_pct']:.1f}%)",
    )
    kpi_tbl.add_row(
        "Active GPU Decode Speed",
        f"{tp['active_decode_tps']:.2f} tok/s",
        "Decode Effective HBM BW",
        f"{hw['effective_decode_bw_gbps']:.1f} GB/s ({hw['l4_theoretical_bw_pct']:.1f}% L4 peak)",
    )
    kpi_tbl.add_row(
        "Makespan / Wall-Clock Time",
        f"{tp['benchmark_makespan_s']:.2f}s / {tp['wall_clock_exec_s']:.2f}s",
        f"SLO Attainment (TTFT≤{slo['slo_ttft_ms']:.0f}ms, TPOT≤{slo['slo_tpot_ms']:.0f}ms)",
        f"{slo['slo_attainment_pct']:.1f}% ({slo['requests_meeting_slo']}/{tp['num_completed_requests']})",
    )
    kpi_tbl.add_row(
        "Server Busy Util / Queue Peak (Avg)",
        f"{tp['server_utilization_pct']:.1f}% / {tp['max_queue_depth']} ({tp['avg_queue_depth']:.2f})",
        "SLO Goodput",
        f"{slo['goodput_rps']:.3f} req/s",
    )
    console.print(kpi_tbl)

    # 2. Request Latency & Waiting Time Percentiles Table
    lat_tbl = Table(
        title="Request Latency & Queueing Breakdown (milliseconds)",
        header_style="bold cyan",
    )
    lat_tbl.add_column("Metric", style="bold white")
    lat_tbl.add_column("Mean", justify="right", overflow="fold")
    lat_tbl.add_column("P50", justify="right", style="green", overflow="fold")
    lat_tbl.add_column("P90", justify="right", style="yellow", overflow="fold")
    lat_tbl.add_column("P95", justify="right", style="yellow", overflow="fold")
    lat_tbl.add_column("P99", justify="right", style="bold red", overflow="fold")
    lat_tbl.add_column("Max", justify="right", style="red", overflow="fold")

    metric_rows = [
        ("Queue Wait (t_start - t_arr)", "queue_wait"),
        ("Pure Prefill (Step 0 GPU)", "prefill"),
        ("TTFT (Wait + Prefill)", "ttft"),
        ("Pure Decode (Steps 1..M-1)", "decode_total"),
        ("Total GPU Service Time", "service_total"),
        ("TPOT (ms / out tok)", "tpot"),
        ("ITL (Step-to-step tok)", "itl"),
        ("End-to-End Latency", "e2e_latency"),
        ("Normalized E2E (ms/tok)", "norm_e2e_per_tok"),
    ]
    for label, key in metric_rows:
        d = lat[key]
        lat_tbl.add_row(
            label,
            f"{d['mean']:.1f}",
            f"{d['p50']:.1f}",
            f"{d['p90']:.1f}",
            f"{d['p95']:.1f}",
            f"{d['p99']:.1f}",
            f"{d['max']:.1f}",
        )
    console.print(lat_tbl)

    # 3. Sampled PyTorch Profiler Hardware Decomposition (if enabled)
    if phase_profile:
        pre = phase_profile["prefill"]
        dec = phase_profile["decode"]
        prof_tbl = Table(
            title=(
                f"PyTorch Profiler Hardware Decomposition · "
                f"Prefill (len={phase_profile['prefill_seq_len']}) vs Decode (ctx={phase_profile['decode_context_len']})"
            ),
            header_style="bold green",
            expand=True,
        )
        prof_tbl.add_column("Stage / Operator", style="white")
        prof_tbl.add_column("Prefill GPU (ms)", justify="right")
        prof_tbl.add_column("Prefill %", justify="right", style="cyan")
        prof_tbl.add_column("Decode GPU (ms)", justify="right")
        prof_tbl.add_column("Decode %", justify="right", style="yellow")

        prof_tbl.add_row(
            "[bold]Step Wall Latency[/bold]",
            f"[bold]{pre['step_wall_ms']:.2f}[/bold]",
            "100.0%",
            f"[bold]{dec['step_wall_ms']:.2f}[/bold]",
            "100.0%",
        )
        prof_tbl.add_row(
            "[bold green]GPU Kernel Active Time[/bold green]",
            f"[bold green]{pre['gpu_active_ms']:.2f}[/bold green]",
            f"[bold green]{pre['gpu_duty_cycle_pct']:.1f}%[/bold green]",
            f"[bold green]{dec['gpu_active_ms']:.2f}[/bold green]",
            f"[bold green]{dec['gpu_duty_cycle_pct']:.1f}%[/bold green]",
        )
        prof_tbl.add_row(
            "[bold red]Host CPU Dispatch / Launch Gap[/bold red]",
            f"[bold red]{pre['host_dispatch_gap_ms']:.2f}[/bold red]",
            f"[bold red]{100.0 - pre['gpu_duty_cycle_pct']:.1f}%[/bold red]",
            f"[bold red]{dec['host_dispatch_gap_ms']:.2f}[/bold red]",
            f"[bold red]{100.0 - dec['gpu_duty_cycle_pct']:.1f}%[/bold red]",
        )
        prof_tbl.add_section()

        for cat in OPERATOR_CATEGORIES:
            p_op = pre["operator_gpu_breakdown"][cat]
            d_op = dec["operator_gpu_breakdown"][cat]
            if p_op["gpu_ms"] >= 0.02 or d_op["gpu_ms"] >= 0.02:
                prof_tbl.add_row(
                    cat,
                    f"{p_op['gpu_ms']:.3f}",
                    f"{p_op['pct_of_gpu']:.1f}%",
                    f"{d_op['gpu_ms']:.3f}",
                    f"{d_op['pct_of_gpu']:.1f}%",
                )
        console.print(prof_tbl)


def print_sweep_report(
    sweep_summaries: List[Dict[str, Any]],
    console: Optional[Console] = None,
) -> None:
    """Renders the classic Latency vs. Throughput (QPS) knee curve summary table."""
    console = console or Console()
    if not sweep_summaries:
        return

    first = sweep_summaries[0]
    wl = first["workload"]
    tbl = Table(
        title=(
            f"QPS Sweep · Engine={first['engine']} · "
            f"Workload={wl['workload']} (N={wl['num_requests']})"
        ),
        header_style="bold cyan",
    )
    tbl.add_column("QPS", style="bold white", justify="right", overflow="fold")
    tbl.add_column("req/s", justify="right", style="green", overflow="fold")
    tbl.add_column("tok/s", justify="right", style="bold green", overflow="fold")
    tbl.add_column("Util", justify="right", overflow="fold")
    tbl.add_column("MaxQ", justify="right", overflow="fold")
    tbl.add_column("Wait P50/P99", justify="right", style="yellow", overflow="fold")
    tbl.add_column("TTFT P50/P99", justify="right", style="bold yellow", overflow="fold")
    tbl.add_column("TPOT P50", justify="right", overflow="fold")
    tbl.add_column("Goodput", justify="right", style="cyan", overflow="fold")

    for s in sweep_summaries:
        tp = s["throughput"]
        lat = s["latencies_ms"]
        slo = s["slo_goodput"]
        q_lbl = "inf" if s["request_rate_qps"] == "inf" else f"{float(s['request_rate_qps']):.2f}"
        tbl.add_row(
            q_lbl,
            f"{tp['request_throughput_rps']:.2f}",
            f"{tp['output_token_throughput_tps']:.1f}",
            f"{tp['server_utilization_pct']:.0f}%",
            str(tp["max_queue_depth"]),
            f"{lat['queue_wait']['p50']:.0f}/{lat['queue_wait']['p99']:.0f}ms",
            f"{lat['ttft']['p50']:.0f}/{lat['ttft']['p99']:.0f}ms",
            f"{lat['tpot']['p50']:.1f}ms",
            f"{slo['goodput_rps']:.2f} ({slo['slo_attainment_pct']:.0f}%)",
        )

    console.print()
    console.print(tbl)


def save_benchmark_json(payload: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
