"""
Profile Visualizer for tiny_vllm.

Extracts fine-grained token-wise latency, sub-operator time composition,
and microsecond-level execution & memory timelines (Gantt chart) from PyTorch Profiler traces:
- CPU Main Thread (Async enqueue and sync wait)
- Memory Track (PCIe HtoD/DtoH transfers, VRAM weight & KV cache streaming)
- Compute Track (Tensor Core GEMMs and Vector ALUs)
- Active VRAM Footprint (caching allocator memory over time)
"""

import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


FINE_GRAINED_CATEGORIES = [
    "Embedding",
    "RMSNorm_Attn",
    "QKV_Linear",
    "RoPE",
    "Attn_Compute",
    "O_Linear",
    "RMSNorm_FFN",
    "FFN_Gate_Up_Linear",
    "FFN_SiLU_Mul",
    "FFN_Down_Linear",
    "RMSNorm_Final",
    "LM_Head",
    "Sampling",
    "Tokenizer_Decode",
]

GPU_CATEGORIES = {
    "Embedding",
    "RMSNorm_Attn",
    "QKV_Linear",
    "RoPE",
    "Attn_Compute",
    "O_Linear",
    "RMSNorm_FFN",
    "FFN_Gate_Up_Linear",
    "FFN_SiLU_Mul",
    "FFN_Down_Linear",
    "RMSNorm_Final",
    "LM_Head",
    "Sampling",
}

CPU_CATEGORIES = {
    "Tokenizer_Decode",
    "Tokenizer_Encode",
}

CATEGORY_COLORS = {
    "Embedding": "#7f8c8d",
    "RMSNorm_Attn": "#1abc9c",
    "QKV_Linear": "#e74c3c",
    "RoPE": "#c0392b",
    "Attn_Compute": "#ff7675",
    "O_Linear": "#d63031",
    "RMSNorm_FFN": "#2ecc71",
    "FFN_Gate_Up_Linear": "#f39c12",
    "FFN_SiLU_Mul": "#e67e22",
    "FFN_Down_Linear": "#d35400",
    "RMSNorm_Final": "#27ae60",
    "LM_Head": "#9b59b6",
    "Sampling": "#3498db",
    "Tokenizer_Decode": "#00cec9",
}

CATEGORY_TERMINAL_STYLES = {
    "Embedding": "dim white",
    "RMSNorm_Attn": "bold cyan",
    "QKV_Linear": "bold red",
    "RoPE": "red",
    "Attn_Compute": "bold bright_red",
    "O_Linear": "dark_red",
    "RMSNorm_FFN": "bold green",
    "FFN_Gate_Up_Linear": "bold yellow",
    "FFN_SiLU_Mul": "bold orange3",
    "FFN_Down_Linear": "bold orange_red1",
    "RMSNorm_Final": "green",
    "LM_Head": "bold magenta",
    "Sampling": "bold blue",
    "Tokenizer_Decode": "dim cyan",
}

OPERATION_METADATA = {
    "FFN_Gate_Up_Linear": {
        "name": "FFN Gate & Up Projections",
        "category": "Feed-Forward (GEMV)",
        "badge": "FFN",
        "desc": "SwiGLU gate_proj & up_proj matrix multiplication across 16 layers (2048 -> 8192)",
        "scaling": "Flat (Memory-bandwidth bound streaming 2.46 GB weights)",
    },
    "FFN_Down_Linear": {
        "name": "FFN Down Projection",
        "category": "Feed-Forward (GEMV)",
        "badge": "FFN",
        "desc": "SwiGLU down_proj matrix multiplication across 16 layers (8192 -> 2048)",
        "scaling": "Flat (Memory-bandwidth bound streaming weights)",
    },
    "LM_Head": {
        "name": "LM Head Unembedding",
        "category": "Output Projection",
        "badge": "Head",
        "desc": "Linear projection from hidden dimension 2048 to 128,256 vocabulary logits",
        "scaling": "Flat (Streams 525 MB vocabulary weights per token)",
    },
    "QKV_Linear": {
        "name": "Q, K, V Projections",
        "category": "Attention Projections",
        "badge": "Attn",
        "desc": "Query (2048->2048), Key (2048->512), Value (2048->512) projections across 16 layers",
        "scaling": "Flat (Memory-bandwidth bound streaming weights)",
    },
    "O_Linear": {
        "name": "Attention Output Projection",
        "category": "Attention Projections",
        "badge": "Attn",
        "desc": "Multi-head attention output projection across 16 layers (2048 -> 2048)",
        "scaling": "Flat (Memory-bandwidth bound streaming weights)",
    },
    "Attn_Compute": {
        "name": "Attention Dot-Product & KV Cache",
        "category": "Attention Mechanism",
        "badge": "Attn",
        "desc": "Scaled dot-product attention (Q*K^T / sqrt(d)), causal mask, softmax, and P*V across KV cache",
        "scaling": "Linear O(N) with KV cache context length",
    },
    "RoPE": {
        "name": "Rotary Position Embedding",
        "category": "Positional Embedding",
        "badge": "RoPE",
        "desc": "Rotary position embedding (complex rotations applied to query and key vectors)",
        "scaling": "Flat (Fast on-chip vector math)",
    },
    "RMSNorm_Attn": {
        "name": "Pre-Attention RMSNorm",
        "category": "Normalization",
        "badge": "Norm",
        "desc": "Root-mean-square normalization preceding attention blocks across 16 layers",
        "scaling": "Flat (Fast memory-bandwidth bound vector kernel)",
    },
    "RMSNorm_FFN": {
        "name": "Pre-FFN RMSNorm",
        "category": "Normalization",
        "badge": "Norm",
        "desc": "Root-mean-square normalization preceding feed-forward blocks across 16 layers",
        "scaling": "Flat (Fast memory-bandwidth bound vector kernel)",
    },
    "RMSNorm_Final": {
        "name": "Final Model RMSNorm",
        "category": "Normalization",
        "badge": "Norm",
        "desc": "Final root-mean-square normalization applied to hidden states before LM head",
        "scaling": "Flat (Single vector normalization)",
    },
    "FFN_SiLU_Mul": {
        "name": "SiLU Activation & Gating",
        "category": "Activation Function",
        "badge": "FFN",
        "desc": "Elementwise SiLU(gate) * up gating vector multiplication across 16 layers",
        "scaling": "Flat (Vector memory bandwidth bound)",
    },
    "Sampling": {
        "name": "Token Sampling & Sync Barrier",
        "category": "Token Sampling",
        "badge": "Sample",
        "desc": "Greedy argmax / multinomial distribution sampling and blocking next_token.item() DtoH sync",
        "scaling": "Flat (Dominated by host-GPU synchronization barrier)",
    },
    "Embedding": {
        "name": "Token Embedding Lookup",
        "category": "Input Embedding",
        "badge": "Embed",
        "desc": "Lookup of input token ID embedding vectors from 128,256 x 2048 embedding matrix",
        "scaling": "Flat (Single indexed memory read)",
    },
    "Tokenizer_Decode": {
        "name": "Tokenizer Detokenization",
        "category": "Host Detokenization",
        "badge": "CPU",
        "desc": "CPU HuggingFace BPE token ID to UTF-8 string detokenization",
        "scaling": "Flat (Host CPU execution)",
    },
}


def extract_single_token_metric(
    prof: torch.profiler.profile,
    step_idx: int,
    token_id: Optional[int] = None,
    token_text: str = "",
) -> Dict[str, Any]:
    """Extracts aggregate fine-grained timings from a single profiled step."""
    cat_times: Dict[str, float] = OrderedDict((cat, 0.0) for cat in FINE_GRAINED_CATEGORIES)

    def accumulate_child_times(event):
        if event.name in GPU_CATEGORIES:
            cat_times[event.name] += event.device_time_total / 1000.0  # us to ms
            return
        elif event.name in CPU_CATEGORIES:
            cat_times[event.name] += event.cpu_time_total / 1000.0  # us to ms
            return

        for child in event.cpu_children:
            accumulate_child_times(child)

    for event in prof.events():
        if event.name.startswith("token_") and len(event.cpu_children) > 0:
            accumulate_child_times(event)

    total_latency_ms = sum(cat_times.values())
    total_device_ms = sum(cat_times[c] for c in GPU_CATEGORIES if c in cat_times)
    total_cpu_ms = sum(cat_times[c] for c in CPU_CATEGORIES if c in cat_times)
    pure_compute_est_ms = 0.021
    mem_wait_est_ms = max(0.0, total_device_ms - pure_compute_est_ms)
    cpu_idle_est_ms = total_cpu_ms
    tot_ms = max(0.001, total_device_ms + total_cpu_ms)
    default_three = {
        "total_latency_ms": round(tot_ms, 2),
        "cpu_idle_ms": round(cpu_idle_est_ms, 2),
        "cpu_idle_pct": round((cpu_idle_est_ms / tot_ms * 100), 1),
        "memory_wait_ms": round(mem_wait_est_ms, 2),
        "memory_wait_pct": round((mem_wait_est_ms / tot_ms * 100), 1),
        "compute_ms": round(pure_compute_est_ms, 3),
        "compute_pct": round((pure_compute_est_ms / tot_ms * 100), 2),
        "duty_cycle_pct": round((total_device_ms / tot_ms * 100), 1),
    }

    return {
        "step": step_idx,
        "token_id": token_id,
        "token_text": token_text,
        "total_latency_ms": total_latency_ms,
        "breakdown": cat_times,
        "three_metrics": default_three,
    }


def extract_timeline_from_trace(
    trace_path: str,
    step_idx: int,
    token_id: Optional[int] = None,
    token_text: str = "",
    seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Parses the exported Chrome trace JSON into distinct execution tracks:
    1. CPU Main Thread (Dispatch & Sync Stall)
    2. Memory Track (PCIe transfers + VRAM weight & KV cache streaming)
    3. Compute Track (GPU Tensor Cores & Vector ALUs)
    4. Active VRAM Footprint Curve
    """
    if not os.path.exists(trace_path):
        return {}

    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    if not events:
        return {}

    # 1. Establish base timestamp and total step duration
    token_ev = next((e for e in events if e.get("name") == f"token_{step_idx}" and e.get("cat") == "user_annotation"), None)
    # Find the token_{step_idx} root event to establish base timestamp and step window
    token_ev = next((e for e in events if e.get("name") == f"token_{step_idx}"), None)
    if not token_ev:
        token_ev = next((e for e in events if e.get("name", "").startswith("token_") and e.get("cat") == "user_annotation"), None)
        token_ev = next((e for e in events if e.get("name", "").startswith("token_")), None)

    if token_ev and "ts" in token_ev and "dur" in token_ev:
        base_ts = token_ev["ts"]
        cpu_dur_us = token_ev["dur"]
        total_dur_us = token_ev["dur"]
    else:
        valid_ts = [e["ts"] for e in events if "ts" in e]
        base_ts = min(valid_ts) if valid_ts else 0
        cpu_dur_us = 1000
        total_dur_us = (max(valid_ts) - base_ts) if valid_ts else 1000

    # Find maximum GPU timestamp across kernels, gpu_memcpy, and gpu_user_annotations
    gpu_events = [
        e for e in events
        if e.get("cat") in ["kernel", "gpu_memcpy", "gpu_user_annotation"]
        and "ts" in e and "dur" in e
    ]
    if gpu_events:
        max_gpu_end_ts = max(e["ts"] + e["dur"] for e in gpu_events)
        total_dur_us = max(cpu_dur_us, max_gpu_end_ts - base_ts)
    else:
        total_dur_us = cpu_dur_us
    total_dur_ms = max(0.01, total_dur_us / 1000.0)
    end_ts = base_ts + total_dur_us

    total_dur_ms = max(0.01, total_dur_us / 1000.0)
    timeline_events = []

    # 2. Track 1: CPU Main Thread (Dispatch & Stall)
    # 1. CPU User Annotations (Sub-operators)
    cpu_ops = [
        e for e in events
        if e.get("cat") == "user_annotation"
        and "ts" in e and "dur" in e
        and not e.get("name", "").startswith("token_")
        and e["ts"] >= base_ts - 100 and e["ts"] <= end_ts + 200
    ]
    parent_cpu_ops = []
    seen_ops = set()
    for op in sorted(cpu_ops, key=lambda x: x["ts"]):
        name = op.get("name", "")
        if name not in seen_ops or len(parent_cpu_ops) < 200:
            parent_cpu_ops.append(op)
            seen_ops.add(name)

    # Detect CPU Sync Stall (Host waiting for .item() / GPU stream synchronization)
    sync_evs = [
        e for e in events
        if e.get("name") in ["cudaStreamSynchronize", "cudaDeviceSynchronize"]
        and e.get("cat") == "cuda_runtime"
        and "ts" in e and "dur" in e
        and (e["dur"] / 1000.0) > 0.05
    ]
    primary_sync = max(sync_evs, key=lambda x: x["dur"]) if sync_evs else None

    last_cpu_end_ms = 0.0
    for op in parent_cpu_ops:
        rel_start = max(0.0, (op["ts"] - base_ts) / 1000.0)
        rel_dur = max(0.005, op["dur"] / 1000.0)
        # If this op encompasses the sync event, trim dispatch duration
        if primary_sync and op["ts"] <= primary_sync["ts"] and (op["ts"] + op["dur"]) >= (primary_sync["ts"] + primary_sync["dur"]):
            rel_dur = max(0.005, (primary_sync["ts"] - op["ts"]) / 1000.0)

        rel_dur = max(0.01, op["dur"] / 1000.0)
        last_cpu_end_ms = max(last_cpu_end_ms, rel_start + rel_dur)
        timeline_events.append({
            "row": "cpu",
            "name": f"{op['name']}",
            "sub": "Python op dispatch",
            "cat": "cat-cpu-dispatch",
            "start_ms": round(rel_start, 3),
            "dur_ms": round(rel_dur, 3),
            "domain": "COMPUTE (CPU Host)",
            "step": f"Dispatch: {op['name']}",
            "other": "Enqueuing CUDA commands",
            "hw": "Host CPU Thread",
            "vram_note": "Launch overhead",
        })

    if primary_sync:
        stall_start_ms = max(0.0, (primary_sync["ts"] - base_ts) / 1000.0)
        stall_dur = primary_sync["dur"] / 1000.0
    else:
        stall_start_ms = max(0.0, last_cpu_end_ms)
        stall_dur = max(0.0, total_dur_ms - last_cpu_end_ms)

    # Detect CPU Sync Stall (Host waiting for .item() synchronization)
    stall_dur = max(0.0, total_dur_ms - last_cpu_end_ms)
    if stall_dur > 0.05:
        timeline_events.append({
            "row": "cpu",
            "name": "CPU Sync Stall: Waiting for .item()",
            "sub": "Host thread blocked waiting for GPU (.item())",
            "cat": "cat-cpu-stall",
            "start_ms": round(last_cpu_end_ms, 3),
            "dur_ms": round(stall_dur, 3),
            "domain": "STALL (Host Synchronizing)",
            "step": "Sampling sync barrier (.item())",
            "other": "GPU is finishing execution",
            "hw": "Host CPU Thread (Blocked/Sleeping)",
            "vram_note": "Blocked on DtoH token return",
        })

    # 3. Track 2: Memory Track (PCIe transfers + VRAM weight & KV cache streaming)
    # 2. PCIe Memory Transfers (gpu_memcpy)
    gpu_memcpys = [
        e for e in events
        if e.get("cat") in ["gpu_memcpy", "memcpy"]
        and "ts" in e and "dur" in e
    ]
    total_pcie_ms = 0.0
    for m in gpu_memcpys:
        rel_start = max(0.0, (m["ts"] - base_ts) / 1000.0)
        rel_dur = max(0.005, m["dur"] / 1000.0)
        total_pcie_ms += rel_dur
        name = m.get("name", "Memcpy")
        is_htod = "HtoD" in name
        timeline_events.append({
            "row": "memory-ops",
            "name": f"PCIe: {name}",
            "sub": "DMA engine transfer over PCIe",
            "cat": "cat-mem-pcie",
            "start_ms": round(rel_start, 3),
            "dur_ms": round(rel_dur, 3),
            "domain": "MEMORY (PCIe HtoD)" if is_htod else "MEMORY (PCIe DtoH)",
            "step": "PCIe Transfer",
            "other": "DMA Engine active",
            "hw": "PCIe Gen4 x16 DMA Engine",
            "vram_note": "Inter-device memory copy",
        })

    # 4. GPU Operations (Compute Track & VRAM Streaming on Memory Track)
    gpu_annots = sorted(
        [
            e for e in events
            if e.get("cat") == "gpu_user_annotation"
            and not e.get("name", "").startswith("token_")
            and "ts" in e and "dur" in e
        ],
        key=lambda x: x["ts"],
    )
    # 3. GPU Compute (Kernels)
    gpu_kernels = [
        e for e in events
        if e.get("cat") == "kernel"
        and "ts" in e and "dur" in e
    ]
    total_kernel_us = sum(k["dur"] for k in gpu_kernels)
    total_kernel_ms = total_kernel_us / 1000.0

    # 4. Calculate the Three Physical Metrics
    # Metric 1: CPU Launch & Driver Gaps (Host dispatch starvation / empty GPU pipeline)
    cpu_idle_ms = max(0.0, total_dur_ms - total_kernel_ms)

    # Metric 3: Pure Math Compute (Tensor Cores & Vector ALUs doing arithmetic)
    # LLaMA-3.2-1B: 1.23B params -> ~2.46 GFLOPs. On NVIDIA L4 (120 TFLOP/s peak BF16):
    eff_seq_len = seq_len if seq_len else 1
    step_flops = 2 * 1.23e9 + (4 * 16 * eff_seq_len * 2048 if step_idx > 0 else 2 * 1.23e9 * eff_seq_len)
    pure_compute_ms = (step_flops / 120e12) * 1000.0  # ~0.021 ms for decode
    pure_compute_ms = min(pure_compute_ms, total_kernel_ms)

    # Metric 2: VRAM Data Wait (GPU execution stalled on memory controller / DRAM bandwidth)
    mem_wait_ms = max(0.0, total_kernel_ms - pure_compute_ms) + total_pcie_ms

    # Normalized percentages of the total token wall-clock latency
    cpu_idle_pct = round((cpu_idle_ms / total_dur_ms) * 100.0, 1) if total_dur_ms > 0 else 0.0
    mem_wait_pct = round((mem_wait_ms / total_dur_ms) * 100.0, 1) if total_dur_ms > 0 else 0.0
    compute_pct = round(max(0.01, 100.0 - cpu_idle_pct - mem_wait_pct), 2)
    duty_cycle_pct = round((total_kernel_ms / total_dur_ms) * 100.0, 1) if total_dur_ms > 0 else 0.0

    # Populate GPU Compute & Memory timeline events
    current_layer = 0
    for op in gpu_annots:
        op_name = op.get("name", "")
        rel_start = max(0.0, (op["ts"] - base_ts) / 1000.0)
        rel_dur = max(0.005, op["dur"] / 1000.0)

        if op_name in ["Embedding", "RMSNorm_Final", "LM_Head", "Sampling"]:
            layer_label = op_name
        else:
            layer_label = f"L{current_layer}: {op_name}"

        is_gemm = any(term in op_name for term in ["Linear", "Head", "proj", "GEMM"])
        cat = "cat-compute-gemm" if is_gemm else "cat-compute-alu"
        domain = "COMPUTE (Tensor Cores / GEMM)" if is_gemm else "COMPUTE (Vector ALUs / SMs)"
        hw = "NVIDIA L4 Tensor Cores" if is_gemm else "NVIDIA L4 CUDA Cores / Vector ALUs"

        timeline_events.append({
            "row": "compute-ops",
            "name": layer_label,
            "sub": "Matrix multiplication" if is_gemm else "Vector arithmetic",
            "cat": cat,
            "start_ms": round(rel_start, 3),
            "dur_ms": round(rel_dur, 3),
            "domain": domain,
            "step": f"Kernel Math: {layer_label}",
            "other": "Pipelined with VRAM tile streaming",
            "hw": hw,
            "vram_note": "Compute execution",
        })

        if any(term in op_name for term in ["Linear", "Head", "Attn", "Embedding", "Gate", "Up", "Down"]):
            bytes_mb = None
            if "Q_Linear" in op_name:
                vram_name = f"L{current_layer}: VRAM Q_proj (8.4 MB)"
                bytes_mb = 8.39
            elif "K_Linear" in op_name:
                vram_name = f"L{current_layer}: VRAM K_proj (2.1 MB)"
                bytes_mb = 2.10
            elif "V_Linear" in op_name:
                vram_name = f"L{current_layer}: VRAM V_proj (2.1 MB)"
                bytes_mb = 2.10
            elif "QKV" in op_name:
                vram_name = f"L{current_layer}: VRAM QKV Weights (12.6 MB)"
                bytes_mb = 12.58
            elif "O_Linear" in op_name:
                vram_name = f"L{current_layer}: VRAM O_proj (8.4 MB)"
                bytes_mb = 8.39
            elif "Gate_Linear" in op_name:
                vram_name = f"L{current_layer}: VRAM Gate_proj (33.5 MB)"
                bytes_mb = 33.55
            elif "Up_Linear" in op_name:
                vram_name = f"L{current_layer}: VRAM Up_proj (33.5 MB)"
                bytes_mb = 33.55
            elif "Gate_Up" in op_name:
                vram_name = f"L{current_layer}: VRAM Gate & Up (67.1 MB)"
                bytes_mb = 67.11
            elif "Down" in op_name:
                vram_name = f"L{current_layer}: VRAM Down_proj (33.5 MB)"
                bytes_mb = 33.55
            elif "Head" in op_name:
                vram_name = "VRAM: LM_Head Weights (525.3 MB)"
                bytes_mb = 525.34
            elif "Embedding" in op_name:
                vram_name = "VRAM: Embedding Weights (525.3 MB)"
                bytes_desc = "Token embedding table lookup"
            elif "Attn" in op_name:
                vram_name = f"L{current_layer}: VRAM KV Cache States"
                bytes_desc = f"KV cache read/write (Context={seq_len or 'N'})"
            else:
                vram_name = f"L{current_layer}: VRAM {op_name}"
                bytes_desc = "VRAM read/write"

            if bytes_mb is not None:
                bw_gb_s = min(300.0, (bytes_mb / (rel_dur / 1000.0)) / 1024.0) if rel_dur > 0 else 0.0
                bytes_desc = f"{bytes_mb:.1f} MB read @ ~{bw_gb_s:.0f} GB/s (L4 Bus)"

            timeline_events.append({
                "row": "memory-ops",
                "name": vram_name,
                "sub": bytes_desc,
                "cat": "cat-mem-vram",
                "start_ms": round(rel_start, 3),
                "dur_ms": round(rel_dur, 3),
                "domain": "MEMORY (VRAM ➔ On-Chip SRAM)",
                "step": f"Streaming {vram_name}",
                "other": f"Overlaps with {layer_label} compute",
                "hw": "GPU Memory Controller / DRAM Bus",
                "vram_note": bytes_desc,
            })

        if op_name == "FFN_Down_Linear":
            current_layer += 1

    # Track 4: VRAM Footprint Curve ([memory] events)
    mem_events = [
        e for e in events
        if e.get("name") == "[memory]"
        and "ts" in e and "args" in e
        and "Total Allocated" in e["args"]
    ]
    memory_curve = []
    peak_vram_mb = 2450.0
    if mem_events:
        for me in mem_events:
            t = max(0.0, (me["ts"] - base_ts) / 1000.0)
            if t <= total_dur_ms:
                mb = me["args"]["Total Allocated"] / (1024 * 1024)
                memory_curve.append({"t": round(t, 2), "vram_mb": round(mb, 1)})
        memory_curve.sort(key=lambda p: p["t"])
        if memory_curve:
            peak_vram_mb = max(p["vram_mb"] for p in memory_curve)
        if len(memory_curve) > 40:
            step_size = len(memory_curve) // 30
            memory_curve = memory_curve[::step_size]
    else:
        memory_curve = [
            {"t": 0.0, "vram_mb": 2450.0},
            {"t": round(total_dur_ms, 2), "vram_mb": 2450.0},
        ]

    ctx_str = f"SeqLen={seq_len}" if seq_len else "Context"
    step_type_str = f"Prompt Prefill ({ctx_str})" if step_idx == 0 else f"Decode Step #{step_idx} ({ctx_str})"

    return {
        "step": step_idx,
        "token_id": token_id,
        "token_text": token_text,
        "type": step_type_str,
        "duration_ms": round(total_dur_ms, 2),
        "three_metrics": {
            "total_latency_ms": round(total_dur_ms, 2),
            "cpu_idle_ms": round(cpu_idle_ms, 2),
            "cpu_idle_pct": cpu_idle_pct,
            "memory_wait_ms": round(mem_wait_ms, 2),
            "memory_wait_pct": mem_wait_pct,
            "compute_ms": round(pure_compute_ms, 3),
            "compute_pct": compute_pct,
            "duty_cycle_pct": duty_cycle_pct,
        },
        "kpis": {
            "compute_pct": compute_pct,
            "compute_ms": f"{pure_compute_ms:.3f} ms",
            "vram_pct": mem_wait_pct,
            "vram_ms": f"{mem_wait_ms:.2f} ms",
            "vram_gb": f"{peak_vram_mb / 1024.0:.2f} GB",
            "cpu_idle_pct": cpu_idle_pct,
            "cpu_idle_ms": f"{cpu_idle_ms:.2f} ms",
            "duty_cycle_pct": duty_cycle_pct,
        },
        "events": timeline_events,
        "memory_curve": memory_curve,
    }


def render_terminal_dashboard(
    token_records: List[Dict[str, Any]],
    prompt: str = "",
    console: Optional[Console] = None,
):
    """Renders a comprehensive terminal dashboard with tables and summaries."""
    if console is None:
        console = Console()

    if not token_records:
        console.print("[yellow][!] No token metrics found to display.[/yellow]")
        return

    total_tokens = len(token_records)
    sampled_records = [r for r in token_records if r.get("is_sampled", True) and r.get("breakdown")]
    if not sampled_records:
        sampled_records = token_records

    decode_records = [r for r in token_records if r["step"] > 0]
    avg_decode_ms = (
        sum(r["total_latency_ms"] for r in decode_records) / len(decode_records)
        if decode_records
        else (token_records[0]["total_latency_ms"] if token_records else 0.0)
    )
    prefill_ms = token_records[0]["total_latency_ms"] if token_records else 0.0
    throughput = (1000.0 / avg_decode_ms) if avg_decode_ms > 0 else 0.0

    sampled_count = len(sampled_records)
    avg_cpu_idle = sum(r.get("three_metrics", {}).get("cpu_idle_ms", 0.0) for r in sampled_records) / sampled_count if sampled_count > 0 else 0.0
    avg_mem_wait = sum(r.get("three_metrics", {}).get("memory_wait_ms", 0.0) for r in sampled_records) / sampled_count if sampled_count > 0 else 0.0
    avg_compute = sum(r.get("three_metrics", {}).get("compute_ms", 0.0) for r in sampled_records) / sampled_count if sampled_count > 0 else 0.0
    avg_duty_cycle = sum(r.get("three_metrics", {}).get("duty_cycle_pct", 0.0) for r in sampled_records) / sampled_count if sampled_count > 0 else 0.0

    header_text = Text()
    if prompt:
        header_text.append(f"Prompt: {prompt}\n", style="italic white")
    header_text.append(f"Sequence Summary: {total_tokens} tokens total | Prefill: {prefill_ms:.2f} ms | Avg Decode: {avg_decode_ms:.2f} ms/token ({throughput:.1f} tok/s)\n", style="bold green")
    header_text.append(f"Sampled Deep Checkpoints: {sampled_count} steps (step 0, every 100th, and last step)\n\n", style="dim white")
    header_text.append("THE THREE PHYSICAL METRICS OF INFERENCE (Sampled Average):\n", style="bold underline yellow")
    header_text.append(f"  1. CPU Launch & Driver Gaps : {avg_cpu_idle:6.2f} ms ({avg_cpu_idle/avg_decode_ms*100:5.1f}%) [Host Starvation / Empty GPU Queue]\n", style="bold blue")
    header_text.append(f"  2. VRAM Data Wait           : {avg_mem_wait:6.2f} ms ({avg_mem_wait/avg_decode_ms*100:5.1f}%) [Memory Bandwidth Saturated / Weight Streaming]\n", style="bold cyan")
    header_text.append(f"  3. Pure Math Compute        : {avg_compute:6.3f} ms ({avg_compute/avg_decode_ms*100:5.2f}%) [Active Tensor Cores & Vector ALUs]\n\n", style="bold red")
    header_text.append(f"Active GPU Duty Cycle: {avg_duty_cycle:.1f}% of total wall-clock time\n", style="bold bright_white")

    console.print(Panel(header_text, title="[bold cyan]LLaMA-3.2 Inference: Latency & Hardware Dashboard[/bold cyan]", expand=False))

    # Table 1: Per-Operation Breakdown across Sampled Checkpoints
    op_table = Table(
        title="[bold yellow]Sampled Checkpoints: Fine-Grained Per-Operation Breakdown (ms)[/bold yellow]",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    op_table.add_column("Step", justify="right", style="cyan", width=6)
    op_table.add_column("Token", justify="left", style="white", width=10)
    op_table.add_column("Total", justify="right", style="bold white", width=8)
    op_table.add_column("FFN Gate/Up", justify="right", style="bold orange3", width=11)
    op_table.add_column("LM Head", justify="right", style="bold magenta", width=9)
    op_table.add_column("FFN Down", justify="right", style="bold yellow", width=10)
    op_table.add_column("QKV Proj", justify="right", style="bold red", width=10)
    op_table.add_column("O Proj", justify="right", style="dark_red", width=8)
    op_table.add_column("Attn Compute", justify="right", style="bold bright_red", width=12)
    op_table.add_column("RoPE", justify="right", style="red", width=8)
    op_table.add_column("RMSNorms", justify="right", style="green", width=9)
    op_table.add_column("Sampling", justify="right", style="blue", width=9)

    for r in sampled_records:
        bd = r.get("breakdown", {}) or {}
        tot = r.get("three_metrics", {}).get("total_latency_ms", r["total_latency_ms"])
        tok_str = repr(r.get("token_text", ""))[:8]
        rms_norm_total = bd.get("RMSNorm_Attn", 0.0) + bd.get("RMSNorm_FFN", 0.0) + bd.get("RMSNorm_Final", 0.0)

        op_table.add_row(
            f"#{r['step']}",
            tok_str,
            f"{tot:.2f}",
            f"{bd.get('FFN_Gate_Up_Linear', 0.0):.2f}",
            f"{bd.get('LM_Head', 0.0):.2f}",
            f"{bd.get('FFN_Down_Linear', 0.0):.2f}",
            f"{bd.get('QKV_Linear', 0.0):.2f}",
            f"{bd.get('O_Linear', 0.0):.2f}",
            f"{bd.get('Attn_Compute', 0.0):.3f}",
            f"{bd.get('RoPE', 0.0):.3f}",
            f"{rms_norm_total:.3f}",
            f"{bd.get('Sampling', 0.0):.3f}",
        )
    console.print(op_table)

    # Table 2: The Three Physical Metrics Decomposition
    phys_table = Table(
        title="[bold green]Sampled Checkpoints: The Three Physical Metrics Decomposition[/bold green]",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    phys_table.add_column("Step", justify="right", style="cyan", width=6)
    phys_table.add_column("Token", justify="left", style="white", width=12)
    phys_table.add_column("Total (ms)", justify="right", style="bold white", width=10)
    phys_table.add_column("CPU Gaps (ms)", justify="right", style="bold blue", width=13)
    phys_table.add_column("CPU %", justify="right", style="blue", width=7)
    phys_table.add_column("VRAM Wait (ms)", justify="right", style="bold cyan", width=14)
    phys_table.add_column("VRAM %", justify="right", style="cyan", width=7)
    phys_table.add_column("Compute (ms)", justify="right", style="bold red", width=12)
    phys_table.add_column("Compute %", justify="right", style="red", width=9)
    phys_table.add_column("Duty Cycle", justify="right", style="green", width=10)

    for r in sampled_records:
        m = r.get("three_metrics", {}) or {}
        tot = m.get("total_latency_ms", r["total_latency_ms"])
        cpu_ms = m.get("cpu_idle_ms", 0.0)
        cpu_pct = m.get("cpu_idle_pct", 0.0)
        vram_ms = m.get("memory_wait_ms", 0.0)
        vram_pct = m.get("memory_wait_pct", 0.0)
        comp_ms = m.get("compute_ms", 0.0)
        comp_pct = m.get("compute_pct", 0.0)
        duty = m.get("duty_cycle_pct", 0.0)
        tok_str = repr(r.get("token_text", ""))[:10]

        phys_table.add_row(
            f"#{r['step']}",
            tok_str,
            f"{tot:.2f}",
            f"{cpu_ms:.2f}",
            f"{cpu_pct:.1f}%",
            f"{vram_ms:.2f}",
            f"{vram_pct:.1f}%",
            f"{comp_ms:.3f}",
            f"{comp_pct:.2f}%",
            f"{duty:.1f}%",
        )

    console.print(phys_table)


def generate_html_dashboard(
    token_records: List[Dict[str, Any]],
    prompt: str = "",
    output_file: str = "profile_dashboard.html",
    timeline_records: Optional[Dict[int, Any]] = None,
):
    """
    Generates a zero-dependency, self-contained interactive HTML/SVG dashboard with:
    1. Overall Token Decode Latency (every token overall)
    2. Time Taken per Operation at Selected Time Step (select time step)
    3. Operation Latency Scaling Across Time Steps (choose operation)
    4. The Three Physical Metrics per Step (Stacked Latency Decomposition)
    5. Microsecond Gantt Timeline (Separated Compute & Memory Tracks)
    """
    if not token_records:
        return

    # Build timeline_records from token_records if not provided explicitly
    if timeline_records is None:
        timeline_records = {}
        for r in token_records:
            if "timeline" in r and r["timeline"]:
                timeline_records[r["step"]] = r["timeline"]

    total_tokens = len(token_records)
    sampled_records = [r for r in token_records if r.get("is_sampled", True) and r.get("breakdown")]
    if not sampled_records:
        sampled_records = token_records

    decode_records = [r for r in token_records if r["step"] > 0]
    avg_decode_latency = (
        sum(r["total_latency_ms"] for r in decode_records) / len(decode_records)
        if decode_records
        else (token_records[0]["total_latency_ms"] if token_records else 0.0)
    )
    prefill_latency = token_records[0]["total_latency_ms"] if token_records else 0.0
    tokens_per_sec = (1000.0 / avg_decode_latency) if avg_decode_latency > 0 else 0.0

    sampled_count = len(sampled_records)
    avg_cpu_idle_ms = sum(r.get("three_metrics", {}).get("cpu_idle_ms", 0.0) for r in sampled_records) / sampled_count if sampled_count > 0 else 0.0
    avg_cpu_idle_pct = (avg_cpu_idle_ms / avg_decode_latency * 100.0) if avg_decode_latency > 0 else 0.0

    avg_mem_wait_ms = sum(r.get("three_metrics", {}).get("memory_wait_ms", 0.0) for r in sampled_records) / sampled_count if sampled_count > 0 else 0.0
    avg_mem_wait_pct = (avg_mem_wait_ms / avg_decode_latency * 100.0) if avg_decode_latency > 0 else 0.0

    avg_compute_ms = sum(r.get("three_metrics", {}).get("compute_ms", 0.0) for r in sampled_records) / sampled_count if sampled_count > 0 else 0.0
    avg_compute_pct = (avg_compute_ms / avg_decode_latency * 100.0) if avg_decode_latency > 0 else 0.0

    avg_duty_cycle = sum(r.get("three_metrics", {}).get("duty_cycle_pct", 0.0) for r in sampled_records) / sampled_count if sampled_count > 0 else 0.0

    profile_data_json = json.dumps({
        "prompt": prompt,
        "tokens": token_records,
        "categories": FINE_GRAINED_CATEGORIES,
        "colors": CATEGORY_COLORS,
        "metadata": OPERATION_METADATA,
        "summary": {
            "total_tokens": total_tokens,
            "sampled_count": sampled_count,
            "prefill_ms": round(prefill_latency, 2),
            "avg_decode_ms": round(avg_decode_latency, 2),
            "throughput_tps": round(tokens_per_sec, 1),
            "avg_cpu_idle_ms": round(avg_cpu_idle_ms, 2),
            "avg_cpu_idle_pct": round(avg_cpu_idle_pct, 1),
            "avg_mem_wait_ms": round(avg_mem_wait_ms, 2),
            "avg_mem_wait_pct": round(avg_mem_wait_pct, 1),
            "avg_compute_ms": round(avg_compute_ms, 3),
            "avg_compute_pct": round(avg_compute_pct, 2),
            "avg_duty_cycle": round(avg_duty_cycle, 1),
        }
    }, indent=2)

    timeline_data_json = json.dumps(timeline_records, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>tiny_vllm - Inference Latency & Operations Profiler</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --card-border: #1f2937;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-blue: #38bdf8;
            --accent-cyan: #06b6d4;
            --accent-purple: #a855f7;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --track-bg: #131b2c;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            padding: 2rem;
            line-height: 1.5;
        }}
        .container {{ max-width: 1440px; margin: 0 auto; }}
        
        header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.25rem;
        }}
        .header-title h1 {{
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--accent-blue);
            letter-spacing: -0.02em;
        }}
        .header-title .subtitle {{
            color: var(--text-muted);
            font-size: 0.92rem;
            margin-top: 0.35rem;
        }}
        .badge-live {{
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* Quick Navigation Bar */
        .nav-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 1.75rem;
            padding: 0.75rem 1rem;
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid var(--card-border);
            border-radius: 0.6rem;
        }}
        .nav-link {{
            color: var(--text-muted);
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.35rem 0.75rem;
            border-radius: 0.375rem;
            background: #1f2937;
            transition: all 0.15s ease;
        }}
        .nav-link:hover {{
            color: #fff;
            background: #374151;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.75rem;
            padding: 1rem 1.25rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .kpi-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}
        .kpi-value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: #fff;
            font-family: monospace;
        }}
        .kpi-sub {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}

        .section {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .section-desc {{
            color: var(--text-muted);
            font-size: 0.88rem;
            margin: 0.35rem 0 1.25rem 0;
        }}

        /* Selectors & Controls */
        .selector-bar {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1.25rem;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--card-border);
            border-radius: 0.6rem;
        }}
        .step-select-btn {{
            background: #1e293b;
            color: #e2e8f0;
            border: 1px solid #334155;
            padding: 0.4rem 0.85rem;
            border-radius: 0.45rem;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .step-select-btn:hover {{ background: #334155; border-color: #64748b; }}
        .step-select-btn.active {{
            background: #2563eb;
            border-color: #3b82f6;
            color: #ffffff;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
        }}
        .op-select-dropdown {{
            background: #1e293b;
            color: #f8fafc;
            border: 1px solid #3b82f6;
            padding: 0.45rem 0.9rem;
            border-radius: 0.45rem;
            font-size: 0.88rem;
            font-weight: 600;
            cursor: pointer;
            outline: none;
        }}
        .op-pill-btn {{
            background: #0f172a;
            color: #94a3b8;
            border: 1px solid #1e293b;
            padding: 0.35rem 0.7rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .op-pill-btn:hover {{ color: #f1f5f9; border-color: #475569; }}
        .op-pill-btn.active {{
            background: rgba(56, 189, 248, 0.2);
            border-color: #38bdf8;
            color: #38bdf8;
            font-weight: 600;
        }}

        /* Horizontal Bar styling */
        .h-bar-row {{
            display: flex;
            align-items: center;
            margin-bottom: 0.55rem;
            font-size: 0.82rem;
            padding: 0.2rem 0;
        }}
        .h-bar-label {{
            width: 250px;
            min-width: 250px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }}
        .h-bar-track {{
            flex-grow: 1;
            height: 22px;
            background: rgba(15, 23, 42, 0.6);
            border-radius: 4px;
            overflow: hidden;
            position: relative;
            margin: 0 1rem;
        }}
        .h-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        .h-bar-val {{
            width: 140px;
            min-width: 140px;
            text-align: right;
            font-family: monospace;
            font-weight: 700;
        }}
        .tag-badge {{
            font-size: 0.65rem;
            padding: 0.15rem 0.45rem;
            border-radius: 3px;
            text-transform: uppercase;
            font-weight: 700;
        }}

        /* Gantt Timeline styling */
        .controls-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1.25rem;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--card-border);
            border-radius: 0.6rem;
        }}
        .legend-items {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 1.1rem;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 0.78rem;
            color: var(--text-main);
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            margin-right: 0.45rem;
            flex-shrink: 0;
        }}
        .zoom-controls {{ display: flex; gap: 0.4rem; }}
        .btn-sm {{
            background: #1f2937;
            color: var(--text-main);
            border: 1px solid #374151;
            padding: 0.3rem 0.65rem;
            border-radius: 0.375rem;
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
        }}
        .btn-sm:hover {{ background: #374151; color: #fff; }}

        .gantt-wrapper {{
            position: relative;
            background: #070a10;
            border: 1px solid var(--card-border);
            border-radius: 0.6rem;
            overflow: hidden;
        }}
        .timeline-header {{
            display: flex;
            border-bottom: 1px solid var(--card-border);
            background: #05070c;
        }}
        .track-labels-header {{
            width: 280px;
            min-width: 280px;
            padding: 0.6rem 1rem;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--text-muted);
            border-right: 1px solid var(--card-border);
            background: #040508;
        }}
        .time-scale-container {{
            flex-grow: 1;
            height: 28px;
            position: relative;
            overflow: hidden;
        }}
        .gantt-body {{ display: flex; }}
        .track-labels-col {{
            width: 280px;
            min-width: 280px;
            border-right: 1px solid var(--card-border);
            background: #040508;
        }}
        .track-label {{
            height: 80px;
            padding: 0.6rem 1rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .track-label:last-child {{ border-bottom: none; height: 72px; }}
        .track-label-title {{
            font-size: 0.85rem;
            font-weight: 600;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }}
        .track-label-desc {{
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
            line-height: 1.25;
        }}
        .track-type-badge {{
            font-size: 0.65rem;
            font-weight: 700;
            padding: 0.12rem 0.4rem;
            border-radius: 3px;
            text-transform: uppercase;
        }}
        .badge-compute {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
        .badge-memory {{ background: rgba(6, 182, 212, 0.2); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.4); }}
        .badge-vram {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}

        .tracks-canvas-col {{
            flex-grow: 1;
            position: relative;
            overflow-x: auto;
            background: #080c15;
        }}
        .track-row {{
            position: relative;
            height: 80px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .track-row:last-child {{ border-bottom: none; height: 72px; }}

        .gantt-block {{
            position: absolute;
            top: 14px;
            height: 52px;
            border-radius: 4px;
            cursor: pointer;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 0 0.35rem;
            font-size: 0.7rem;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.15);
            transition: transform 0.1s ease, filter 0.1s ease;
        }}
        .gantt-block:hover {{
            filter: brightness(1.25);
            transform: translateY(-2px);
            z-index: 10;
        }}
        .active-selection {{
            outline: 2px solid #38bdf8;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.6);
            z-index: 20;
        }}

        .cat-cpu-dispatch {{ background: #1d4ed8; color: #fff; }}
        .cat-cpu-stall {{ background: #475569; color: #cbd5e1; border-color: #64748b; }}
        .cat-pcie-htod {{ background: #d97706; color: #fff; }}
        .cat-pcie-dtoh {{ background: #b45309; color: #fff; }}
        .cat-vram-weight {{ background: #0891b2; color: #fff; }}
        .cat-vram-kv {{ background: #0284c7; color: #fff; }}
        .cat-tensor-gemm {{ background: #dc2626; color: #fff; }}
        .cat-vector-math {{ background: #ea580c; color: #fff; }}

        .block-title {{
            font-weight: 700;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .block-sub {{
            font-size: 0.65rem;
            opacity: 0.85;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        /* Tooltip & Inspector */
        .tooltip {{
            position: fixed;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid var(--accent-blue);
            color: #fff;
            padding: 0.65rem 0.9rem;
            border-radius: 0.5rem;
            font-size: 0.78rem;
            pointer-events: none;
            z-index: 100;
            display: none;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.7);
            max-width: 320px;
        }}
        .tooltip-title {{ font-weight: 700; color: var(--accent-blue); margin-bottom: 0.25rem; }}
        .tooltip-row {{ display: flex; justify-content: space-between; gap: 0.5rem; margin-top: 0.15rem; }}
        .tooltip-label {{ color: var(--text-muted); }}

        .inspector-card {{
            margin-top: 1.25rem;
            background: #0f172a;
            border: 1px solid var(--card-border);
            border-radius: 0.5rem;
            padding: 1rem 1.25rem;
        }}
        .inspector-title {{
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .inspector-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        .inspector-item-label {{ font-size: 0.72rem; color: var(--text-muted); }}
        .inspector-item-val {{
            font-size: 0.88rem;
            font-weight: 600;
            margin-top: 0.15rem;
            color: #fff;
            font-family: monospace;
        }}

        /* SVG & Tables */
        svg {{ width: 100%; height: auto; overflow: visible; }}
        .chart-svg text {{ font-family: monospace; font-size: 11px; fill: var(--text-muted); }}
        .grid-line {{ stroke: var(--card-border); stroke-dasharray: 4; stroke-width: 0.8; }}
        .bar-segment {{ transition: opacity 0.15s ease; cursor: pointer; }}
        .bar-segment:hover {{ opacity: 0.85; }}

        table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; text-align: left; }}
        th, td {{ padding: 0.6rem 0.75rem; border-bottom: 1px solid var(--card-border); }}
        th {{ background: rgba(15, 23, 42, 0.8); color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.7rem; }}
        tr:hover {{ background: rgba(56, 189, 248, 0.05); }}
    </style>
</head>
<body>
    <div id="tooltip" class="tooltip"></div>
    <div class="container">
        <header>
            <div class="header-title">
                <h1>⚡ tiny_vllm - Inference Latency & Operations Profiler</h1>
                <div class="subtitle">Full-Sequence Token Timing, Per-Operation Breakdowns & Physical Hardware Decomposition</div>
            </div>
            <div class="badge-live">Hardware Profiler Active</div>
        </header>

        <!-- QUICK NAVIGATION -->
        <nav class="nav-bar">
            <a href="#section-overall-latency" class="nav-link">📈 1. Overall Decode Latency (Every Token)</a>
            <a href="#section-step-operations" class="nav-link">🔍 2. Per-Operation Breakdown (Select Step)</a>
            <a href="#section-operation-scaling" class="nav-link">📊 3. Operation Latency Scaling (Choose Op)</a>
            <a href="#section-three-metrics" class="nav-link">⚡ 4. The Three Physical Metrics</a>
            <a href="#gantt-section" class="nav-link">⏱️ 5. Microsecond Gantt Timeline</a>
        </nav>

        <!-- HERO KPI GRID -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Generated Tokens</div>
                <div class="kpi-value">{total_tokens} <span style="font-size:1rem;color:#94a3b8;">tokens</span></div>
                <div class="kpi-sub">{sampled_count} sampled checkpoints</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #38bdf8;">
                <div class="kpi-label" style="color:#38bdf8;">Avg Decode Latency</div>
                <div class="kpi-value" style="color:#38bdf8;">{avg_decode_latency:.2f} <span style="font-size:1rem;color:#94a3b8;">ms</span></div>
                <div class="kpi-sub">{tokens_per_sec:.1f} tokens/sec throughput</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Prefill Latency (Step 0)</div>
                <div class="kpi-value">{prefill_latency:.2f} <span style="font-size:1rem;color:#94a3b8;">ms</span></div>
                <div class="kpi-sub">Prompt processing</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #3b82f6;">
                <div class="kpi-label" style="color: #60a5fa;">1. CPU Launch Gaps</div>
                <div class="kpi-value" style="color: #60a5fa;">{avg_cpu_idle_ms:.2f} <span style="font-size:1rem;color:#94a3b8;">ms</span></div>
                <div class="kpi-sub">{avg_cpu_idle_pct:.1f}% of decode time (Host overhead)</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #06b6d4;">
                <div class="kpi-label" style="color: #22d3ee;">2. VRAM Data Wait</div>
                <div class="kpi-value" style="color: #22d3ee;">{avg_mem_wait_ms:.2f} <span style="font-size:1rem;color:#94a3b8;">ms</span></div>
                <div class="kpi-sub">{avg_mem_wait_pct:.1f}% of decode time (Bandwidth bound)</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #10b981;">
                <div class="kpi-label" style="color: #34d399;">Active GPU Duty Cycle</div>
                <div class="kpi-value" style="color: #34d399;">{avg_duty_cycle:.1f}%</div>
                <div class="kpi-sub">Active kernel execution fraction</div>
            </div>
        </div>

        <!-- 1. OVERALL DECODE LATENCY FOR EVERY TOKEN -->
        <div class="section" id="section-overall-latency">
            <div class="section-title">📈 1. Overall Token-by-Token Decode Latency (Every Token Overall)</div>
            <div class="section-desc">Measured wall-clock decode latency for all {total_tokens} generated tokens. Distinct gold diamond markers indicate sampled checkpoints with deep kernel profiling (click any to view its operations).</div>
            <div id="overall-latency-container"></div>
        </div>

        <!-- 2. TIME TAKEN PER OPERATION AT SELECTED TIME STEP -->
        <div class="section" id="section-step-operations">
            <div class="section-title">🔍 2. Time Taken per Operation at Selected Time Step</div>
            <div class="section-desc">Select any sampled time step below to view the execution time and percentage breakdown for each individual model operation.</div>
            
            <div class="selector-bar" id="step-selector-container">
                <span style="font-size:0.85rem; font-weight:700; color:#94a3b8; margin-right:0.5rem;">Select Time Step:</span>
                <div id="step-buttons" style="display:flex; flex-wrap:wrap; gap:0.4rem;"></div>
            </div>

            <div class="kpi-grid" id="step-info-banner" style="margin-bottom: 1.25rem;"></div>

            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 1rem;">
                <div style="font-weight:700; font-size:0.95rem; color:#fff;">Operation Latency Distribution</div>
                <div style="display:flex; gap:0.4rem;">
                    <button class="btn-sm active" id="sort-dur-btn" onclick="toggleOpSort('duration')">Sort: Duration (High ➔ Low)</button>
                    <button class="btn-sm" id="sort-arch-btn" onclick="toggleOpSort('arch')">Sort: Architectural Order</button>
                </div>
            </div>

            <div id="step-ops-bars" style="margin-bottom: 1.75rem;"></div>

            <div style="overflow-x:auto;">
                <table id="step-ops-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Operation</th>
                            <th>Category</th>
                            <th>Time (ms)</th>
                            <th>% Share of Device Time</th>
                            <th>Architectural Role & Scaling Nature</th>
                        </tr>
                    </thead>
                    <tbody id="step-ops-table-body"></tbody>
                </table>
            </div>
        </div>

        <!-- 3. OPERATION LATENCY SCALING ACROSS DIFFERENT TIME STEPS -->
        <div class="section" id="section-operation-scaling">
            <div class="section-title">📊 3. Operation Latency Across Different Time Steps</div>
            <div class="section-desc">Choose any operation from the selector below to analyze how its execution time behaves across different time steps as sequence length grows.</div>
            
            <div class="selector-bar" style="gap: 1rem;">
                <div style="display:flex; align-items:center; gap:0.6rem;">
                    <label for="op-selector" style="font-size:0.85rem; font-weight:700; color:#94a3b8;">Choose Operation:</label>
                    <select id="op-selector" class="op-select-dropdown" onchange="selectOperation(this.value)"></select>
                </div>
                <div id="op-quick-pills" style="display:flex; flex-wrap:wrap; gap:0.4rem;"></div>
            </div>

            <div class="inspector-card" id="op-insight-card" style="margin-bottom: 1.25rem;"></div>

            <div id="op-scaling-chart-container"></div>
        </div>

        <!-- 4. THE THREE PHYSICAL METRICS STACKED BAR CHART -->
        <div class="section" id="section-three-metrics">
            <div class="section-title">⚡ 4. The Three Physical Metrics per Step (Stacked Latency Decomposition)</div>
            <div class="section-desc">Decomposes 100% of wall-clock token generation time into Pure Math Compute (Red), VRAM Data Wait (Cyan), and CPU Launch & Driver Gaps (Blue).</div>
            <div style="display:flex;flex-wrap:wrap;gap:0.75rem 1.25rem;margin:1rem 0;padding:0.75rem 1rem;background:rgba(15,23,42,0.6);border-radius:0.5rem;" id="op-legend"></div>
            <div id="stacked-bar-container"></div>
        </div>

        <!-- 5. GANTT TIMELINE SECTION -->
        <div class="section" id="gantt-section">
            <div class="section-header">
                <div>
                    <div class="section-title">⏱️ 5. Microsecond Gantt Timeline (Compute & Memory Separated)</div>
                    <div class="section-desc">Track 1: Host CPU Dispatch & Sync Stall. Track 2: VRAM & PCIe Data Movement. Track 3: Tensor Core & Vector ALU Compute.</div>
                </div>
                <div class="step-tabs" id="step-tabs" style="display:flex; gap:0.4rem; margin-top:0.5rem; margin-bottom:1rem;"></div>
            </div>

            <div class="controls-bar">
                <div class="legend-items">
                    <div class="legend-item"><span class="legend-color" style="background:#2563eb;"></span><span>CPU Dispatch</span></div>
                    <div class="legend-item"><span class="legend-color" style="background:#64748b;"></span><span>CPU Sync Stall (.item)</span></div>
                    <div class="legend-item"><span class="legend-color" style="background:#f59e0b;"></span><span>PCIe Memory (HtoD / DtoH)</span></div>
                    <div class="legend-item"><span class="legend-color" style="background:#06b6d4;"></span><span>VRAM Memory (Weight/KV Streaming)</span></div>
                    <div class="legend-item"><span class="legend-color" style="background:#ef4444;"></span><span>Tensor Core Math (Compute)</span></div>
                    <div class="legend-item"><span class="legend-color" style="background:#ea580c;"></span><span>Vector ALU Math (RoPE / Softmax)</span></div>
                </div>
                <div class="zoom-controls">
                    <button class="btn-sm" onclick="setZoom(1)">1x</button>
                    <button class="btn-sm" onclick="setZoom(2)">2x</button>
                    <button class="btn-sm" onclick="setZoom(4)">4x</button>
                    <button class="btn-sm" onclick="setZoom(8)">8x</button>
                    <button class="btn-sm" onclick="zoom(1.25)">🔍 (+)</button>
                    <button class="btn-sm" onclick="zoom(0.8)">🔍 (-)</button>
                    <button class="btn-sm" onclick="resetZoom()">Reset</button>
                </div>
            </div>

            <div class="gantt-wrapper">
                <div class="timeline-header">
                    <div class="track-labels-header">Hardware Stream & Domain</div>
                    <div class="time-scale-container" id="time-scale"></div>
                </div>

                <div class="gantt-body">
                    <div class="track-labels-col">
                        <div class="track-label">
                            <div class="track-label-title"><span style="color:#38bdf8;">●</span> CPU Main Thread <span class="track-type-badge badge-compute">Host</span></div>
                            <div class="track-label-desc">Op enqueue ➔ .item() stall</div>
                        </div>
                        <div class="track-label">
                            <div class="track-label-title"><span style="color:#06b6d4;">⚡</span> MEMORY TRACK <span class="track-type-badge badge-memory">Memory</span></div>
                            <div class="track-label-desc">PCIe transfers + VRAM weight/KV streaming</div>
                        </div>
                        <div class="track-label">
                            <div class="track-label-title"><span style="color:#ef4444;">🔥</span> COMPUTE TRACK <span class="track-type-badge badge-compute">Compute</span></div>
                            <div class="track-label-desc">Tensor Cores & Vector ALUs (Q ➔ K ➔ V)</div>
                        </div>
                        <div class="track-label">
                            <div class="track-label-title"><span style="color:#10b981;">📊</span> Active VRAM Footprint <span class="track-type-badge badge-vram">VRAM</span></div>
                            <div class="track-label-desc">Resident weights & activation memory</div>
                        </div>
                    </div>

                    <div class="tracks-canvas-col" id="tracks-canvas">
                        <div class="track-row" id="row-cpu"></div>
                        <div class="track-row" id="row-memory-ops"></div>
                        <div class="track-row" id="row-compute-ops"></div>
                        <div class="track-row" id="row-vram-curve"></div>
                    </div>
                </div>
            </div>

            <div class="inspector-card">
                <div class="inspector-title"><span>🔍 Operation Inspector (Click any block in either Memory or Compute track)</span></div>
                <div class="inspector-grid">
                    <div class="inspector-item">
                        <div class="inspector-item-label">Operation / Step Name</div>
                        <div class="inspector-item-val" id="insp-name">Click any block to inspect</div>
                    </div>
                    <div class="inspector-item">
                        <div class="inspector-item-label">Domain Classification</div>
                        <div class="inspector-item-val" id="insp-domain">-</div>
                    </div>
                    <div class="inspector-item">
                        <div class="inspector-item-label">Timeline Window (Start ➔ Dur)</div>
                        <div class="inspector-item-val" id="insp-time">-</div>
                    </div>
                    <div class="inspector-item">
                        <div class="inspector-item-label">Sequential Step Context</div>
                        <div class="inspector-item-val" id="insp-step">-</div>
                    </div>
                    <div class="inspector-item">
                        <div class="inspector-item-label">Parallel Activity on Other Track</div>
                        <div class="inspector-item-val" id="insp-other">-</div>
                    </div>
                    <div class="inspector-item">
                        <div class="inspector-item-label">Hardware Device</div>
                        <div class="inspector-item-val" id="insp-hw">-</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const profileData = {profile_data_json};
        const timelineData = {timeline_data_json};

        let sampledTokens = profileData.tokens.filter(t => t.is_sampled && t.breakdown);
        if (sampledTokens.length === 0) sampledTokens = profileData.tokens;
        let availableSteps = sampledTokens.map(t => t.step);
        let currentStep = availableSteps.length > 0 ? availableSteps[0] : 0;
        let currentStepForOps = currentStep;
        let currentSelectedOp = 'Attn_Compute';
        let opSortMode = 'duration';
        let zoomScale = 1.0;

        const tooltip = document.getElementById("tooltip");
        function showTooltip(e, ev) {{
            tooltip.innerHTML = `
                <div class="tooltip-title">${{ev.name || ''}}</div>
                <div class="tooltip-row"><span class="tooltip-label">Domain:</span><span style="font-weight:700;color:#fff;">${{ev.domain || ''}}</span></div>
                <div class="tooltip-row"><span class="tooltip-label">Context:</span><span>${{ev.step || ''}}</span></div>
                ${{ev.start_ms !== undefined ? `<div class="tooltip-row"><span class="tooltip-label">Window:</span><span>${{ev.start_ms.toFixed(2)}} ms ➔ ${{((ev.start_ms + (ev.dur_ms || 0))).toFixed(2)}} ms</span></div>` : ''}}
                <div class="tooltip-row"><span class="tooltip-label">Duration:</span><span style="color:#38bdf8;font-weight:700;">${{((ev.dur_ms || 0) * 1000).toFixed(0)}} μs (${{(ev.dur_ms || 0).toFixed(2)}} ms)</span></div>
                ${{ev.other ? `<div class="tooltip-row"><span class="tooltip-label">Info:</span><span style="color:#10b981;">${{ev.other}}</span></div>` : ''}}
            `;
            tooltip.style.display = "block";
            tooltip.style.left = (e.clientX + 15) + "px";
            tooltip.style.top = (e.clientY + 15) + "px";
        }}
        function hideTooltip() {{ tooltip.style.display = "none"; }}

        // 1. Render Overall Latency Chart (Every Token Overall)
        function renderOverallLatencyChart() {{
            const container = document.getElementById("overall-latency-container");
            const data = profileData.tokens;
            if (!data || data.length === 0) return;

            const w = 1200, h = 280, padL = 70, padR = 40, padT = 30, padB = 45;
            const chartW = w - padL - padR;
            const chartH = h - padT - padB;

            const maxVal = Math.max(...data.map(d => (d.three_metrics ? d.three_metrics.total_latency_ms : d.total_latency_ms) || 1)) * 1.15 || 1;
            const stepW = chartW / (data.length > 1 ? (data.length - 1) : 1);

            let points = [];
            data.forEach((d, idx) => {{
                const lat = d.three_metrics ? d.three_metrics.total_latency_ms : d.total_latency_ms;
                const x = padL + idx * stepW;
                const y = padT + chartH - (lat / maxVal) * chartH;
                points.push(`${{x}},${{y}}`);
            }});

            let svg = `<svg viewBox="0 0 ${{w}} ${{h}}" class="chart-svg">`;
            for (let i = 0; i <= 4; i++) {{
                const yVal = (maxVal / 4) * i;
                const yPos = padT + chartH - (chartH / 4) * i;
                svg += `<line x1="${{padL}}" y1="${{yPos}}" x2="${{w - padR}}" y2="${{yPos}}" class="grid-line" />`;
                svg += `<text x="${{padL - 10}}" y="${{yPos + 4}}" text-anchor="end">${{yVal.toFixed(1)}} ms</text>`;
            }}

            svg += `<polyline points="${{points.join(' ')}}" fill="none" stroke="#38bdf8" stroke-width="2" />`;

            data.forEach((d, idx) => {{
                const lat = d.three_metrics ? d.three_metrics.total_latency_ms : d.total_latency_ms;
                const x = padL + idx * stepW;
                const y = padT + chartH - (lat / maxVal) * chartH;
                const isSampled = d.is_sampled && d.breakdown;

                if (isSampled) {{
                    const size = 7;
                    const pts = `${{x}},${{y - size}} ${{x + size}},${{y}} ${{x}},${{y + size}} ${{x - size}},${{y}}`;
                    svg += `<polygon points="${{pts}}" fill="#f59e0b" stroke="#ffffff" stroke-width="2"
                        style="cursor: pointer;"
                        onclick="selectTimeStep(${{d.step}}); document.getElementById('section-step-operations').scrollIntoView({{behavior: 'smooth'}});"
                        onmousemove="showTooltip(event, {{name: 'Sampled Checkpoint: Step #${{d.step}}', domain: 'Click to inspect operations breakdown', step: 'Token: ${{d.token_text ? d.token_text.replace(/'/g, '') : ''}}', start_ms: 0, dur_ms: ${{lat}}, other: 'Deep Profiler Active'}})"
                        onmouseleave="hideTooltip()" />`;
                    svg += `<text x="${{x}}" y="${{y - 12}}" text-anchor="middle" fill="#f59e0b" font-weight="700" font-size="11">#${{d.step}}</text>`;
                }} else {{
                    svg += `<circle cx="${{x}}" cy="${{y}}" r="2.5" fill="#38bdf8"
                        onmousemove="showTooltip(event, {{name: 'Token #${{d.step}}', domain: 'Native Decode (Unprofiled)', step: 'Token: ${{d.token_text ? d.token_text.replace(/'/g, '') : ''}}', start_ms: 0, dur_ms: ${{lat}}, other: 'Latency: ${{lat.toFixed(2)}} ms'}})"
                        onmouseleave="hideTooltip()" />`;
                }}

                if (data.length <= 25 || idx % Math.ceil(data.length / 15) === 0 || idx === data.length - 1) {{
                    svg += `<text x="${{x}}" y="${{padT + chartH + 20}}" text-anchor="middle">#${{d.step}}</text>`;
                }}
            }});

            svg += `</svg>`;
            container.innerHTML = svg;
        }}

        // 2. Render Step Operations (Select Time Step)
        function initStepSelector() {{
            const container = document.getElementById("step-buttons");
            container.innerHTML = "";
            sampledTokens.forEach(t => {{
                const btn = document.createElement("button");
                btn.className = `step-select-btn ${{t.step === currentStepForOps ? 'active' : ''}}`;
                btn.dataset.step = t.step;
                btn.innerText = t.step === 0 ? "Step #0 (Prefill)" : `Step #${{t.step}}`;
                btn.onclick = () => selectTimeStep(t.step);
                container.appendChild(btn);
            }});
        }}

        function selectTimeStep(step) {{
            currentStepForOps = step;
            currentStep = step;
            document.querySelectorAll(".step-select-btn").forEach(btn => {{
                btn.classList.toggle("active", parseInt(btn.dataset.step) === step);
            }});
            document.querySelectorAll(".step-btn").forEach(btn => {{
                btn.classList.toggle("active", parseInt(btn.dataset.step) === step);
            }});
            renderStepOperations();
            renderGantt();
        }}

        function toggleOpSort(mode) {{
            opSortMode = mode;
            document.getElementById("sort-dur-btn").classList.toggle("active", mode === "duration");
            document.getElementById("sort-arch-btn").classList.toggle("active", mode === "arch");
            renderStepOperations();
        }}

        function renderStepOperations() {{
            let target = sampledTokens.find(t => t.step === currentStepForOps);
            if (!target && sampledTokens.length > 0) {{
                target = sampledTokens[0];
                currentStepForOps = target.step;
            }}
            if (!target) return;

            const isPrefill = target.step === 0;
            const banner = document.getElementById("step-info-banner");
            const bd = target.breakdown || {{}};
            const totalStepMs = target.three_metrics ? target.three_metrics.total_latency_ms : target.total_latency_ms;
            const deviceTimeMs = Object.values(bd).reduce((a, b) => a + b, 0);

            banner.innerHTML = `
                <div class="kpi-card" style="padding: 0.75rem 1rem;">
                    <div class="kpi-label">Selected Step</div>
                    <div class="kpi-value" style="font-size: 1.25rem;">#${{target.step}} <span style="font-size:0.8rem;color:#38bdf8;">(${{isPrefill ? 'Prompt Prefill' : 'Decode'}})</span></div>
                </div>
                <div class="kpi-card" style="padding: 0.75rem 1rem;">
                    <div class="kpi-label">Decoded Token</div>
                    <div class="kpi-value" style="font-size: 1.25rem; font-family: monospace;"><code>${{(target.token_text || '').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')}}</code></div>
                </div>
                <div class="kpi-card" style="padding: 0.75rem 1rem;">
                    <div class="kpi-label">Step Total Wall-Clock</div>
                    <div class="kpi-value" style="font-size: 1.25rem; color: #38bdf8;">${{totalStepMs.toFixed(2)}} ms</div>
                </div>
                <div class="kpi-card" style="padding: 0.75rem 1rem;">
                    <div class="kpi-label">Active Device Kernel Time</div>
                    <div class="kpi-value" style="font-size: 1.25rem; color: #06b6d4;">${{deviceTimeMs.toFixed(2)}} ms</div>
                </div>
            `;

            let opEntries = Object.keys(bd).map(k => {{
                const val = bd[k] || 0.0;
                const pct = deviceTimeMs > 0 ? (val / deviceTimeMs * 100) : 0.0;
                const meta = profileData.metadata[k] || {{ name: k, category: 'Other', badge: 'Op', desc: '', scaling: 'Flat' }};
                const color = profileData.colors[k] || '#38bdf8';
                return {{ key: k, name: meta.name, category: meta.category, badge: meta.badge, desc: meta.desc, scaling: meta.scaling, val, pct, color }};
            }});

            if (opSortMode === "duration") {{
                opEntries.sort((a, b) => b.val - a.val);
            }}

            const maxOpVal = Math.max(...opEntries.map(e => e.val)) || 1;
            const barContainer = document.getElementById("step-ops-bars");
            let barsHtml = "";
            opEntries.forEach(op => {{
                if (op.val < 0.0001 && opSortMode === "duration") return;
                const fillW = Math.max(1, (op.val / maxOpVal) * 100);
                barsHtml += `
                    <div class="h-bar-row" onmousemove="showTooltip(event, {{name: '${{op.name}}', domain: '${{op.category}}', step: 'Step #${{target.step}}', start_ms: 0, dur_ms: ${{op.val}}, other: '${{op.desc}}'}})" onmouseleave="hideTooltip()">
                        <div class="h-bar-label">
                            <span class="tag-badge" style="background:${{op.color}}22; color:${{op.color}}; border: 1px solid ${{op.color}}44;">${{op.badge}}</span>
                            <span style="color:#e2e8f0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${{op.name}}">${{op.key}}</span>
                        </div>
                        <div class="h-bar-track">
                            <div class="h-bar-fill" style="width:${{fillW}}%; background:${{op.color}};"></div>
                        </div>
                        <div class="h-bar-val" style="color:${{op.color}};">${{op.val.toFixed(3)}} ms <span style="color:#94a3b8;font-size:0.72rem;">(${{op.pct.toFixed(1)}}%)</span></div>
                    </div>
                `;
            }});
            barContainer.innerHTML = barsHtml;

            const tbody = document.getElementById("step-ops-table-body");
            let tableHtml = "";
            opEntries.forEach((op, rank) => {{
                tableHtml += `
                    <tr>
                        <td><strong>#${{rank + 1}}</strong></td>
                        <td><strong style="color:${{op.color}};">${{op.key}}</strong> <span style="color:#94a3b8;font-size:0.75rem;">(${{op.name}})</span></td>
                        <td><span class="tag-badge" style="background:${{op.color}}22; color:${{op.color}}; border: 1px solid ${{op.color}}44;">${{op.category}}</span></td>
                        <td><strong style="font-family:monospace;color:#fff;">${{op.val.toFixed(3)}} ms</strong></td>
                        <td><strong style="color:#38bdf8;">${{op.pct.toFixed(1)}}%</strong></td>
                        <td style="color:#cbd5e1;font-size:0.8rem;">${{op.desc}} <em>[${{op.scaling}}]</em></td>
                    </tr>
                `;
            }});
            tbody.innerHTML = tableHtml;
        }}

        // 3. Render Operation Scaling (Choose Operation)
        function initOperationSelector() {{
            const select = document.getElementById("op-selector");
            select.innerHTML = "";
            profileData.categories.forEach(cat => {{
                const meta = profileData.metadata[cat] || {{ name: cat, category: 'Op' }};
                const opt = document.createElement("option");
                opt.value = cat;
                opt.innerText = `${{cat}} (${{meta.name}})`;
                if (cat === currentSelectedOp) opt.selected = true;
                select.appendChild(opt);
            }});

            const pills = document.getElementById("op-quick-pills");
            pills.innerHTML = "";
            const quickPillOps = ["Attn_Compute", "FFN_Gate_Up_Linear", "LM_Head", "QKV_Linear", "O_Linear", "RoPE", "Sampling"];
            quickPillOps.forEach(op => {{
                const btn = document.createElement("button");
                btn.className = `op-pill-btn ${{op === currentSelectedOp ? 'active' : ''}}`;
                btn.dataset.op = op;
                btn.innerText = op;
                btn.onclick = () => selectOperation(op);
                pills.appendChild(btn);
            }});
        }}

        function selectOperation(opName) {{
            currentSelectedOp = opName;
            const sel = document.getElementById("op-selector");
            if (sel && sel.value !== opName) sel.value = opName;
            document.querySelectorAll(".op-pill-btn").forEach(btn => {{
                btn.classList.toggle("active", btn.dataset.op === opName);
            }});
            renderOperationScaling();
        }}

        function renderOperationScaling() {{
            const op = currentSelectedOp;
            if (!sampledTokens || sampledTokens.length === 0) return;

            const meta = profileData.metadata[op] || {{ name: op, category: 'Operation', badge: 'Op', desc: '', scaling: 'Flat' }};
            const color = profileData.colors[op] || '#38bdf8';

            const opPoints = sampledTokens.map(t => {{
                const val = (t.breakdown && t.breakdown[op]) || 0.0;
                const total = t.three_metrics ? t.three_metrics.total_latency_ms : t.total_latency_ms;
                const pct = total > 0 ? (val / total * 100) : 0.0;
                return {{ step: t.step, val, pct, total, token: t.token_text }};
            }});

            const prefillVal = opPoints[0] ? opPoints[0].val : 0.0;
            const firstDecodeVal = opPoints[1] ? opPoints[1].val : (opPoints[0] ? opPoints[0].val : 0.0);
            const lastDecodeVal = opPoints[opPoints.length - 1] ? opPoints[opPoints.length - 1].val : 0.0;

            const insightCard = document.getElementById("op-insight-card");
            insightCard.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 0.75rem;">
                    <div>
                        <span class="tag-badge" style="background:${{color}}22; color:${{color}}; border: 1px solid ${{color}}44; font-size: 0.8rem; padding: 0.2rem 0.5rem;">${{meta.category}}</span>
                        <h3 style="font-size: 1.25rem; font-weight: 700; color: #fff; margin-top: 0.35rem;">${{op}} <span style="font-size: 0.9rem; font-weight: normal; color: #94a3b8;">(${{meta.name}})</span></h3>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Scaling Nature</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8;">${{meta.scaling}}</div>
                    </div>
                </div>
                <p style="color: #cbd5e1; font-size: 0.88rem; margin-bottom: 1rem;">${{meta.desc}}</p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem;">
                    <div style="background: rgba(15,23,42,0.8); border: 1px solid #1e293b; padding: 0.6rem 0.85rem; border-radius: 0.4rem;">
                        <div style="font-size: 0.72rem; color: #94a3b8;">Step #0 (Prefill)</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: ${{color}};">${{prefillVal.toFixed(3)}} ms</div>
                    </div>
                    <div style="background: rgba(15,23,42,0.8); border: 1px solid #1e293b; padding: 0.6rem 0.85rem; border-radius: 0.4rem;">
                        <div style="font-size: 0.72rem; color: #94a3b8;">Step #${{opPoints[1] ? opPoints[1].step : 0}} (First Decode)</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #38bdf8;">${{firstDecodeVal.toFixed(3)}} ms</div>
                    </div>
                    <div style="background: rgba(15,23,42,0.8); border: 1px solid #1e293b; padding: 0.6rem 0.85rem; border-radius: 0.4rem;">
                        <div style="font-size: 0.72rem; color: #94a3b8;">Step #${{opPoints[opPoints.length - 1].step}} (Latest Decode)</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #10b981;">${{lastDecodeVal.toFixed(3)}} ms</div>
                    </div>
                </div>
            `;

            const container = document.getElementById("op-scaling-chart-container");
            const w = 1200, h = 300, padL = 70, padR = 40, padT = 30, padB = 45;
            const chartW = w - padL - padR;
            const chartH = h - padT - padB;

            const maxVal = Math.max(...opPoints.map(p => p.val)) * 1.25 || 0.1;
            const stepW = chartW / opPoints.length;
            const barW = Math.max(18, Math.min(60, stepW * 0.55));

            let svg = `<svg viewBox="0 0 ${{w}} ${{h}}" class="chart-svg">`;
            for (let i = 0; i <= 4; i++) {{
                const yVal = (maxVal / 4) * i;
                const yPos = padT + chartH - (chartH / 4) * i;
                svg += `<line x1="${{padL}}" y1="${{yPos}}" x2="${{w - padR}}" y2="${{yPos}}" class="grid-line" />`;
                svg += `<text x="${{padL - 10}}" y="${{yPos + 4}}" text-anchor="end">${{yVal.toFixed(3)}} ms</text>`;
            }}

            opPoints.forEach((pt, idx) => {{
                const x = padL + idx * stepW + (stepW - barW) / 2;
                const barH = (pt.val / maxVal) * chartH;
                const y = padT + chartH - barH;

                svg += `<rect x="${{x}}" y="${{y}}" width="${{barW}}" height="${{barH}}" fill="${{color}}" rx="3"
                    onmousemove="showTooltip(event, {{name: '${{meta.name}}', domain: '${{meta.category}}', step: 'Step #${{pt.step}} (${{pt.step === 0 ? 'Prefill' : 'Decode'}})', start_ms: 0, dur_ms: ${{pt.val}}, other: '${{pt.val.toFixed(3)}} ms (${{pt.pct.toFixed(1)}}% of step)'}})"
                    onmouseleave="hideTooltip()" />`;

                svg += `<text x="${{x + barW / 2}}" y="${{Math.max(padT + 12, y - 6)}}" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">${{pt.val.toFixed(3)}} ms</text>`;
                svg += `<text x="${{x + barW / 2}}" y="${{padT + chartH + 20}}" text-anchor="middle">#${{pt.step}} ${{pt.step === 0 ? '(Prefill)' : ''}}</text>`;
            }});

            svg += `</svg>`;
            container.innerHTML = svg;
        }}

        // 4. Render Stacked Bar Chart for the Three Physical Metrics
        function renderStackedBarChart() {{
            const metricsConfig = [
                {{ key: 'compute_ms', pctKey: 'compute_pct', name: '3. Pure Math Compute', color: '#ef4444', desc: 'Active Tensor Cores & Vector ALUs' }},
                {{ key: 'memory_wait_ms', pctKey: 'memory_wait_pct', name: '2. VRAM Data Wait', color: '#06b6d4', desc: 'Memory bus bandwidth saturation / weight streaming' }},
                {{ key: 'cpu_idle_ms', pctKey: 'cpu_idle_pct', name: '1. CPU Launch & Driver Gaps', color: '#3b82f6', desc: 'Host CPU dispatch starvation & empty GPU queue' }}
            ];

            const legendContainer = document.getElementById('op-legend');
            legendContainer.innerHTML = '';
            [metricsConfig[2], metricsConfig[1], metricsConfig[0]].forEach(m => {{
                const item = document.createElement('div');
                item.style.display = 'flex';
                item.style.alignItems = 'center';
                item.style.fontSize = '0.85rem';
                item.innerHTML = `<span style="width:14px;height:14px;border-radius:3px;margin-right:0.4rem;background:${{m.color}};"></span><strong>${{m.name}}</strong><span style="color:#94a3b8;font-size:0.75rem;margin-left:0.35rem;">(${{m.desc}})</span>`;
                legendContainer.appendChild(item);
            }});

            const container = document.getElementById('stacked-bar-container');
            const data = sampledTokens;
            const w = 1200, h = 360, padL = 70, padR = 20, padT = 20, padB = 40;
            const chartW = w - padL - padR;
            const chartH = h - padT - padB;

            const maxVal = Math.max(...data.map(d => (d.three_metrics ? d.three_metrics.total_latency_ms : d.total_latency_ms))) * 1.15 || 1;
            const barW = Math.max(16, Math.min(65, (chartW / data.length) * 0.65));
            const stepW = chartW / data.length;

            let svg = `<svg viewBox="0 0 ${{w}} ${{h}}" class="chart-svg">`;
            for (let i = 0; i <= 5; i++) {{
                const yVal = (maxVal / 5) * i;
                const yPos = padT + chartH - (chartH / 5) * i;
                svg += `<line x1="${{padL}}" y1="${{yPos}}" x2="${{w - padR}}" y2="${{yPos}}" class="grid-line" />`;
                svg += `<text x="${{padL - 10}}" y="${{yPos + 4}}" text-anchor="end">${{yVal.toFixed(1)}} ms</text>`;
            }}

            data.forEach((d, idx) => {{
                const m = d.three_metrics || {{
                    total_latency_ms: d.total_latency_ms,
                    cpu_idle_ms: d.total_latency_ms * 0.74,
                    memory_wait_ms: d.total_latency_ms * 0.26,
                    compute_ms: 0.021,
                    cpu_idle_pct: 74.0,
                    memory_wait_pct: 26.0,
                    compute_pct: 0.04,
                }};
                const x = padL + idx * stepW + (stepW - barW) / 2;
                let currentBottom = padT + chartH;

                const stackItems = [
                    {{ val: m.compute_ms, pct: m.compute_pct, name: '3. Pure Math Compute', color: '#ef4444', desc: 'Tensor Cores & Vector ALUs' }},
                    {{ val: m.memory_wait_ms, pct: m.memory_wait_pct, name: '2. VRAM Data Wait', color: '#06b6d4', desc: 'Memory bus bandwidth saturation' }},
                    {{ val: m.cpu_idle_ms, pct: m.cpu_idle_pct, name: '1. CPU Launch & Driver Gaps', color: '#3b82f6', desc: 'Host CPU dispatch starvation' }}
                ];

                stackItems.forEach(item => {{
                    if (item.val <= 0.0001) return;
                    const barH = (item.val / maxVal) * chartH;
                    const y = currentBottom - barH;

                    svg += `<rect class="bar-segment" x="${{x}}" y="${{y}}" width="${{barW}}" height="${{barH}}" fill="${{item.color}}"
                        onmousemove="showTooltip(event, {{name: '${{item.name}}', domain: '${{item.desc}}', step: 'Step #${{d.step}}', start_ms: 0, dur_ms: ${{item.val}}, other: '${{item.pct}}% of Step Latency (${{item.val.toFixed(2)}} ms)'}})"
                        onmouseleave="hideTooltip()" />`;
                    currentBottom = y;
                }});

                svg += `<text x="${{x + barW/2}}" y="${{padT + chartH + 18}}" text-anchor="middle">#${{d.step}}</text>`;
            }});

            svg += `</svg>`;
            container.innerHTML = svg;
        }}

        // 5. Render Gantt Timeline
        function initGanttTabs() {{
            const tabsContainer = document.getElementById("step-tabs");
            tabsContainer.innerHTML = "";
            availableSteps.forEach(s => {{
                const btn = document.createElement("button");
                btn.className = `step-btn ${{s === currentStep ? 'active' : ''}}`;
                btn.dataset.step = s;
                btn.innerText = s === 0 ? "Step #0 (Prefill)" : `Step #${{s}}`;
                btn.onclick = () => selectTimeStep(s);
                tabsContainer.appendChild(btn);
            }});
        }}

        function renderGantt() {{
            const d = timelineData[currentStep];
            if (!d) return;

            const maxMs = d.duration_ms || 15.0;
            const containerW = Math.max(1000, 1250 * zoomScale);

            const scaleContainer = document.getElementById("time-scale");
            scaleContainer.style.width = `${{containerW}}px`;
            scaleContainer.innerHTML = "";

            const tickCount = 10;
            for (let i = 0; i <= tickCount; i++) {{
                const fraction = i / tickCount;
                const timeVal = (maxMs * fraction).toFixed(2);
                const x = fraction * (containerW - 80);
                const tick = document.createElement("div");
                tick.style.position = "absolute";
                tick.style.left = `${{x + 10}}px`;
                tick.style.top = "6px";
                tick.style.fontSize = "10px";
                tick.style.fontFamily = "monospace";
                tick.style.color = "#9ca3af";
                tick.innerText = `${{timeVal}} ms`;
                scaleContainer.appendChild(tick);
            }}

            const rows = {{
                "cpu": document.getElementById("row-cpu"),
                "memory-ops": document.getElementById("row-memory-ops"),
                "compute-ops": document.getElementById("row-compute-ops"),
                "vram-curve": document.getElementById("row-vram-curve")
            }};

            Object.values(rows).forEach(r => {{
                r.style.width = `${{containerW}}px`;
                r.innerHTML = "";
            }});

            (d.events || []).forEach(ev => {{
                const targetRow = rows[ev.row];
                if (!targetRow) return;

                const left = (ev.start_ms / maxMs) * (containerW - 40);
                const width = Math.max(8, (ev.dur_ms / maxMs) * (containerW - 40));

                const block = document.createElement("div");
                block.className = `gantt-block ${{ev.cat}}`;
                block.style.left = `${{left + 10}}px`;
                block.style.width = `${{width}}px`;
                const showSub = width >= 45;
                block.innerHTML = `
                    <div class="block-title" title="${{ev.name}}">${{ev.name}}</div>
                    ${{showSub && ev.sub ? `<div class="block-sub">${{ev.sub}}</div>` : ''}}
                `;

                block.onmouseenter = (e) => showTooltip(e, ev);
                block.onmouseleave = hideTooltip;
                block.onclick = () => selectEvent(ev, block);

                targetRow.appendChild(block);
            }});

            renderMemoryCurve(rows["vram-curve"], d.memory_curve || [], maxMs, containerW);

            const firstGpu = (d.events || []).find(e => e.row === "compute-ops") || (d.events || [])[0];
            if (firstGpu) updateInspector(firstGpu);
        }}

        function renderMemoryCurve(container, curveData, maxMs, width) {{
            const h = 72;
            if (!curveData || curveData.length === 0) return;
            const minV = Math.min(...curveData.map(c => c.vram_mb)) - 5;
            const maxV = Math.max(...curveData.map(c => c.vram_mb)) + 15;

            let points = [];
            curveData.forEach(pt => {{
                const x = 10 + (pt.t / maxMs) * (width - 40);
                const y = h - 14 - ((pt.vram_mb - minV) / (maxV - minV || 1)) * (h - 26);
                points.push(`${{x}},${{y}}`);
            }});

            let svg = `
                <svg width="${{width}}" height="${{h}}" style="position:absolute; top:0; left:0; pointer-events:none;">
                    <defs>
                        <linearGradient id="memGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stop-color="#10b981" stop-opacity="0.35"/>
                            <stop offset="100%" stop-color="#10b981" stop-opacity="0.0"/>
                        </linearGradient>
                    </defs>
                    <polygon points="10,${{h-10}} ${{points.join(' ')}} ${{width-30}},${{h-10}}" fill="url(#memGrad)" />
                    <polyline points="${{points.join(' ')}}" fill="none" stroke="#10b981" stroke-width="2" />
            `;

            curveData.forEach(pt => {{
                const x = 10 + (pt.t / maxMs) * (width - 40);
                const y = h - 14 - ((pt.vram_mb - minV) / (maxV - minV || 1)) * (h - 26);
                svg += `
                    <circle cx="${{x}}" cy="${{y}}" r="3" fill="#0b0f19" stroke="#10b981" stroke-width="2" />
                    <text x="${{x + 6}}" y="${{y - 4}}" fill="#10b981" font-size="10" font-family="monospace">${{pt.vram_mb}} MB</text>
                `;
            }});

            svg += `</svg>`;
            container.innerHTML = svg;
        }}

        function selectEvent(ev, element) {{
            document.querySelectorAll(".gantt-block").forEach(b => b.classList.remove("active-selection"));
            if (element) element.classList.add("active-selection");
            updateInspector(ev);
        }}

        function updateInspector(ev) {{
            if (!ev) return;
            document.getElementById("insp-name").innerText = ev.name;
            document.getElementById("insp-domain").innerText = ev.domain || '-';
            document.getElementById("insp-domain").style.color = ev.row === "compute-ops" ? "#f87171" : (ev.row === "memory-ops" ? "#22d3ee" : "#38bdf8");
            document.getElementById("insp-time").innerText = `${{ev.start_ms.toFixed(2)}} ms ➔ ${{((ev.start_ms + ev.dur_ms)).toFixed(2)}} ms (${{(ev.dur_ms*1000).toFixed(0)}} μs)`;
            document.getElementById("insp-step").innerText = ev.step || '-';
            document.getElementById("insp-other").innerText = ev.other || '-';
            document.getElementById("insp-hw").innerText = ev.hw || '-';
        }}

        function setZoom(val) {{
            zoomScale = val;
            renderGantt();
        }}
        function zoom(factor) {{
            zoomScale = Math.max(0.5, Math.min(8.0, zoomScale * factor));
            renderGantt();
        }}
        function resetZoom() {{
            zoomScale = 1.0;
            renderGantt();
        }}

        // Initialize All Dashboard Visualizations
        renderOverallLatencyChart();
        initStepSelector();
        renderStepOperations();
        initOperationSelector();
        renderOperationScaling();
        renderStackedBarChart();
        if (availableSteps.length > 0) {{
            initGanttTabs();
            renderGantt();
        }}
    </script>
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[*] Standalone HTML profile dashboard generated at: {output_file}")


def save_json_metrics(
    token_records: List[Dict[str, Any]],
    prompt: str = "",
    output_file: str = "token_metrics.json",
    timeline_records: Optional[Dict[int, Any]] = None,
):
    """Saves token metrics as a structured JSON file for downstream analysis."""
    total_tokens = len(token_records)
    sampled_records = [r for r in token_records if r.get("is_sampled", True) and r.get("breakdown")]
    if not sampled_records:
        sampled_records = token_records

    decode_records = [r for r in token_records if r["step"] > 0]
    avg_decode_ms = (
        round(sum(r["total_latency_ms"] for r in decode_records) / len(decode_records), 2)
        if decode_records
        else (token_records[0]["total_latency_ms"] if token_records else 0.0)
    )
    prefill_ms = round(token_records[0]["total_latency_ms"], 2) if token_records else 0.0
    throughput = round(1000.0 / avg_decode_ms, 1) if avg_decode_ms > 0 else 0.0

    data = {
        "prompt": prompt,
        "tokens": token_records,
        "timelines": timeline_records or {},
        "categories": FINE_GRAINED_CATEGORIES,
        "metadata": OPERATION_METADATA,
        "summary": {
            "total_tokens": total_tokens,
            "sampled_checkpoints": len(sampled_records),
            "prefill_latency_ms": prefill_ms,
            "avg_decode_latency_ms": avg_decode_ms,
            "throughput_tok_per_sec": throughput,
            "three_metrics_avg": {
                "cpu_idle_ms": round(sum(r.get("three_metrics", {}).get("cpu_idle_ms", 0.0) for r in sampled_records) / len(sampled_records), 2) if sampled_records else 0.0,
                "memory_wait_ms": round(sum(r.get("three_metrics", {}).get("memory_wait_ms", 0.0) for r in sampled_records) / len(sampled_records), 2) if sampled_records else 0.0,
                "compute_ms": round(sum(r.get("three_metrics", {}).get("compute_ms", 0.0) for r in sampled_records) / len(sampled_records), 3) if sampled_records else 0.0,
                "duty_cycle_pct": round(sum(r.get("three_metrics", {}).get("duty_cycle_pct", 0.0) for r in sampled_records) / len(sampled_records), 1) if sampled_records else 0.0,
            },
        },
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[*] Token metrics JSON saved at: {output_file}")
