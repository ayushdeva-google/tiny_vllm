# Chapter: Systems Profiling and Hardware Interplay in Autoregressive LLM Inference

> **Abstract**: Autoregressive decoding in Large Language Models (LLMs) presents unique systems bottlenecks fundamentally distinct from prompt prefill and batched training. This chapter consolidates empirical profiling experiments conducted on **LLaMA-3.2-1B-Instruct** running on an **NVIDIA L4 GPU**. We establish a rigorous physical framework decomposing token latency into three fundamental components: Host CPU Launch Gaps, VRAM Data Wait, and Pure Math Compute. We address core systems paradoxes—why modern GPUs sit predominantly idle despite asynchronous CUDA dispatch, how memory bus saturation physically proves hardware starvation, why memory-compute overlap prevents sequential kernel decomposition, and how future optimizations structurally alter these hardware metrics.

---

## 1. The Physical Hardware Hierarchy & Interplay

To analyze inference bottlenecks, one must view the hardware not as an abstract compute engine, but as a hierarchical network of memory tiers and execution units separated by physical buses.

```
┌────────────────────────────────────────────────────────────────────────┐
│                              HOST CPU                                  │
│  • System Memory (RAM): Python runtime, Tokenizer, Dispatch buffers    │
│  • CPU Execution Cores: HuggingFace BPE, PyTorch op scheduling         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
             ═══════════════════════╪═══════════════════════
                    PCIe Gen4 x16 Bus (~31.5 GB/s bidirectional)
                    • Host-to-Device (HtoD): input_ids (~80 Bytes)
                    • Device-to-Host (DtoH): next_token_id.item() (8 Bytes)
             ═══════════════════════╪═══════════════════════
                                    │
┌───────────────────────────────────┴────────────────────────────────────┐
│                        NVIDIA L4 GPU (AD104)                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    VRAM (GDDR6, 24 GB)                           │  │
│  │  • Model Parameters: 2.46 GB resident (BF16)                     │  │
│  │  • KV Cache & Activations: [1, seq_len, 2048]                    │  │
│  └────────────────────────────────┬─────────────────────────────────┘  │
│                                   │ Memory Bus (Peak: 300 GB/s)        │
│  ┌────────────────────────────────┴─────────────────────────────────┐  │
│  │                    On-Chip Silicon                               │  │
│  │  • L2 Cache: 24 MB (Shared across all SMs)                       │  │
│  │  • Streaming Multiprocessors (SMs) / Registers (~20 TB/s)        │  │
│  │  • 4th Gen Tensor Cores & Vector ALUs (Peak: 120 TFLOP/s BF16)   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Data Transfer Frequencies: PCIe vs. VRAM Round-Trips

A common misconception is that inference is bottlenecked by PCIe communication between CPU and GPU. Empirical tracking reveals two completely different orders of magnitude:

1. **PCIe Transfers (Host $\leftrightarrow$ Device)**:
   - **Frequency**: Strictly **twice per token**.
   - **Volume**: Negligible (< 100 bytes). The host pushes token IDs ($~80$ bytes for initial prompt or $2$ bytes per decode token), and retrieves a single 8-byte scalar (`next_token_id.item()`).
   - **Impact**: Latency penalty is driven not by PCIe bandwidth, but by the mandatory **synchronization barrier** forced by `.item()` (see Section 4.1).

2. **VRAM Transfers (GPU Silicon $\leftrightarrow$ High-Bandwidth Memory)**:
   - **Frequency**: **Hundreds of transfers per token**.
   - **Cause**: Because standard deep learning operations are unfused, **every individual kernel must read inputs from VRAM DRAM and write intermediate activations back to VRAM DRAM**.

#### Case Study: The Five VRAM Round-Trips in a Single SwiGLU Block
Consider the MLP block in LLaMA-3.2 ($d=2048, d_{\text{mlp}}=8192$):
```
VRAM (Global DRAM)
  ▲          ▲          ▼            ▲           ▼            ▲           ▼
  │ Read x   │ Read x   │ Write      │ Read      │ Write      │ Read both │ Write
  │          │          │ gate_out   │ gate_out  │ silu_out   │ & silu    │ final
┌─┴────────┐┌┴────────┐ └──────────┐┌┴─────────┐ └──────────┐┌┴───────────┤ └────────┐
│Gate_Proj ││ Up_Proj │            ││   SiLU   │            ││  Multiply  │  │Down_Proj│
│ (GEMV)   ││ (GEMV)  │            ││ (Vector) │            ││  (Vector)  │  │ (GEMV)   │
└──────────┘└─────────┘            ▼└──────────┘            ▼└─────────────┘ └─────────┘
```
1. `gate_proj(x)`: Reads $x$ from VRAM $\rightarrow$ computes GEMV $\rightarrow$ writes `gate_out` to VRAM (**Round-trip 1**).
2. `up_proj(x)`: Reads $x$ from VRAM $\rightarrow$ computes GEMV $\rightarrow$ writes `up_out` to VRAM (**Round-trip 2**).
3. `silu(gate_out)`: Reads `gate_out` from VRAM $\rightarrow$ applies activation $\rightarrow$ writes `silu_out` to VRAM (**Round-trip 3**).
4. `mul(silu_out, up_out)`: Reads `silu_out` and `up_out` from VRAM $\rightarrow$ multiplies $\rightarrow$ writes intermediate to VRAM (**Round-trip 4**).
5. `down_proj(...)`: Reads intermediate from VRAM $\rightarrow$ computes GEMV $\rightarrow$ writes output to VRAM (**Round-trip 5**).

Across 16 transformer layers, the GPU executes over 80 round-trips to VRAM just for MLP activations, reading and writing tens of megabytes of ephemeral tensors that exist for mere microseconds.

---

## 2. Prefill vs. Decode: Arithmetic Intensity and the Roofline Model

The fundamental performance regime of an LLM depends on whether it is processing the prompt (**prefill**) or generating new tokens (**decode**).

```
                        ROOFLINE MODEL (NVIDIA L4)
       Performance
        (TFLOP/s)
          120 ┬─────────────────────────────────────────── Peak Compute: 120 TFLOP/s
              │                                      /
              │                                     /  Prefill (GEMM, Batch M > 1)
              │                                    /
              │                                   /
              │                                  /
              │                                 /
              │                                /
              │  Decode (GEMV, M=1)           /
              │  1 FLOP/Byte (~0.24 TFLOP/s) /
            0 ┴──────────▲──────────────────▲─────────────
                         │                  │
                      1.0                400.0 (Ridge Point)
                              Arithmetic Intensity (FLOPs / Byte)
```

### 2.1 Mathematical Formulation of the Ridge Point
For any hardware architecture, the break-even arithmetic intensity ($I_{\text{ridge}}$) defining the boundary between memory-bound and compute-bound regimes is:

$$I_{\text{ridge}} = \frac{\text{Peak Compute Performance (FLOP/s)}}{\text{Peak Memory Bandwidth (Bytes/s)}}$$

On the NVIDIA L4 GPU:
- **Peak Compute ($\text{Peak}_{\text{FLOPs}}$)**: $120\text{ TFLOP/s} = 120 \times 10^{12}\text{ FLOP/s}$ (BF16 Tensor Cores)
- **Peak Bandwidth ($\text{Peak}_{\text{BW}}$)**: $300\text{ GB/s} = 300 \times 10^9\text{ Bytes/s}$ (GDDR6)

$$I_{\text{ridge}} = \frac{120 \times 10^{12}}{300 \times 10^9} = \mathbf{400\text{ FLOPs / Byte}}$$

Any operation with an arithmetic intensity below $400\text{ FLOP/Byte}$ cannot saturate the Tensor Cores and is physically bottlenecked by memory bandwidth.

### 2.2 GEMM vs. GEMV in Autoregressive Generation
- **Prompt Prefill (GEMM: $M > 1$)**: When processing an initial prompt of $S$ tokens, input activations form an $S \times d$ matrix. Each loaded weight parameter is multiplied by $S$ distinct activation vectors:
  $$I_{\text{prefill}} \approx \frac{2 \times S \times \text{Params}}{2 \times \text{Params}} = S\text{ FLOPs / Byte}$$
  For prompts where $S \ge 400$, arithmetic intensity crosses $I_{\text{ridge}}$, reaching compute saturation.

- **Autoregressive Decode (GEMV: $M = 1$)**: When generating one new token, the input is a single $1 \times d$ vector. Each 2-byte parameter loaded from VRAM is used for exactly one multiply-accumulate operation ($2\text{ FLOPs}$):
  $$I_{\text{decode}} = \frac{2 \text{ FLOPs}}{2 \text{ Bytes}} = \mathbf{1.0\text{ FLOP / Byte}}$$

Because $1.0 \ll 400$, autoregressive decode on batch size 1 operates in extreme memory starvation: the Tensor Cores sit idle for $\mathbf{99.75\%}$ of their clock cycles waiting on memory.

---

## 3. Empirical Decomposition: The Three Physical Metrics

By correlating high-resolution PyTorch Profiler Kineto traces with silicon hardware counters, we decompose total wall-clock token generation time into **three mutually exclusive physical components**:

$$\text{Latency}_{\text{Token}} = T_{\text{Host Gaps}} + T_{\text{VRAM Wait}} + T_{\text{Math Compute}}$$

```
Observed Wall-Clock Latency per Token (~57 ms on NVIDIA L4):
┌──────────────────────────────────────┬────────────────────────┬────────┐
│  1. CPU Launch & Driver Gaps         │ 2. VRAM Data Wait      │ 3. Math│
│  (Empty pipeline between 844 kernels)│ (Streaming 2.46 GB)    │ Compute│
│  41.81 ms (73.4%)                    │ 15.10 ms (26.5%)       │ 0.02 ms│
└──────────────────────────────────────┴────────────────────────┴────────┘
                                                                 ▲
                                                      Barely visible (0.04%)
```

### Empirical Measurements (LLaMA-3.2-1B on NVIDIA L4, Step 100)

| Metric | Real Experiment Value | Physical Mechanism |
|---|---|---|
| **Total Wall-Clock Latency** | **$56.93\text{ ms}$** | End-to-end token latency from prompt/previous token to next token |
| **CPU Launch & Driver Gaps** | **$41.81\text{ ms}$ ($73.4\%$)** | GPU execution pipeline sitting 100% idle between micro-kernel dispatches |
| **VRAM Data Wait** | **$15.10\text{ ms}$ ($26.5\%$)** | GPU execution units stalled waiting for weights/activations from VRAM |
| **Pure Math Compute** | **$0.021\text{ ms}$ ($0.04\%$)** | Actual time Tensor Cores and ALUs spent computing FLOPs |
| **Active GPU Duty Cycle** | **$26.6\%$** | Fraction of wall-clock time any GPU silicon core was active |
| **Kernels Launched per Token** | **844 kernels** | Discrete CUDA kernel invocations per single token |

---

## 4. Systems Paradoxes & Core Architectural Learnings

### 4.1 The Asynchronous Scheduling Paradox (The Chef & Eater)
**The Question**: *CUDA kernel launches (`cudaLaunchKernel`) are asynchronous non-blocking C calls. The CPU enqueues commands and returns immediately. Why then does CPU dispatch overhead degrade GPU utilization during inference?*

**The Resolution**: While asynchronous queues decouple CPU and GPU, they only prevent starvation if the producer's launch rate exceeds or matches the consumer's execution rate.

Think of the CPU as a **Chef** writing order tickets, and the GPU as an **Eater** consuming dishes:
- **Case 1: Batched Compute or Prefill**: The Chef takes $15\ \mu\text{s}$ to enqueue a large GEMM kernel that runs for $5,000\ \mu\text{s}$ ($5\text{ ms}$). The Chef stays dozens of kernels ahead; the GPU command buffer never empties.
- **Case 2: Single-Token Unfused Decode**: Generating one token in an unfused 16-layer transformer requires launching **844 separate micro-kernels** (RMSNorms, RoPE rotations, KV indexing, Softmax, elementwise operations).
  - A small `RMSNorm` or `RoPE` kernel executes on L4 silicon in **$1\text{ to }3\ \mu\text{s}$**.
  - The Python interpreter, PyTorch dispatcher, autograd state checkers, and CUDA driver take **$15\text{ to }40\ \mu\text{s}$** to prepare and launch the next kernel.

$$T_{\text{GPU Kernel Execution}} (2\ \mu\text{s}) \ll T_{\text{CPU Dispatch Overhead}} (30\ \mu\text{s})$$

```
CPU Thread:  [-- Enqueue Op 1 (30 μs) --] ────────► [-- Enqueue Op 2 (30 μs) --]
                                                     ▲
GPU Hardware:[ Op 1 (2 μs) ] [ IDLE WAITING FOR CPU (28 μs) ] [ Op 2 ]
```

The GPU consumes each micro-kernel instantly, looks in its queue, finds it empty, and goes completely idle. This phenomenon is **Host Launch Starvation** (GPU Pipeline Underflow).

#### The Forced Pipeline Drain: `.item()`
Autoregressive sampling requires the scalar token value to update greedy state or evaluate stop tokens:
```python
next_token = torch.argmax(logits[:, -1, :], dim=-1)
token_val = next_token.item()  # <--- FORCED DEVICE-TO-HOST SYNCHRONIZATION
```
Calling `.item()` halts the CPU thread until all previously enqueued GPU kernels finish and the 8-byte integer is transferred over PCIe. This completely drains the CUDA command queue. Every single token begins from a dead stop with a cold execution pipeline.

---

### 4.2 Physical Proof: Memory Wait vs. Math Compute
**The Question**: *When the GPU is active for $15.12\text{ ms}$, how do we prove it is stalled waiting on VRAM rather than busy doing math?*

**The Resolution**: We establish proof through two independent physical constraints:

#### Proof 1: The Memory Wire Speedometer
1. **Payload**: LLaMA-3.2-1B in `bfloat16` contains $1.23 \times 10^9$ parameters $\times 2\text{ bytes} \approx \mathbf{2.46\text{ GB}}$ of weights.
2. **Physical Wire Limit**: The NVIDIA L4 GDDR6 memory bus has a maximum theoretical bandwidth of **$300\text{ GB/s}$**.
3. **Minimum Wire Transfer Time**: Even if the Tensor Cores had infinite frequency ($0.0\text{ ms}$ math time), the absolute physical minimum time to stream 2.46 GB through the memory controllers is:
   $$T_{\text{min\_wire}} = \frac{2.46\text{ GB}}{300\text{ GB/s}} = \mathbf{8.20\text{ ms}}$$

In our profiling trace:
- **Measured Kernel Duration**: $15.12\text{ ms}$
- **Achieved Bandwidth**: $\frac{2.46\text{ GB}}{0.01512\text{ s}} \approx \mathbf{162.7\text{ GB/s}}$ ($54.2\%$ of theoretical peak across the entire step, with individual GEMV kernels reaching **$252.5\text{ GB/s}$ / $84.2\%$** of wire capacity).
- **Compute Flops Required**: $2 \times 1.23 \times 10^9 = 2.46\text{ GFLOPs}$.
- **Pure Math Time**: $\frac{2.46 \times 10^9}{120 \times 10^{12}} = \mathbf{0.0000205\text{ s}} = \mathbf{0.0205\text{ ms}}\ (20.5\ \mu\text{s})$.

The compute units finished their math in $20.5\ \mu\text{s}$ and spent the remaining $15.10\text{ ms}$ ($99.86\%$) waiting for GDDR6 memory controllers to deliver weights.

#### Proof 2: Hardware Performance Counters
Nsight Compute (`ncu`) monitors hardware warp schedulers cycle-by-cycle:
- **`Stall Math Pipe Throttle`**: Warp paused because ALUs or Tensor Cores are occupied $\rightarrow$ **$\approx 0\%$**.
- **`Stall Long Scoreboard`**: Warp paused waiting for an outstanding memory request from VRAM/L2 $\rightarrow$ **$75\% - 85\%$ of all cycles**.

This hardware counter directly proves that the SM execution units were starved of operands waiting on memory transactions.

---

### 4.3 Why Isn't Memory Wait Sequential on a Timeline? (Micro-Interleaving)
**The Question**: *Can we visualize memory wait and compute as sequential segments on a timeline (e.g. 15 ms loading weights followed by 0.02 ms doing math)?*

**The Resolution**: No. At the micro-architectural level, memory transfers and compute are **deeply interleaved via hardware pipelined double-buffering (tiling)**.

```
Sequential (Idealized, Incorrect Model):
[──────────────── 15 ms: Stream all 2.46 GB weights from VRAM ────────────────] ➔ [ 0.02 ms Math ]

Micro-Interleaved (Physical Reality inside GPU):
[Wait 200 cycles][2c Math][Wait 200 cycles][2c Math][Wait 200 cycles][2c Math] ...
```

Inside a GEMV kernel:
1. The SM issues an asynchronous global load instruction for a small matrix tile (e.g. $128 \times 64$ elements) into Shared Memory/Registers.
2. It takes **$\sim 200$ clock cycles** for bytes to travel across the memory bus (the warp stalls on `Long Scoreboard`).
3. Once the tile arrives, Tensor Cores process the tile in **$\sim 2$ clock cycles**.
4. Concurrently, memory controllers are already fetching the next tile into a secondary buffer.

Because memory access latency ($200\text{ cycles}$) dwarfs math latency ($2\text{ cycles}$), memory and math occur concurrently inside the same microsecond, but the hardware is stalled on memory for $99\%$ of the time.

Profilers represent this accurately using **Parallel Swimlane Tracks**:
- **Host CPU Track**: Shows dispatch bursts vs. the blocking `.item()` synchronization barrier.
- **Compute Track**: Shows each active GPU kernel and highlights idle gaps.
- **Memory Bus Track**: Displays memory bus throughput saturation during each kernel execution.

---

### 4.4 Scaling Behavior: Sequence Length ($S$) and the KV Cache
In autoregressive generation, sequence length $S$ impacts operations differently depending on caching:

| Component | Baseline Without KV Cache | Production With KV Cache |
|---|---|---|
| **Linear Projections ($Q, K, V, O, \text{FFN}$)** | Scales as $O(S)$ due to full-prompt recomputation | Strictly **$O(1)$** fixed GEMV for 1 new token |
| **Attention Matrix ($Q K^T$)** | Quadratic $O(S^2)$ recomputation from scratch | Scales as **$O(S)$** dot-products against cached keys |
| **Memory Footprint** | Static weights + transient activation tensors | Increases monotonically with $S$: $2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times S \times 2\text{ B}$ |

---

## 5. Profiling Methodology & Tooling Hierarchy

Comprehensive systems analysis requires instrumentation across multiple abstraction layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Macro / Framework Layer: PyTorch Profiler (Kineto)                 │
│    • Operator call stack, Chrome Trace JSON export (`step_N_trace.json`)│
│    • Sampled Profiling (profiling step 0, interval K, and last step)   │
│    • Eliminates multi-gigabyte trace memory blowup over 2048 tokens    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────┴────────────────────────────────────┐
│ 2. System / Driver Layer: NVIDIA Nsight Systems (`nsys`)               │
│    • NVTX user annotations (`torch.autograd.profiler.emit_nvtx()`)     │
│    • Hardware API triggers (`cudaProfilerStart()` / `Stop()`)          │
│    • CPU-GPU stream correlation, OS context switches, Driver overhead │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────┴────────────────────────────────────┐
│ 3. Silicon / Micro-Architecture Layer: NVIDIA Nsight Compute (`ncu`)   │
│    • Hardware performance counters, Tensor Core pipe occupancy         │
│    • Stall reason breakdown (`Stall Long Scoreboard`, `Math Throttle`) │
│    • Empirical Roofline placement and memory bus saturation percentage │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Optimization Roadmap: Structural Impact on Profiling Metrics

Understanding these physical bottlenecks provides a roadmap for modern LLM inference systems. Each technique targets a specific component of the three physical metrics:

| Optimization | Primary Bottleneck Addressed | Physical Mechanism | Metric Transformation |
|---|---|---|---|
| **Kernel Fusion**<br>*(Fused QKV, SwiGLU)* | VRAM Data Wait & Launch Gaps | Keeps intermediate activations in registers/SRAM; reduces kernel launches ($844 \rightarrow <150$) | Duty cycle increases from $26.6\% \rightarrow 70\%+$; inter-kernel gaps shrink |
| **CUDA Graphs**<br>*(`graph.replay()`)* | CPU Launch Gaps | Captures the execution graph once; replays 800+ kernels in a single driver call | CPU idle gap drops from $41.8\text{ ms} \rightarrow <0.5\text{ ms}$; duty cycle approaches $95\%+$ |
| **Weight Quantization**<br>*(FP8 / INT4 AWQ)* | VRAM Data Wait | Halves (FP8: 1.23 GB) or quarters (INT4: 0.61 GB) weights transferred over bus | VRAM wait drops from $15.1\text{ ms} \rightarrow 7.5\text{ ms}$ (FP8) or $3.8\text{ ms}$ (INT4) |
| **Continuous Batching**<br>*($B > 1$)* | Low Arithmetic Intensity | Reuses weights across $B$ tokens ($y = W X_B$), turning GEMV into GEMM | Arithmetic intensity scales as $B \times 1.0\text{ FLOP/Byte}$; MFU scales from $0.2\% \rightarrow 30\%+$ |

---

## 7. Summary & Key Formulas Reference

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CORE SYSTEMS CHEAT SHEET                                     │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Break-Even Arithmetic Intensity:                                                            │
│    I_ridge = Peak_Compute / Peak_Bandwidth = 120 TFLOP/s / 300 GB/s = 400 FLOPs/Byte           │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Single-Token Arithmetic Intensity (GEMV):                                                   │
│    I_decode = 2 FLOPs / 2 Bytes = 1.0 FLOP/Byte  (Severe Memory Bandwidth Bottleneck)          │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Minimum Wire Streaming Time:                                                                │
│    T_min_wire = Model_Weights / VRAM_Bandwidth = 2.46 GB / 300 GB/s = 8.20 ms                  │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Three Physical Metrics:                                                                     │
│    Latency_Token = T_Host_Gaps (41.8 ms) + T_VRAM_Wait (15.1 ms) + T_Math_Compute (0.02 ms)    │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. Active GPU Duty Cycle:                                                                      │
│    Duty_Cycle = T_Active_Kernels / T_Wall_Clock = 15.12 ms / 56.93 ms = 26.6%                  │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```
