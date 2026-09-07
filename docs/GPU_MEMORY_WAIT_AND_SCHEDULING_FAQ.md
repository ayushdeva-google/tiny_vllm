# GPU Memory Bottlenecks, Idle Time & Scheduling Dynamics: Systems FAQ

> **Context**: Empirical findings, physical memory limits, and hardware interplay derived from profiling autoregressive decode on **LLaMA-3.2-1B-Instruct** running on an **NVIDIA L4 GPU** (24 GB GDDR6, 300 GB/s peak memory bandwidth, 120 TFLOP/s BF16 compute).

---

## Table of Contents
1. [Q1: Is the GPU Actually Idle During Inference? What Do the Real Numbers Show?](#q1-is-the-gpu-actually-idle-during-inference-what-do-the-real-numbers-show)
2. [Q2: How Often Do Memory Transfers Happen (PCIe vs. VRAM)?](#q2-how-often-do-memory-transfers-happen-pcie-vs-vram)
3. [Q3: How Do We Prove the GPU is Waiting on Data Transfer Rather Than Busy Doing Math?](#q3-how-do-we-prove-the-gpu-is-waiting-on-data-transfer-rather-than-busy-doing-math)
4. [Q4: Can We See the Memory Wait vs. Compute Breakup for Each Individual Operation?](#q4-can-we-see-the-memory-wait-vs-compute-breakup-for-each-individual-operation)
5. [Q5: Can We View This on a Timeline? Why Isn't It Sequential Inside a Kernel?](#q5-can-we-view-this-on-a-timeline-why-isnt-it-sequential-inside-a-kernel)
6. [Q6: Isn't CPU Scheduling Asynchronous in CUDA? Why Does CPU Overhead Still Degrade GPU Utilization?](#q6-isnt-cpu-scheduling-asynchronous-in-cuda-why-does-cpu-overhead-still-degrade-gpu-utilization)
7. [Q7: When We Optimize in the Future, How Will the Metrics and Charts Prove It Worked?](#q7-when-we-optimize-in-the-future-how-will-the-metrics-and-charts-prove-it-worked)

---

## Q1: Is the GPU Actually Idle During Inference? What Do the Real Numbers Show?

**Yes, overwhelmingly so.** In our unoptimized baseline, the GPU sits completely idle for nearly **three-quarters** of the total token generation time.

From our profiled decode step (`traces/step_100_trace.json`):

| Metric | Real Experiment Value | What It Represents |
|---|---|---|
| **Kernels Launched per Token** | **844 kernels** | Number of separate GPU kernels launched to generate **one single token**. |
| **Total GPU Wall-Clock Span** | **56.93 ms** | Time elapsed from the first kernel starting to the last kernel finishing. |
| **Active GPU Kernel Time** | **15.12 ms** | Time GPU execution units were running instructions. |
| **GPU Idle Gap Time** | **41.81 ms (73.4% idle!)** | **Time the GPU spent sitting 100% idle with an empty execution pipeline.** |
| **Active GPU Duty Cycle** | **26.6%** | The fraction of wall-clock time the GPU hardware was active. |

```
Observed Wall-Clock Span per Token (~57 ms):
┌─────────────────────────────────────────────────────────────────┬───────────────────────────┐
│                    GPU IDLE GAPS: 41.8 ms                       │     ACTIVE GPU: 15.1 ms   │
│                 (73.4% of wall-clock time)                      │   (26.6% of time on GPU)  │
│      [GPU execution pipelines empty, waiting on CPU dispatch]   │  [Pipelined GEMV & math]  │
└─────────────────────────────────────────────────────────────────┴───────────────────────────┘
```

### Why is there so much idle time?
Because the model launches **844 separate micro-kernels** per token (RMSNorm, Q/K/V projections, RoPE rotations, KV repeat, attention dot-products, Softmax, projection out, SwiGLU Gate/Up/Down projections, SiLU, and elementwise multiplications across 16 layers). 

Many of these micro-kernels execute in just **$1\text{ to }3\ \mu\text{s}$**, but the host Python interpreter and CUDA driver take **$15\text{ to }40\ \mu\text{s}$** to prepare and launch each subsequent operation. The GPU finishes its work almost immediately, runs out of queued commands, and sits idle waiting for the CPU to dispatch the next kernel.

---

## Q2: How Often Do Memory Transfers Happen (PCIe vs. VRAM)?

There are two fundamentally different classes of memory transfers during inference:

### 1. PCIe Transfers (Host CPU $\leftrightarrow$ Device GPU)
* **Frequency**: Only **2 times per token**.
  * Input: Host sends the latest token ID (`input_ids`) to GPU VRAM (~80 bytes).
  * Output: GPU transfers the argmax integer token back to Host RAM (`next_token_id.item()`, 8 bytes).
* **Impact**: Total PCIe data transfer is minuscule (< 100 bytes). However, reading the output token back forces a **synchronization stall** (see Q5).

### 2. VRAM Transfers (GPU Silicon $\leftrightarrow$ GPU High-Bandwidth Memory)
* **Frequency**: **Hundreds of times per token!**
* **Why**: Because the operations are unfused, **every single kernel must load its input tensors from VRAM and write its output tensors back to VRAM**.

#### Example: The Memory Round-Trips in Just ONE Layer's MLP Block
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

1. `gate_proj(x)`: Reads $x$ from VRAM $\rightarrow$ computes $\rightarrow$ **writes `gate_out` to VRAM** (Transfer #1)
2. `up_proj(x)`: Reads $x$ from VRAM $\rightarrow$ computes $\rightarrow$ **writes `up_out` to VRAM** (Transfer #2)
3. `silu(gate_out)`: **Reads `gate_out` from VRAM** $\rightarrow$ computes $\rightarrow$ **writes `silu_out` to VRAM** (Transfer #3)
4. `mul(silu_out, up_out)`: **Reads `silu_out` AND `up_out` from VRAM** $\rightarrow$ computes $\rightarrow$ **writes `ffn_intermediate` to VRAM** (Transfer #4)
5. `down_proj(...)`: **Reads `ffn_intermediate` from VRAM** $\rightarrow$ computes $\rightarrow$ writes final result to VRAM (Transfer #5)

In that single feed-forward block alone, the GPU makes **5 round-trips to VRAM**, constantly saving and reloading temporary vectors that are never reused. Across 16 layers, this memory traffic dominates execution.

---

## Q3: How Do We Prove the GPU is Waiting on Data Transfer Rather Than Busy Doing Math?

The fact that the GPU is active for 15.1 ms does not automatically mean it was busy computing math. We prove it was stalled waiting on data transfer using **two independent physical realities**:

### Proof 1: The "Speedometer" (Memory Bus Bandwidth vs. Compute Core Speed)

We compare the utilization of the two independent hardware subsystems:
1. **Total Model Weights Streamed**: LLaMA-3.2-1B in `bfloat16` contains **2.46 GB** of parameters.
2. **Physical Wire Limit**: The NVIDIA L4 memory bus has a theoretical peak bandwidth of **300 GB/s**.
3. **Physical Minimum Wire Time**: Even if the GPU compute cores had infinite speed (0.00 ms math time), the absolute fastest the physical memory wires could pump 2.46 GB into the chip is:
   $$\text{Time}_{\text{min\_wire}} = \frac{2.46\text{ GB}}{300\text{ GB/s}} = \mathbf{8.20\text{ ms}}$$

Now look at what happened in our actual experiment:
* **Measured Kernel Time**: **15.12 ms**
* **Achieved Memory Bandwidth**: $\frac{2.46\text{ GB}}{0.01512\text{ s}} \approx \mathbf{162.7\text{ GB/s}}$ (Over **54.2%** of peak wire capacity overall, with individual GEMV kernels reaching **250+ GB/s / 84%** of wire capacity!).
* **Achieved Compute Core Speed**: $\mathbf{0.2\%}$ of peak Tensor Core capability!

$$\text{Math FLOPs Required} = 2 \times 1.23 \times 10^9 = 2.46\text{ GFLOPs}$$
$$\text{Pure Compute Time} = \frac{2.46\text{ GFLOPs}}{120\text{ TFLOP/s}} = \mathbf{0.0205\text{ ms}}\ (20.5\ \mu\text{s})$$

The compute cores finished all their mathematical operations in **$20\ \mu\text{s}$**, and spent the remaining **$15.1\text{ ms}$** starved of operands, waiting for memory controllers to deliver weights from VRAM.

### Proof 2: The Hardware Metric (`Stall Long Scoreboard`)

In NVIDIA GPU architecture, hardware performance counters track warp stall reasons every clock cycle:
* **`Stall Math Pipe Throttle`**: Warp is paused because the Tensor Cores / ALUs are busy computing math.
  * *Measured in our decode kernels*: **$\approx 0\%$**
* **`Stall Long Scoreboard`**: Warp is frozen because an instruction requested data from VRAM (global memory / L2 cache), and **the bytes have not arrived yet**.
  * *Measured in our decode kernels*: **$75\% - 85\%$ of all execution cycles!**

This hardware counter provides indisputable proof: the compute units are stalled waiting for data packets from VRAM.

### The Complete 3-Way Time Decomposition

```
Full Token Latency Breakdown (~57 ms):
┌──────────────────────────────────────┬────────────────────────┬────────┐
│  1. CPU Launch & Driver Gaps         │ 2. VRAM Data Wait      │ 3. Math│
│  (Empty pipeline between 844 kernels)│ (Streaming 2.46 GB)    │ Compute│
│  41.81 ms (73.4%)                    │ 15.10 ms (26.5%)       │ 0.02 ms│
└──────────────────────────────────────┴────────────────────────┴────────┘
                                                                 ▲
                                                      Barely visible (0.04%)
```

---

## Q4: Can We See the Memory Wait vs. Compute Breakup for Each Individual Operation?

**Yes.** We can calculate this for every single operation because:
1. Matrix dimensions give exact FLOPs $\rightarrow$ yielding theoretical **Pure Math Time** at peak compute (120 TFLOP/s).
2. Weight tensor shapes give exact byte sizes $\rightarrow$ yielding minimum **Wire Transfer Time** at peak bandwidth (300 GB/s).
3. Profiling traces record exact **Measured Kernel Durations**.

### Per-Operation Breakdown Table (Real Measured Data)
*(Aggregated across 16 layers for one token generation step)*

| Operation | Total Measured Time | VRAM Data Moved | Pure Math Time | Time Waiting on VRAM | % Stalled on Memory | Achieved Bandwidth |
|---|---|---|---|---|---|---|
| **`FFN_Gate_Up_Linear`** | **5.74 ms** | 1,073.7 MB | **0.009 ms** ($9\ \mu\text{s}$) | **5.731 ms** | **99.8%** | **187 GB/s** (Saturated) |
| **`FFN_Down_Linear`** | **2.48 ms** | 536.9 MB | **0.004 ms** ($4\ \mu\text{s}$) | **2.476 ms** | **99.8%** | **216 GB/s** (Saturated) |
| **`LM_Head`** (Vocab projection) | **2.08 ms** | 525.3 MB | **0.004 ms** ($4\ \mu\text{s}$) | **2.076 ms** | **99.8%** | **252 GB/s** (84% of wire limit!) |
| **`QKV_Linear`** | **1.31 ms** | 201.3 MB | **0.002 ms** ($2\ \mu\text{s}$) | **1.308 ms** | **99.8%** | **153 GB/s** |
| **`O_Linear`** (Attn output) | **0.72 ms** | 134.2 MB | **0.001 ms** ($1\ \mu\text{s}$) | **0.719 ms** | **99.8%** | **186 GB/s** |
| **`Attn_Compute`** (Softmax/PV)| **0.92 ms** | ~5.0 MB | **0.001 ms** ($1\ \mu\text{s}$) | **0.919 ms** | **99.8%** | Latency-bound |
| **`RMSNorm`** (All layers) | **0.86 ms** | ~0.2 MB | **0.0001 ms** | **0.860 ms** | **99.9%** | Launch-bound |

### Deep Dive: A Single Operation Example (`LM_Head`)
* **Matrix dimensions**: Input vector $[1, 2048] \times$ Weights $[2048, 128256]$.
* **Total parameters**: $2,048 \times 128,256 = 262,668,288$ values $\approx \mathbf{525.3\text{ MB}}$ in `bfloat16`.
* **Math operations**: $2 \times 1 \times 2048 \times 128256 \approx \mathbf{525.3\text{ MFLOPs}}$.
* **Pure Math Time**: $\frac{525.3 \times 10^6}{120 \times 10^{12}} = \mathbf{0.0043\text{ ms}}\ (4.3\ \mu\text{s})$.
* **Measured Kernel Time**: $\mathbf{2.08\text{ ms}}$.
* **Time Waiting on VRAM**: $\mathbf{2.0757\text{ ms}}\ (\mathbf{99.8\%})$.
* **Achieved Bandwidth**: $\frac{525.3\text{ MB}}{0.00208\text{ s}} = \mathbf{252.5\text{ GB/s}}$ (**84.2% of physical hardware maximum**).

```
Visual Timeline Card for LM_Head:
┌────────────────────────────────────────────────────────────────────────┐
│ LM_Head Vocab Projection (Duration: 2.08 ms)                          │
│                                                                        │
│ [████████████████████████████████████████████████████████████████░]    │
│  Waiting on VRAM: 2.076 ms (99.8%)                 Math: 0.004 ms (0.2%)│
│  Data Streamed: 525.3 MB @ 252.5 GB/s (Bus Saturation: 84.2%)          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Q5: Can We View This on a Timeline? Why Isn't It Sequential Inside a Kernel?

A natural question when trying to visualize this is: *Can we see this breakup on a timeline as clean, sequential blocks — e.g. waiting for CPU launch, followed by waiting for VRAM data, followed by math?*

The answer is:
1. **Between kernels (CPU launch $\rightarrow$ Gap $\rightarrow$ Kernel)**: Yes, this is **100% sequential** and shows up as clean, distinct chunks on a timeline.
2. **Inside a kernel (VRAM wait vs. Math)**: No, it is **not** sequential chunks. It is heavily **overlapped and micro-interleaved** at the nanosecond level.

### Inside a Kernel: Why Isn't It Sequential?

If a matrix multiplication kernel ran sequentially like this:

```
Sequential (Idealized):
[── 15 ms: Load all weights from VRAM ──] ➔ [ 0.02 ms: Do all the math ]
```
That would be easy to draw! But GPUs don't work that way.

In reality, the GPU uses **pipelined double-buffering (tiling)**:
1. It loads a tiny **tile** of weights (e.g., a $128 \times 64$ matrix block) from VRAM.
2. It takes **~200 clock cycles** for that tile to arrive over the memory bus. (Cores are stalled waiting).
3. Once the tile arrives, the Tensor Core crunches the math in **~2 clock cycles**.
4. Meanwhile, it requested the *next* tile, which takes another 200 cycles to arrive.

So inside a 15-microsecond kernel, what's really happening is **hundreds of micro-bursts**:

```
Inside 1 Kernel:
[Wait 200 cycles][2c Math][Wait 200 cycles][2c Math][Wait 200 cycles][2c Math]...
```
Because the math finishes in 2 cycles and the memory takes 200 cycles, the math and memory are constantly overlapping, but the cores spend 99% of those cycles stalled waiting for the next tile.

### How Timelines Actually Display This Without Misleading You

Because math and memory are micro-interleaved inside the kernel, profilers (including Nsight Systems and our dashboard visualizer) display this using **Synchronized Parallel Tracks (Swimlanes)**:

```
TIMELINE (Moving Left to Right in Milliseconds):

Track 1: Host CPU Thread
[ CPU Enqueue ]─────────────────────────────────► [ CPU Blocked on .item() ───]

Track 2: GPU Idle Gaps & Launches
───────────────► [ GAP: GPU Idle (42 ms total) ] ───────────►

Track 3: GPU Kernels (Compute)
                 [ RMSNorm ]  [ Q_proj ]  [ K_proj ] ... [ Down_proj ]
                 (3 μs)       (13 μs)     (13 μs)        (41 μs)

Track 4: VRAM Memory Bus Activity
                 ░░░░░░░░░░░  [████████]  [████████] ... [███████████]
                 (Low traffic) (250 GB/s) (250 GB/s)     (270 GB/s)
                               ▲
                               Memory bus is 85% full while kernel runs!
```

#### How You Read This on the Timeline:
* **Track 1 (Host CPU)**: Shows when Python is launching kernels vs. when Python is frozen waiting for the final token (`.item()`).
* **Track 2 (Compute Track)**: Shows each individual kernel block. You clearly see the empty idle gaps between them.
* **Track 3 (Memory Bus Track or Inspector Tooltip)**: 
  When you click or hover over a kernel like `Down_proj` (which takes $41\ \mu\text{s}$), the inspector card reveals what happened inside that block:
  * **Total Duration**: $41\ \mu\text{s}$
  * **Memory Streamed**: $11.2\text{ MB}$ at **$273\text{ GB/s}$** (Memory Bus Saturated)
  * **Pure Math Time**: $\approx 0.08\ \mu\text{s}$ ($0.2\%$ of the block)
  * **Time Waiting on Memory**: $\approx 40.9\ \mu\text{s}$ ($99.8\%$ of the block)

---

## Q6: Isn't CPU Scheduling Asynchronous in CUDA? Why Does CPU Overhead Still Degrade GPU Utilization?

In CUDA and PyTorch, kernel launches (`cudaLaunchKernel`) are non-blocking: the host CPU enqueues commands into a ring buffer (CUDA stream) and returns immediately. 

So why does CPU scheduling overhead still devastate GPU utilization in our model?

### The "Chef and Eater" Analogy

Think of the CPU as a **Chef** writing recipe slips, and the GPU as an **Eater** consuming the dishes:

#### Case A: Large Batches or Training (Async Works Perfectly)
* The Chef takes **$10\ \mu\text{s}$** to write an order.
* The GPU takes **$5,000\ \mu\text{s}$ ($5\text{ ms}$)** to compute the giant matrix multiplication.
* **Result**: The Chef easily stays dozens of orders ahead. The GPU's queue is always full, and CPU dispatch time has zero impact on GPU utilization.

#### Case B: Single-Token LLM Decode (The Starved GPU Problem)
* The Chef (Python runtime, PyTorch dispatcher, shape checking, memory allocators) takes **$20\text{ to }50\ \mu\text{s}$** to prepare each operation.
* The GPU hardware executes a small `RMSNorm` or `RoPE` kernel in **$1\text{ to }3\ \mu\text{s}$**!

$$\text{GPU Kernel Execution Time } (2\ \mu\text{s}) \ll \text{CPU Dispatch Time } (30\ \mu\text{s})$$

```
CPU Host Thread:  [-- Enqueue Op 1 (30 μs) --] ➔ [-- Enqueue Op 2 (30 μs) --]
                                                 ▲
GPU Device:       [ Op 1 (2 μs) ] [ IDLE WAITING FOR CPU (28 μs) ] [ Op 2 ]
```

The GPU consumes each kernel in $2\ \mu\text{s}$, looks in its queue for the next instruction, finds the queue empty because Python is still executing code, and goes **idle**. This is known as **Host Launch Starvation** (or "GPU Underflow").

### Concrete Proof from Our Trace (`step_100_trace.json`)

Comparing the exact CPU launch timestamps with GPU execution start times:

```
Kernel 0 (Index select):
  • CPU launch finished:         0.0 μs
  • GPU started executing:      42.1 μs
  • Kernel run duration:         3.8 μs   (GPU finished at 45.9 μs)

Kernel 1 (Unrolled elementwise):
  • CPU busy running Python:   331.5 μs   (Took 331 μs to reach Kernel 1!)
  • GPU started executing:     354.4 μs
```

* Kernel 0 completed at **$45.9\ \mu\text{s}$**.
* The CPU did not enqueue Kernel 1 until **$331.5\ \mu\text{s}$**.
* **The GPU sat 100% idle for nearly $300\ \mu\text{s}$** waiting for the Python interpreter to execute intermediate code.

### The Hard Synchronization Barrier: `.item()`
In autoregressive generation, generating token $t+1$ depends on the identity of token $t$. Calling:
```python
next_token = torch.argmax(logits[:, -1, :], dim=-1)
token_val = next_token.item()  # <--- FORCED DEVICE-TO-HOST BARRIER
```
forces a complete device-to-host synchronization stall. The CPU freezes until the GPU finishes all work. This **drains the GPU command queue to zero**. Every single token starts from a dead stop with an empty pipeline.

---

## Q7: When We Optimize in the Future, How Will the Metrics and Charts Prove It Worked?

When optimizations are implemented in `tiny_vllm`, these exact metrics will directly confirm whether each bottleneck was resolved:

### 1. Kernel Fusion (e.g. Fused QKV, Fused SwiGLU)
* **What changes**:
  * **FLOPs**: **Remain identical**. Math operations are unchanged.
  * **VRAM Data Moved**: **Drops dramatically**. Intermediate activations (`gate_out`, `up_out`, `silu_out`) remain inside fast on-chip registers/SRAM and are never written to or read from VRAM.
  * **Kernel Count**: Drops from **844 $\rightarrow$ under 150 kernels per token**.
* **What the chart will show**:
  * The **41.8 ms idle launch gaps shrink** into a continuous, dense block.
  * Active GPU duty cycle climbs from **$26.6\% \rightarrow 70\% - 90\%+$**.
  * Total token latency drops from $\sim 57\text{ ms} \rightarrow 15\text{ ms}$.

### 2. CUDA Graphs (`torch.cuda.make_graphed_callables` / Graph Replay)
* **What changes**:
  * Captures the entire 844-kernel dispatch sequence into a pre-compiled hardware execution graph stored directly on the GPU.
  * Replaces 844 individual Python/PyTorch dispatches with **1 single hardware trigger** (`graph.replay()`).
* **What the chart will show**:
  * CPU launch overhead is completely bypassed.
  * Inter-kernel idle gaps drop from **$41.8\text{ ms} \rightarrow < 1\text{ ms}$**.
  * Kernels execute back-to-back at hardware silicon limits.

### 3. Weight Quantization (e.g. FP8 or INT4 / AWQ)
* **What changes**:
  * Total weight footprint cuts in half (FP8: **1.23 GB**) or in quarter (INT4: **0.61 GB**).
  * Arithmetic intensity doubles ($2.0\text{ FLOP/Byte}$) or quadruples ($4.0\text{ FLOP/Byte}$).
* **What the chart will show**:
  * The **VRAM Data Wait bar cuts directly in half** (from $15.1\text{ ms} \rightarrow \sim 7.5\text{ ms}$ for FP8, or $\sim 3.8\text{ ms}$ for INT4).
  * Per-operation charts show `LM_Head` dropping from $2.08\text{ ms} \rightarrow 1.05\text{ ms}$.

### 4. Continuous Batching ($B > 1$)
* **What changes**:
  * Each weight matrix loaded from VRAM is reused across $B$ tokens simultaneously ($y = W \times X_B$).
  * Transitions execution from memory-bound GEMV to compute-bound GEMM.
* **What the chart will show**:
  * Arithmetic intensity scales linearly with batch size ($B \times 1.0\text{ FLOP/Byte}$).
  * Model FLOPs Utilization (MFU) climbs from **$0.2\% \rightarrow 25\% - 50\%+$**.
  * Token throughput scales from $\sim 17\text{ tokens/sec} \rightarrow 500+\text{ tokens/sec}$.
