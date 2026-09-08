# Unoptimized Attention: The Quadratic Scaling Paradox & Memory Physics

> **Context**: Hardware analysis, arithmetic intensity derivation, and memory hierarchy mechanics explaining why attention latency grows quadratically during unoptimized autoregressive decoding on **LLaMA-3.2-1B** and the **NVIDIA L4 GPU** (24 GB GDDR6, 300 GB/s memory bandwidth, 24 MB L2 Cache, 120 TFLOP/s BF16 Tensor Cores).

---

## Executive Summary: Resolving the Core Paradox

During autoregressive profiling in **Chapter 1** ([`chapter_1/llama_inference_with_profiling.py`](../chapter_1/llama_inference_with_profiling.py)), we observe that the latency of `Attn_Compute` explodes quadratically from **0.48 ms at step 0** to over **312 ms at step 2,048**.

This raises a fundamental systems paradox:
> *"If kernel execution time is bounded by VRAM read/write latency rather than pure math compute (with compute taking < 1% of time), and sequence length $S$ grows linearly token-by-token, why does the attention latency scale quadratically ($O(S^2)$) instead of linearly ($O(S)$)?"*

### The Resolution:
1. **The premise is correct**: Pure math execution is indeed **< 1% of total latency**. Attention is overwhelmingly **memory-bandwidth bound**.
2. **The catch**: In naive autoregressive generation without a KV cache, **the volume of data read and written to VRAM is itself strictly quadratic ($O(S^2)$)**. 
3. **The culprit**: At step $S$, the model recomputes attention across all past tokens from scratch. The intermediate attention matrix has shape **$(32\ \text{heads}, S, S)$**. Every single element-wise operation (`mask`, `softmax`, `dropout`, `matmul`) flushes this $S \times S$ matrix to VRAM DRAM and reads it back, moving tens of gigabytes across the memory bus per step.
4. **The L2 Cache Cliff**: Once $S > 500$, the $S \times S$ attention tensor exceeds the **24 MB on-chip L2 cache**, spilling entirely into the 300 GB/s GDDR6 DRAM and causing the latency curve to steepen even further.

---

## 1. Math vs. Memory: The Roofline Reality of Attention

Let us analyze the arithmetic intensity of scaled dot-product attention for **LLaMA-3.2-1B** ($H = 32\ \text{heads}$, $D = 64\ \text{head dimension}$, $S = \text{sequence length}$):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{D}} + M\right) V$$

### 1.1 Pure Math Compute (FLOPs)
At sequence length $S = 2,048$:
- **Matrix Multiply 1 ($Q \times K^T$)**: 
  - $Q \in \mathbb{R}^{S \times D}$, $K^T \in \mathbb{R}^{D \times S} \implies 2 \times S \times S \times D\ \text{FLOPs}$ per head.
  - For 32 heads: $32 \times 2 \times (2048)^2 \times 64 \approx \mathbf{1.72 \times 10^{10}\ \text{FLOPs}}\ (17.2\ \text{GFLOPs})$.
- **Matrix Multiply 2 ($P \times V$)**:
  - $P \in \mathbb{R}^{S \times S}$, $V \in \mathbb{R}^{S \times D} \implies 2 \times S \times S \times D\ \text{FLOPs}$ per head.
  - For 32 heads: $\mathbf{17.2\ \text{GFLOPs}}$.
- **Softmax & Scaling**:
  - $\sim 5\ \text{FLOPs}$ per element $\times\ 32 \times (2048)^2 \approx \mathbf{0.67\ \text{GFLOPs}}$.
- **Total Math across all 16 Layers**:
  $$\text{Total FLOPs} = 16 \times (17.2 + 17.2 + 0.67)\ \text{GFLOPs} \approx \mathbf{561\ \text{GFLOPs}}$$

### 1.2 Theoretical Execution Time on NVIDIA L4
The NVIDIA L4 provides **120 TFLOP/s** of BF16 Tensor Core compute:
$$T_{\text{pure\_math}} = \frac{561 \times 10^9\ \text{FLOPs}}{120 \times 10^{12}\ \text{FLOP/s}} = \mathbf{0.00468\ \text{seconds}} = \mathbf{4.68\ \text{ms}}$$

If the GPU were solely compute-bound, computing attention across all 16 layers at sequence length 2,048 would take **less than 5 ms**. Yet, our profiler measures **over 312 ms**. 

Where does the remaining **98.5% of wall-clock time** go? It is spent stalling on **VRAM data transfers**.

---

## 2. Why VRAM Read/Write Scales Quadratically ($O(S^2)$)

In naive autoregressive decoding ([`chapter_1/llama_inference.py`](../chapter_1/llama_inference.py)), **there is no KV-cache**. On every new token generated, the model re-feeds the entire prompt and previous generations as a sequence of length $S$:

```python
# chapter_1/llama_inference.py: Attention.forward()
bsz, seqlen, _ = x.shape  # seqlen = S

# 1. Project Q, K, V for ALL S tokens
xq = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)    # (1, S, 32, 64)
xk = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim) # (1, S, 8, 64)
xv = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim) # (1, S, 8, 64)

# Repeat KV for GQA and transpose
xq = xq.transpose(1, 2)  # (1, 32, S, 64)
xk = xk.transpose(1, 2)  # (1, 32, S, 64)
xv = xv.transpose(1, 2)  # (1, 32, S, 64)

# 2. Scaled Dot-Product Attention: The (S x S) Materialization
scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (1, 32, S, S) <--- O(S^2)!

# 3. Causal Masking
if seqlen > 1:
    mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), ...), diagonal=1)
    scores = scores + mask  # Reads (32, S, S), writes (32, S, S) <--- O(S^2)!

# 4. Softmax
probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)          # Reads (32, S, S), writes (32, S, S) <--- O(S^2)!

# 5. Output Projection
output = torch.matmul(probs, xv)  # Reads (32, S, S) and xv, writes (32, S, 64)
```

### 2.1 The Unfused Kernel Pipeline
Because standard PyTorch executes these operations via separate PyTorch kernels, each kernel cannot retain the full $S \times S$ matrix in registers or SRAM. It **must write intermediate results to global VRAM DRAM and read them back in the next kernel**:

```
[Kernel 1: BMM Q*K^T] ──(Write S x S)──► VRAM DRAM ──(Read S x S)──► [Kernel 2: Add Mask]
                                                                             │
                                                                       (Write S x S)
                                                                             ▼
[Kernel 4: BMM P*V]   ◄──(Read S x S)─── VRAM DRAM ◄──(Write S x S)── [Kernel 3: Softmax]
```

### 2.2 Concrete Numbers: VRAM Traffic Growth
Every element in `scores` is 2 bytes (BF16), and PyTorch evaluates softmax in 4 bytes (FP32) for numerical stability.

$$\text{Attention Matrix Elements per Layer} = 32\ \text{heads} \times S^2$$

| Step ($S$) | Single Matrix Size | Softmax FP32 Tensor | VRAM Reads + Writes per Step (16 Layers) | Min DRAM Transfer Time (@ 300 GB/s) |
|---|---|---|---|---|
| **$S = 250$** | **4.0 MB** | 8.0 MB | **0.25 GB** | **0.83 ms** |
| **$S = 500$** | **16.0 MB** | 32.0 MB | **1.02 GB** | **3.40 ms** |
| **$S = 1,000$** | **64.0 MB** | 128.0 MB | **4.10 GB** | **13.65 ms** |
| **$S = 1,500$** | **144.0 MB** | 288.0 MB | **9.22 GB** | **30.72 ms** |
| **$S = 2,048$** | **268.4 MB** | 536.8 MB | **17.18 GB** | **57.26 ms** |

Notice that moving from $S = 250$ to $S = 2,048$ is an **$8.2\times$ increase in sequence length**, but results in a **$67.1\times$ increase in VRAM bytes transferred**.

The VRAM read and write volume itself is **strictly quadratic ($O(S^2)$)**.

---

## 3. The 24 MB L2-Cache Cliff: Why the Curve Steepens

If you inspect the context scaling curve in the interactive dashboard ([`profile_dashboard.html`](../chapter_1/profile_results/profile_dashboard.html)), the latency does not just increase as a smooth parabola—it exhibits an **aggressive inflection point around $S = 500 \rightarrow 750$**.

This inflection is driven by the physical memory hierarchy of the NVIDIA L4 GPU:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   NVIDIA L4 ON-CHIP SILICON HIERARCHY                    │
│                                                                          │
│   Registers & SM Shared Memory (SRAM)      Bandwidth: ~20,000 GB/s       │
│                  ▲                                                       │
│                  │ (Per-SM tile transfers)                               │
│                  ▼                                                       │
│   Shared L2 Cache (24 MB)                  Bandwidth: ~2,500 GB/s        │
│                  ▲                                                       │
│ ═════════════════╪══════════════════════════════════════════════════════ │
│                  │  L2 Cache Eviction / Miss Barrier                     │
│ ═════════════════╪══════════════════════════════════════════════════════ │
│                  ▼                                                       │
│   Off-Chip GDDR6 VRAM (24 GB)              Bandwidth: 300 GB/s (12x drop)│
└──────────────────────────────────────────────────────────────────────────┘
```

### The Mechanism of the Cliff:
1. **$S \le 500$ (In-Cache Regime)**:
   - The $(32, S, S)$ matrix size is **16.0 MB**, which fits inside the **24 MB L2 Cache**.
   - Kernel 1 writes into L2; Kernel 2 reads directly from L2 at **~2,500 GB/s**.
   - GDDR6 memory controllers are not saturated; latency is low.

2. **$S \ge 1,000$ (Out-of-Cache DRAM Eviction Regime)**:
   - The attention tensor requires **64 MB to 268 MB** (and up to **536 MB** for FP32 Softmax).
   - This completely blows past the 24 MB L2 capacity.
   - The L2 cache experiences continuous capacity misses and cacheline thrashing.
   - Every read and write is forced across the narrow **300 GB/s GDDR6 bus**.
   - Memory bandwidth drops by **$\sim 8.3\times$** (from 2,500 GB/s L2 bandwidth to 300 GB/s DRAM bandwidth), causing the observed latency to spike.

---

## 4. Why Linear Layers Stay Flat While Attention Explodes

A striking insight from our profiling breakdown is comparing `Attn_Compute` with the linear projection layers (`Q_Linear`, `LM_Head`, `FFN_Gate_Up_Linear`):

```
Latency (ms)
  350 ┬                                                      / Attn_Compute: O(S^2)
      │                                                     /
  300 ┼                                                    /
  250 ┼                                                   /
  200 ┼                                                  /
  150 ┼                                                 /
  100 ┼                                                /
   50 ┼── Flat Weight Streaming: O(1) ───────────────/
    0 ┴──────────▲──────────────────────────────────▲─────────► Sequence Step (S)
               Step 250                           Step 2048
```

| Layer Type | What Dominates VRAM Traffic? | Traffic Scaling with $S$ | Latency Behavior |
|---|---|---|---|
| **Linear Layers (`FFN`, `Linear`)** | Loading the static **2.46 GB model weights** from VRAM into registers. Activations ($1 \times S \times 2048$) are small relative to weights. | **$O(1)$** (constant ~2.46 GB weight stream per token) | **Flat** (~20–40 ms total across all linear layers) |
| **Attention Compute (`Attn_Compute`)** | Loading and writing the **intermediate attention matrix ($S \times S$)** between unfused kernels. Zero weights loaded! | **$O(S^2)$** (from 4 MB at step 250 to 268 MB at step 2,048) | **Explosive Quadratic** (0.48 ms $\rightarrow$ 312 ms) |

In linear layers, the weight traffic is massive (2.46 GB) but invariant to sequence length. In attention, there are **no model weights**; the entire cost is moving ephemeral activations whose volume grows quadratically with $S$.

---

## 5. Architectural Preview: How Optimizations Fix This

Understanding this physical breakdown shows exactly why future optimizations work:

### Optimization 1: KV-Cache (Chapter 2)
* **What it changes**: Instead of passing $Q \in \mathbb{R}^{S \times D}$, we only pass the **single new query token** $q \in \mathbb{R}^{1 \times D}$. Past keys and values are retrieved from a pre-allocated cache ($K_{\text{cache}} \in \mathbb{R}^{S \times D}$).
* **Impact on Tensor Shape**:
  $$q \cdot K_{\text{cache}}^T \implies (1 \times D) \times (D \times S) = \mathbf{(1 \times S)}$$
* The attention matrix shrinks from an **$S \times S$ matrix** down to a **$1 \times S$ vector**!
* **Result**: VRAM read traffic for keys scales **linearly ($O(S)$)**, and the $1 \times S$ vector easily fits in on-chip SRAM/L2 cache, completely eliminating the $O(S^2)$ memory explosion during decode.

### Optimization 2: Kernel Fusion & FlashAttention
* **What it changes**: Fuses $Q K^T$, masking, softmax, and $P V$ into a **single CUDA kernel**.
* **Impact on VRAM Traffic**: Tiled blocks of $Q$ and $K$ are brought into fast SRAM (~20 TB/s). Softmax is computed online using the Running Softmax algorithm.
* The large intermediate matrix ($S \times S$ or $1 \times S$) is **never written to off-chip VRAM DRAM**.
* **Result**: Completely eliminates round-trips 1, 2, and 3 to VRAM, reducing DRAM bandwidth traffic by **$4\times - 8\times$**.

---

## Summary Cheat Sheet

| Question | Physical Reality |
|---|---|
| **Is Attention compute-bound or memory-bound?** | **Overwhelmingly memory-bound**. Pure math takes < 1.5% of total time; > 98.5% is spent waiting for GDDR6 memory controllers. |
| **Why did I think VRAM traffic scales linearly?** | You were thinking of the sequence length $S$ or of a KV-cached decode (where $Q$ is $1 \times D$ and $K$ is $S \times D \implies 1 \times S$). |
| **Why does our Chapter 1 code scale quadratically?** | Without a KV cache, $Q$ is $S \times D$ and $K$ is $S \times D$. Their product is an **$S \times S$ matrix**. Across 32 heads, this creates $32 \times S^2$ elements that must be repeatedly written to and read from VRAM. |
| **Why is there a sharp knee in the curve?** | The **24 MB L2 Cache Cliff**. Tensors up to $S \approx 500$ fit in L2 cache (~2,500 GB/s). Tensors above $S = 1,000$ spill into the 300 GB/s off-chip GDDR6 DRAM. |

