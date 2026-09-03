# LLaMA-3.2-1B Autoregressive Inference: Systems Architecture, Hardware Interplay & Performance Guide

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [End-to-End Inference Steps](#2-end-to-end-inference-steps)
3. [Hardware Topology](#3-hardware-topology)
4. [Step-by-Step Hardware Interplay (CPU, GPU, Memory, PCIe)](#4-step-by-step-hardware-interplay)
5. [In-Depth Systems FAQ](#5-in-depth-systems-faq)

---

## 1. Architecture Overview

### Model Specification: meta-llama/Llama-3.2-1B-Instruct
* **Parameters**: 1.23 Billion
* **Hidden Dimension ($d_{\text{model}}$)**: 2,048
* **Layers**: 16
* **Attention Heads ($n_{\text{heads}}$)**: 32 (Head dimension $d_{\text{head}} = 64$)
* **KV Heads ($n_{\text{kv\_heads}}$)**: 8 (Grouped-Query Attention, 4 queries per KV head)
* **MLP Hidden Dimension ($d_{\text{mlp}}$)**: 8,192 (SwiGLU)
* **Vocabulary Size ($V$)**: 128,256
* **Normalization**: Root Mean Square Layer Normalization (RMSNorm, $\epsilon = 10^{-5}$)
* **Positional Encoding**: Rotary Position Embeddings (RoPE) with LLaMA 3.2 wavelength frequency scaling ($\theta = 500,000$)
* **Weight Tying**: Word embeddings (`embed_tokens`) and output vocabulary projection (`lm_head`) share identical weights.

### Structural Block Diagram

```
                 ┌──────────────────────────────────────┐
                 │          Input Text Prompt           │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │       BPE Tokenizer (CPU/Rust)       │
                 └──────────────────┬───────────────────┘
                                    │ input_ids [1, S]
                                    ▼
                 ┌──────────────────────────────────────┐
                 │      Embedding Layer (2048-dim)      │
                 └──────────────────┬───────────────────┘
                                    │ h [1, S, 2048]
                                    ▼
           ┌──────────────────────────────────────────────────┐
           │        Transformer Layer (Repeated 16x)          │
           │                                                  │
           │   x ───► RMSNorm ──► Attention ──(+)──► x        │
           │           │             ▲                        │
           │           │  RoPE(Q,K)  │                        │
           │           └─────────────┘                        │
           │                                                  │
           │   x ───► RMSNorm ──► SwiGLU FFN ─(+)──► x        │
           └────────────────────────┬─────────────────────────┘
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │            Final RMSNorm             │
                 └──────────────────┬───────────────────┘
                                    │ Slices ONLY last position: h[:, [-1], :]
                                    ▼
                 ┌──────────────────────────────────────┐
                 │      LM_HEAD (Shared Embeddings)     │
                 │     Projects 2048 -> 128256 Vocab    │
                 └──────────────────┬───────────────────┘
                                    │ Next-Token Logits [1, 1, 128256]
                                    ▼
                 ┌──────────────────────────────────────┐
                 │       Argmax / Softmax Sampling      │
                 └──────────────────┬───────────────────┘
                                    │ next_token_id (Scalar int)
                                    ▼
     [Append to curr_ids [1, S+1] & Recompute full sequence without KV Cache]
```

---

## 2. End-to-End Inference Steps

1. **Tokenizer (CPU)**: Converts raw input string into token IDs `[1, seq_len]`.
2. **Embedding (`embed_tokens`)**: Maps token IDs to dense hidden vectors `[1, seq_len, 2048]`. *(RoPE is NOT added here)*.
3. **16x Transformer Blocks**:
   - **RMSNorm $\rightarrow$ Attention $\rightarrow$ Residual Add**:
     - Pre-normalization via RMSNorm.
     - Projects $Q$ (32 heads), $K$ (8 heads), and $V$ (8 heads).
     - **Applies RoPE to $Q$ and $K$ vectors directly** (rotates vector pairs).
     - Broadcasts $K$ and $V$ across query groups (GQA, $4\times$).
     - Computes causal attention: $\text{softmax}\left(\frac{Q K^T}{\sqrt{d}} + M_{\text{causal}}\right) V$.
     - Linear output projection $O$ and residual addition: $x = x + \text{Attention}(x)$.
   - **RMSNorm $\rightarrow$ SwiGLU FFN $\rightarrow$ Residual Add**:
     - Pre-normalization via RMSNorm.
     - SwiGLU computation: $\text{down\_proj}(\text{SiLU}(\text{gate\_proj}(x)) \odot \text{up\_proj}(x))$.
     - Residual addition: $x = x + \text{FFN}(x)$.
4. **Final RMSNorm**: Normalizes the output hidden states from layer 16.
5. **LM_HEAD (Vocabulary Projection)**:
   - Slices **strictly the final sequence position** $h[:, [-1], :]$.
   - Projects 2,048 dimensions to 128,256 vocabulary logits `[1, 1, 128256]`.
6. **Sampling**: Applies greedy $\text{argmax}$ (or temperature softmax + multinomial) to select the single integer `next_token_id`.
7. **Append & Loop Back**: Decodes `next_token_id` to text for streaming, concatenates it to `curr_ids` `[1, seq_len + 1]`, and re-enters **Step 2** with the full accumulated sequence.

---

## 3. Hardware Topology

Understanding execution bottlenecks requires tracking where data sits and which buses it traverses:

```
┌────────────────────────────────────────────────────────────────────────┐
│                              CPU (HOST)                                │
│  • Host System Memory (RAM): OS, Python Runtime, Tokenizer Engine      │
│  • CPU Compute Cores: String parsing, HuggingFace BPE, Driver dispatch │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
             ═══════════════════════╪═══════════════════════
                    PCIe Bus (Gen 4 x16: ~31.5 GB/s bidirectional)
                    • HtoD (Host-to-Device): input_ids (~80 B)
                    • DtoH (Device-to-Host): next_token_id.item() (~8 B)
             ═══════════════════════╪═══════════════════════
                                    │
┌───────────────────────────────────┴────────────────────────────────────┐
│                              GPU (DEVICE)                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     GPU VRAM (HBM / GDDR6)                       │  │
│  │  • Model Weights: ~2.46 GB resident (bfloat16)                   │  │
│  │  • Activations: h [1, S, 2048], RoPE frequency tables           │  │
│  │  • Scratch buffers: attention matrices, logits                   │  │
│  └────────────────────────────────┬─────────────────────────────────┘  │
│                                   │ GPU Memory Bus (~300 GB/s on L4)   │
│  ┌────────────────────────────────┴─────────────────────────────────┐  │
│  │                    GPU Streaming Multiprocessors                 │  │
│  │  • L2 Cache (Shared on-chip buffer: 24 - 48 MB)                  │  │
│  │  • SM Shared Memory / SRAM & Register Files (~20 TB/s bandwidth) │  │
│  │  • 4th Gen Tensor Cores & FP32/FP16 ALUs (~120 TFLOPs bfloat16)  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Step-by-Step Hardware Interplay

| Step | Operation | Compute Device | Primary Memory Used | Bus Activity | Synchronization Behavior |
|---|---|---|---|---|---|
| **0** | Model Loading | CPU $\rightarrow$ GPU | Disk $\rightarrow$ Host RAM $\rightarrow$ GPU VRAM | PCIe HtoD: 2.46 GB transfer | Blocking (one-time initialization) |
| **1** | Tokenize Prompt | **CPU** | Host RAM | None | None |
| **1b** | Transfer `input_ids` | Driver | Host RAM $\rightarrow$ GPU VRAM | **PCIe HtoD**: ~80 bytes | Non-blocking async transfer |
| **2** | `embed_tokens` | **GPU SM** | GPU VRAM (reads 525 MB table) | GPU Memory Bus: ~300 GB/s | Asynchronous CUDA kernel launch |
| **3** | 16x Transformer Blocks | **GPU Tensor Cores & SMs** | GPU VRAM $\leftrightarrow$ SM Registers/L2 Cache | GPU Memory Bus: Streams ~1.9 GB weights | Asynchronous (CPU queues kernels ahead) |
| **4** | Final RMSNorm | **GPU SM** | GPU VRAM / L2 Cache | GPU Memory Bus | Asynchronous |
| **5** | LM_HEAD Projection | **GPU Tensor Cores** | GPU VRAM (reads 525 MB weights) | GPU Memory Bus | Asynchronous |
| **6a** | Argmax / Sampling | **GPU SM** | GPU VRAM (reads 256 KB logits) | GPU Memory Bus | Asynchronous (outputs 8-byte scalar) |
| **6b** | `next_token_id.item()` | **CPU & GPU** | GPU VRAM $\rightarrow$ Host RAM | **PCIe DtoH**: 8 bytes | **CRITICAL PIPELINE STALL**: CPU halts until GPU finishes all prior work |
| **7** | Detokenize & Cat | **CPU & GPU** | CPU: Host RAM (decode)<br>GPU: VRAM (`torch.cat`) | PCIe: Idle | CPU resumes immediately |

### The Critical CPU-GPU Synchronization Point (`.item()`)
During inference, PyTorch executes CUDA operations asynchronously: the CPU thread queues kernels into the GPU command stream and immediately moves to the next line of code without waiting for the GPU to finish.

However, calling `token_val = next_token_id.item()` requires the actual scalar integer to make control-flow decisions (checking stop tokens and streaming text). This forces a **device-to-host synchronization stall**:
1. The CPU thread completely freezes.
2. The GPU must drain its entire execution queue (all 16 layers, LM Head, and Argmax).
3. The 8-byte token integer is transferred across PCIe into Host RAM.
4. Only then does the CPU thread unblock to print the token and launch the next step.

---

## 5. In-Depth Systems FAQ

### Q1: In the Attention layer, do the linear projections ($W_q, W_k, W_v$) happen one by one or all at once?
**In our baseline implementation: Strictly one-by-one.**
* PyTorch launches 3 independent kernels (`q_proj`, `k_proj`, `v_proj`).
* The GPU is forced to read the input activation tensor $x$ from VRAM **three separate times**, and launches three separate kernel dispatches.
* **Production Optimization (Fused QKV)**: Advanced runtimes (vLLM, TensorRT-LLM) pack the weight matrices into a single fused tensor:
  $$W_{\text{qkv}} = \begin{bmatrix} W_q \\ W_k \\ W_v \end{bmatrix}$$
  A single fused kernel loads the activation vector $x$ from VRAM **once**, streams $W_{\text{qkv}}$ in one pass, computes $Q, K, V$ simultaneously, and writes them to the KV cache.

---

### Q2: What is GEMV vs. GEMM, and why does it matter?
* **GEMM (General Matrix-Matrix Multiply)**:
  $$C = A \times B \quad (M > 1)$$
  Multiplies a 2D matrix by a 2D matrix. Used in **prompt prefill** and batched training.
* **GEMV (General Matrix-Vector Multiply)**:
  $$y = A \times x \quad (M = 1)$$
  Multiplies a 2D weight matrix by a **single 1D vector** (one token). Used in **autoregressive decode**.

#### The Arithmetic Intensity Reality
$$\text{Arithmetic Intensity} = \frac{\text{Computation FLOPs}}{\text{Memory Transferred (Bytes)}}$$

On our NVIDIA L4 GPU:
* **Tensor Core Speed**: $120\text{ TFLOPs/s}$
* **Memory Bandwidth**: $300\text{ GB/s}$
* **Breakeven Ratio**: $\frac{120 \times 10^{12}}{300 \times 10^9} = \mathbf{400\text{ FLOPs/Byte}}$

During single-token decode (**GEMV**):
* We read 2 bytes of weight (bfloat16) from VRAM.
* We perform 1 multiply-add with the single token feature ($2\text{ FLOPs}$).
* Arithmetic intensity:
  $$\frac{2\text{ FLOPs}}{2\text{ Bytes}} = \mathbf{1\text{ FLOP/Byte}}$$
Because $1 \ll 400$, the GPU Tensor Cores sit idle **99.7% of the time**, stalled waiting for the memory bus to stream weights from VRAM. Autoregressive token generation is fundamentally **memory-bandwidth bound**.

---

### Q3: How does sequence length ($S$) affect performance in our setup vs. production?

#### In Our Baseline Setup (No KV Cache):
Because we re-pass all accumulated tokens through the model on every decode step:
1. **Linear Projections Scale Linearly ($O(S)$)**:
   * When $S$ is small (1–16 tokens), it acts as a memory-bound GEMV.
   * As $S$ grows (100–500+ tokens), $M = S$ increases, transitioning toward GEMM. While weight reuse improves, total compute increases linearly with $S$.
2. **Attention Scales Quadratically ($O(S^2)$)**:
   * Computing $Q K^T$ requires an $S \times S$ matrix multiplication from scratch every single token.
   * Latency per token progressively degrades as sequence length increases.

#### In Production Systems (With KV Cache):
1. **Linear Projections Remain Constant ($O(1)$)**:
   * Only the **1 new token** is projected through $W_q, W_k, W_v$ and the FFN.
   * Linear projections remain strictly fixed GEMV operations regardless of whether context is 10 tokens or 10,000 tokens.
2. **Attention Scales Linearly ($O(S)$)**:
   * The single query vector ($1 \times d$) attends to the cached key tensor ($S \times d$).
   * Only the memory traffic for loading cached keys and values increases with sequence length.

---

### Key Formulas & Cheat Sheet

| Metric | Formula |
|---|---|
| **Total Model Weight Size (bfloat16)** | $\text{Params} \times 2\text{ bytes} \approx 1.23 \times 10^9 \times 2 \approx 2.46\text{ GB}$ |
| **Minimum VRAM Bandwidth Time per Token** | $\frac{\text{Model Weights}}{\text{VRAM Bandwidth}} = \frac{2.46\text{ GB}}{300\text{ GB/s}} \approx \mathbf{8.2\text{ ms}}$ |
| **Theoretical Peak Decode Throughput (Batch 1)** | $\frac{1}{8.2\text{ ms}} \approx \mathbf{121\text{ tokens/sec}}$ (On L4 GPU) |
| **Empirical Decode Throughput Measured** | $\mathbf{77.2\text{ tokens/sec}}$ (~12.9 ms/token, reflecting kernel launch & synchronization overhead) |

