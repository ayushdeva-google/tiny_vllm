# tiny_vllm

A lightweight, from-scratch implementation and deep systems profiling suite for LLaMA-3.2 autoregressive inference on NVIDIA GPUs.

---

## Architecture & Features

- **Minimalist LLaMA-3.2-1B Engine**: Autoregressive decoding from first principles with HuggingFace safetensors loader, RoPE, RMSNorm, SwiGLU, and Grouped-Query Attention (GQA).
- **Fine-Grained Sampled Profiler**: Instruments model operations (`Q_Linear`, `K_Linear`, `V_Linear`, `RoPE`, `Attn_Compute`, `FFN`, `LM_Head`) using PyTorch Profiler (`torch.profiler.record_function`) and exports per-step Chrome trace JSONs.
- **Hardware Decomposition**: Measures active GPU kernel time vs. host CPU launch/dispatch gaps to identify memory-bound, compute-bound, and host-starvation bottlenecks.
- **Interactive Visualizer**: Standalone HTML dashboard generator (`profile_visualizer.py`) featuring:
  - Microsecond Gantt execution timeline with zoom controls.
  - Interactive per-step operation breakdown.
  - Context-length scaling curve (e.g. quadratic attention recalculation without KV cache vs flat weight streaming).
  - VRAM consumption and duty cycle KPIs.
- **Evaluation Suite**: Validates generation quality against HuggingFace transformers using WikiText-2 perplexity, greedy token parity, and ARC-Easy benchmark.

---

## Learning Sessions & Chapter Structure

To ensure every learning session and experiment is isolated and reproducible, the project organizes codes, artifacts, metrics, and dashboards into dedicated chapter directories:

### [Chapter 1: Learning How Profiling Works](file:///home/ayushdeva_google_com/tiny_vllm/chapter_1)

**Focus**: Understanding systems profiling for LLM autoregressive inference from first principles.
- **Key Concepts Explored**:
  - Fine-grained instrumentation of attention and feed-forward layers (`torch.profiler.record_function`).
  - Active GPU kernel compute vs. host CPU launch/dispatch gaps (identifying driver overhead and GPU starvation).
  - Context-length scaling behavior (quadratic attention recalculation without KV cache vs. memory-bound weight streaming).
  - Multi-tier visual reporting via Rich terminal summaries, Chrome trace exports (`.trace.json`), and standalone interactive HTML dashboards.
- **Directory Contents** (`chapter_1/`):
  - [llama_inference.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_1/llama_inference.py): Minimalist from-scratch LLaMA-3.2-1B inference implementation.
  - [llama_inference_with_profiling.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_1/llama_inference_with_profiling.py): Profiled inference runner with sampled step profiling.
  - [profile_visualizer.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_1/profile_visualizer.py): Interactive HTML dashboard generator and terminal analyzer.
  - [eval_quality.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_1/eval_quality.py): Quality and accuracy validation suite (WikiText-2, greedy parity, ARC-Easy).
  - [profile_results/](file:///home/ayushdeva_google_com/tiny_vllm/chapter_1/profile_results): Saved profiling session data (`token_metrics.json`), PyTorch Chrome traces, and the interactive dashboard (`profile_dashboard.html`).

### [Chapter 2: Key-Value (KV) Cache](file:///home/ayushdeva_google_com/tiny_vllm/chapter_2_kvcache)

**Focus**: Implementing production-grade Key-Value caching from first principles and eliminating quadratic decode recomputation.
- **Key Concepts Explored**:
  - Pre-allocated static tensor buffer (`KVCache`) to eliminate GPU reallocations and memory fragmentation.
  - Grouped-Query Attention (GQA) cache footprint: caching 8 unrepeated KV heads instead of 32 query heads to save 75% VRAM and memory bandwidth.
  - Prefill vs. Decode execution split ($O(1)$ linear projections during decode, bypassing causal masking).
  - Shift in arithmetic intensity: from compute-bound quadratic scaling to constant-time memory-bandwidth-bound streaming.
- **Directory Contents** (`chapter_2_kvcache/`):
  - [llama_inference_with_profiling.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_2_kvcache/llama_inference_with_profiling.py): Complete, standalone LLaMA-3.2-1B inference engine with static `KVCache` and fine-grained sampled profiling.
  - [profile_visualizer.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_2_kvcache/profile_visualizer.py): Standalone HTML dashboard generator and Gantt timeline analyzer.
  - [profile_results/](file:///home/ayushdeva_google_com/tiny_vllm/chapter_2_kvcache/profile_results): Generated profiling sessions, Chrome traces, and interactive HTML dashboard.

### [Chapter 3: Kernel Fusion & FlashAttention](file:///home/ayushdeva_google_com/tiny_vllm/chapter_3_kernel_fusion)

**Focus**: Eliminating host launch starvation and VRAM roundtrips with custom Triton and CUDA C++ kernels for `RMSNorm` and `FlashAttention`.
- **Key Concepts Explored**:
  - **Part A — RMSNorm Fusion**: Fusing multi-kernel PyTorch eager operations into single-pass Triton (`triton_rmsnorm.py`) and raw CUDA warp-shuffle (`cuda_rmsnorm/`) kernels.
  - **Part B — FlashAttention from First Principles**: Deriving Online Softmax, Split-K Flash Decode (`cuda_flash_decode/`, `cuda_warp_decode/`), and native BFloat16 Split-K decode + tiled causal prefill (`cuda_bf16_flash_attention/`).
  - Single-source correctness and latency/VRAM verification (`benchmark_chapter3.py`).
- **Directory Contents** (`chapter_3_kernel_fusion/`):
  - [benchmark_chapter3.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_3_kernel_fusion/benchmark_chapter3.py): Unified correctness, decode latency, and prefill VRAM benchmark for all Chapter 3 attention kernels.
  - [cuda_rmsnorm/](file:///home/ayushdeva_google_com/tiny_vllm/chapter_3_kernel_fusion/cuda_rmsnorm) & [triton_rmsnorm.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_3_kernel_fusion/triton_rmsnorm.py): Custom CUDA and Triton `RMSNorm` kernels.
  - [cuda_flash_decode/](file:///home/ayushdeva_google_com/tiny_vllm/chapter_3_kernel_fusion/cuda_flash_decode), [cuda_warp_decode/](file:///home/ayushdeva_google_com/tiny_vllm/chapter_3_kernel_fusion/cuda_warp_decode), [cuda_bf16_flash_attention/](file:///home/ayushdeva_google_com/tiny_vllm/chapter_3_kernel_fusion/cuda_bf16_flash_attention): Progressive CUDA attention implementations.
  - [llama_inference_with_profiling.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_3_kernel_fusion/llama_inference_with_profiling.py) & [profile_visualizer.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_3_kernel_fusion/profile_visualizer.py): End-to-end model profiling with fused `RMSNorm` and speedup dashboard generator.

### [Chapter 5: Batch Inference — Evaluation, Static Batching & Continuous Batching](file:///home/ayushdeva_google_com/tiny_vllm/chapter_5_batch_inference)

**Focus**: Online/offline serving evaluation, queueing physics ($\lambda, \beta$ Gamma/Poisson arrivals), and progressive batching schedulers in pure PyTorch (`bfloat16` + KV cache).
- **Key Concepts Explored**:
  - Open-loop arrival simulation ($\Delta t \sim \text{Gamma}(\beta, \frac{1}{\lambda\beta})$) across real ShareGPT conversations (`data/sharegpt_200.json`) and synthetic WikiText-2 length distributions.
  - **Stage 0 — Sequential FCFS (`B=1`)**: Head-of-Line (HoL) queueing blowup (`~0.34 req/s` saturation ceiling on L4) and HBM weight-streaming bottleneck (`39.3 tok/s`).
  - **Stage 1 — Static (Request-Level) Padded Batching**: Left-padded batched prefill with per-sequence RoPE `position_ids` + lockstep decode; quantifies prefill padding waste (`~64–75%`) and zombie decode slot waste (`~39–47%`) on skewed prompts.
  - **Stage 2 — Continuous (Iteration-Level + Selective) Batching**: Orca-style per-iteration KV slot recycling + ragged unpadded token packing (`300.9 tok/s` / `7.65x` burst throughput, `140x` lower median TTFT at busy online QPS with `0%` padding/zombie waste).
- **Directory Contents** (`chapter_5_batch_inference/`):
  - [llama_model.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_5_batch_inference/llama_model.py) & [engines.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_5_batch_inference/engines.py): Standalone `B=1` Sequential FCFS baseline engine.
  - [static_batching.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_5_batch_inference/static_batching.py): Standalone Static Padded Batching model, KV cache, and scheduler.
  - [continuous_batching.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_5_batch_inference/continuous_batching.py): Standalone Continuous Batching model, `SlotKVCache`, `SelectiveAttention`, and iteration-level scheduler.
  - [workloads.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_5_batch_inference/workloads.py) & [profiler_metrics.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_5_batch_inference/profiler_metrics.py): Arrival schedule generator, serving percentile/goodput aggregator, and `torch.profiler` trace analyzer.
  - [benchmark_serving.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_5_batch_inference/benchmark_serving.py) & [benchmark_batch_comparison.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_5_batch_inference/benchmark_batch_comparison.py): Single-engine and multi-engine $(\lambda, \beta)$ benchmark + FP32 parity verification runners.
  - [results/](file:///home/ayushdeva_google_com/tiny_vllm/chapter_5_batch_inference/results): Saved JSON benchmark results (`batch_comparison_sharegpt.json`, `sequential_fcfs_b1_sharegpt_sweep.json`, etc.).

---

## Quickstart & Setup

### 1. Environment Setup

Ensure your virtual environment is activated and dependencies are installed:

```bash
# Activate virtual environment
source .venv/bin/activate

# Verify PyTorch with CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0)})')"
```

---

## Chapter 1 Profiling & Dashboard Workflow

Commands can be run either from the project root or by navigating into `chapter_1/`.

### Step 1: Run Inference with Profiling

Run [llama_inference_with_profiling.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_1/llama_inference_with_profiling.py) to generate tokens and output fine-grained execution metrics + Chrome traces into `chapter_1/profile_results/`:

```bash
python chapter_1/llama_inference_with_profiling.py \
  --prompt "Write a comprehensive guide on quantum computing principles." \
  --max_new_tokens 2048 \
  --profile_interval 250 \
  --ignore_eos
```

#### Key Arguments:
- `--prompt`: Input prompt to seed generation (default: `"Write a comprehensive guide on quantum computing principles."`).
- `--max_new_tokens`: Maximum new tokens to generate (e.g., `500` for a fast run, `2048` for deep context scaling).
- `--profile_interval`: Sampling interval for deep PyTorch trace captures (e.g., `100` or `250`). Trace captures step 0, every N steps, and the final step.
- `--ignore_eos`: Continue generating up to `max_new_tokens` even if `<|eot_id|>` or `<|end_of_text|>` is sampled (ideal for fixed-length latency benchmarking).
- `--profile_output_dir`: Target directory for traces and JSON metrics (default: `chapter_1/profile_results`).

> **Tip for quick testing**:
> ```bash
> python chapter_1/llama_inference_with_profiling.py --max_new_tokens 500 --profile_interval 100 --ignore_eos
> ```

---

### Step 2: Generate Interactive HTML Dashboard

Run [profile_visualizer.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_1/profile_visualizer.py) on the generated `token_metrics.json` file:

```bash
python chapter_1/profile_visualizer.py
```

*(By default, it automatically loads `chapter_1/profile_results/token_metrics.json` and generates `chapter_1/profile_results/profile_dashboard.html`)*

#### Optional Flags:
- `--json`: Path to the input metrics JSON (default: `chapter_1/profile_results/token_metrics.json`).
- `--output`: Output path for the HTML dashboard (default: `chapter_1/profile_results/profile_dashboard.html`).
- `--terminal`: Also print detailed Rich terminal summary tables.

Open the resulting dashboard in any browser:
```text
chapter_1/profile_results/profile_dashboard.html
```

---

## Additional Tools & Evaluations

### Quality Evaluation Suite
Run [eval_quality.py](file:///home/ayushdeva_google_com/tiny_vllm/chapter_1/eval_quality.py) to measure perplexity and ensure accuracy parity:

```bash
python chapter_1/eval_quality.py
```

---

## Deep-Dive Documentation & Presentations

Interactive visual decks in `presentations/`:
- [CUDA 101 Interactive Deck](file:///home/ayushdeva_google_com/tiny_vllm/presentations/cuda-101.html)
- [Chapter 2: KV Cache Dashboard](file:///home/ayushdeva_google_com/tiny_vllm/presentations/chapter2-kvcache-dashboard-v3.html)
- [Chapter 3: RMSNorm Speedup Dashboard](file:///home/ayushdeva_google_com/tiny_vllm/presentations/chapter3-rmsnorm-speedup-dashboard.html)
- [Chapter 3: FlashAttention Derivation](file:///home/ayushdeva_google_com/tiny_vllm/presentations/chapter3-flash-attention-derivation.html)
- [Chapter 5: Request Arrival & Queueing Eval Explorer](file:///home/ayushdeva_google_com/tiny_vllm/presentations/chapter5-request-arrival-explorer.html)
- [Chapter 5: Static vs. Continuous Batching Benchmark Dashboard](file:///home/ayushdeva_google_com/tiny_vllm/presentations/chapter5-batch-comparison-dashboard.html)

For detailed analysis of GPU execution mechanics and memory systems, see the guides in `docs/`:
- [CUDA 101 Guide](file:///home/ayushdeva_google_com/tiny_vllm/docs/CUDA_101.md)
- [Quantization 101 Guide](file:///home/ayushdeva_google_com/tiny_vllm/docs/QUANTIZATION_101.md)
- [Unoptimized Attention Breakdown & Memory Physics](file:///home/ayushdeva_google_com/tiny_vllm/docs/unoptimized_attention_breakdown.md)
- [GPU Memory Wait & Scheduling FAQ](file:///home/ayushdeva_google_com/tiny_vllm/docs/GPU_MEMORY_WAIT_AND_SCHEDULING_FAQ.md)
- [Inference Architecture & Hardware Interplay](file:///home/ayushdeva_google_com/tiny_vllm/docs/INFERENCE_ARCHITECTURE_AND_HARDWARE_INTERPLAY.md)
- [Profiling Experiments & Systems Learnings](file:///home/ayushdeva_google_com/tiny_vllm/docs/PROFILING_EXPERIMENTS_AND_SYSTEMS_LEARNINGS.md)



