# Chapter 5: Batch Inference — Evaluation, Static Batching & Continuous Batching

This directory implements an industry-standard **online/offline LLM serving benchmark and hardware profiling suite** for **Llama-3.2-1B-Instruct** in pure eager PyTorch (`bfloat16`) on an **NVIDIA L4 (24 GB)**, comparing three completely standalone, independent serving architectures:

1. **Baseline — Sequential FCFS (`B=1`)**: [`llama_model.py`](llama_model.py) + [`engines.py`](engines.py)
2. **Stage 1 — Static (Request-Level) Padded Batching**: [`static_batching.py`](static_batching.py)
3. **Stage 2 — Continuous (Iteration-Level + Selective) Batching**: [`continuous_batching.py`](continuous_batching.py)

---

## Directory Structure

- [`llama_model.py`](llama_model.py) & [`engines.py`](engines.py): Standalone `B=1` Sequential FCFS baseline engine with static `KVCache`.
- [`static_batching.py`](static_batching.py): Standalone native PyTorch model + `StaticBatchEngine` (left-padded batched prefill with per-sequence RoPE `position_ids` + causal/pad masking, and lockstep batched decode until the longest straggler in the batch finishes).
- [`continuous_batching.py`](continuous_batching.py): Standalone native PyTorch model + Orca-style `ContinuousBatchEngine` with `SlotKVCache`:
  - **Iteration-Level Slot Recycling**: Completed sequences free their KV slot at the end of *that exact step*; waiting requests claim free slots on the very next iteration (`0%` zombie decode waste).
  - **Selective Ragged Token Packing**: Token-independent layers (`Embedding`, `RMSNorm`, `Q/K/V/O_Linear`, `RoPE`, `SwiGLU`, `LM_Head`) pack all newly admitted prefill tokens and active decode tokens into a single unpadded `(1, T_total, dim)` GEMM (`0%` prefill padding waste).
- [`workloads.py`](workloads.py): Deterministic ShareGPT (`../data/sharegpt_200.json`) & synthetic WikiText-2 workloads + open-loop Gamma/Poisson arrival generator (`--request-rate` $\lambda$, `--burstiness` $\beta$).
- [`profiler_metrics.py`](profiler_metrics.py): Latency/queueing percentile aggregator, Little's Law queue depth, SLO Goodput, and observer-free `torch.profiler` Chrome trace decomposition.
- [`benchmark_serving.py`](benchmark_serving.py): Single-engine benchmark & QPS-sweep CLI.
- [`benchmark_batch_comparison.py`](benchmark_batch_comparison.py): Unified correctness gate (FP32 exact token parity) + multi-engine $(\lambda, \beta)$ comparison & hardware profiler CLI.
- [`results/`](results/): Saved JSON benchmark artifacts (`batch_comparison_sharegpt.json`, `sequential_fcfs_b1_sharegpt_sweep.json`, `sequential_fcfs_b1_synthetic-skewed_sweep.json`, `sequential_fcfs_b1_synthetic-uniform_qps_1.5.json`).
- **Interactive HTML Presentations** (`../presentations/`):
  - [`chapter5-request-arrival-explorer.html`](../presentations/chapter5-request-arrival-explorer.html): Interactive 100ms Poisson/Gamma arrival stream, ShareGPT query archetype scatter, and `B=1` queue waterfall explorer.
  - [`chapter5-batch-comparison-dashboard.html`](../presentations/chapter5-batch-comparison-dashboard.html): Interactive NVIDIA L4 $(\lambda, \beta)$ benchmark curves and side-by-side scheduler waterfall (`B=1` vs. `Static B=6` vs. `Continuous B=6`).

---

## Quickstart Commands

```bash
# 1. Run full ShareGPT (lambda, beta) comparison across B=1, Static(B=8,16), and Continuous(B=8,16)
.venv/bin/python chapter_5_batch_inference/benchmark_batch_comparison.py \
  --workload sharegpt --num-requests 36 --batch-sizes 8,16

# 2. Fast custom scenario comparison (e.g., busy Poisson vs micro-bursts vs offline burst)
.venv/bin/python chapter_5_batch_inference/benchmark_batch_comparison.py \
  --workload sharegpt --num-requests 24 --batch-sizes 8,16 \
  --scenarios 1.0:1.0,2.5:1.0,2.5:0.3,inf:1.0

# 3. Single baseline QPS sweep
.venv/bin/python chapter_5_batch_inference/benchmark_serving.py \
  --workload sharegpt --num-requests 20 --max-output-tokens 128 \
  --qps-sweep 0.15,0.35,0.6,1.0,inf
```

---

## Empirical Results on NVIDIA L4 (`ShareGPT`, `N=36` Uncapped Queries, `in=40..474`, `out=34..374`)

Full structured metrics saved in [`results/batch_comparison_sharegpt.json`](results/batch_comparison_sharegpt.json).
**Correctness Gate**: `8/8` requests (`160/160` tokens, including mixed prefill+decode steps) achieved **100% exact FP32 greedy token parity** across `B=1`, `StaticBatch`, and `ContinuousBatch`.

### 1. Sweep A — Varying Arrival Rate $\lambda$ (Standard Poisson, $\beta = 1.0$)

| Scenario ($\lambda, \beta$) | Engine | Out `tok/s` (Speedup) | `req/s` | Queue Wait P50 / P99 | TTFT P50 / P99 | TPOT P50 | SLO Goodput (`TTFT≤2s, TPOT≤50ms`) | Prefill Pad / Zombie Waste |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Light** (`0.25, 1.0`) | `sequential_fcfs_b1` | `38.8` (`1.00x`) | `0.20` | `27,666 / 58,868 ms` | `27,695 / 58,896 ms` | `25.5 ms` | `0.03 rps (14%)` | `0% / 0%` |
| | `static_batch_b8` | `54.7` (`1.41x`) | `0.28` | `1,937 / 8,933 ms` | `2,035 / 8,962 ms` | `27.3 ms` | `0.14 rps (50%)` | `27.2% / 21.2%` |
| | `continuous_batch_b8` | **`57.2` (`1.47x`)** | **`0.30`** | **`12 / 28 ms`** | **`51 / 74 ms`** | `28.9 ms` | **`0.30 rps (100%)`** | **`0% / 0%`** |
| **Moderate** (`1.0, 1.0`) | `sequential_fcfs_b1` | `39.4` (`1.00x`) | `0.20` | `78,366 / 140,806 ms` | `78,394 / 140,834 ms` | `25.3 ms` | `0.01 rps (6%)` | `0% / 0%` |
| | `static_batch_b8` | `138.8` (`3.52x`) | `0.72` | `5,275 / 12,364 ms` | `5,564 / 12,519 ms` | `27.7 ms` | `0.12 rps (17%)` | `68.9% / 36.6%` |
| | `static_batch_b16` | `155.5` (`3.94x`) | `0.81` | `4,190 / 9,598 ms` | `4,437 / 9,883 ms` | `28.8 ms` | `0.20 rps (25%)` | `63.9% / 39.1%` |
| | `continuous_batch_b8` | `174.3` (`4.42x`) | `0.91` | `22 / 2,037 ms` | `60 / 2,076 ms` | `30.3 ms` | `0.88 rps (97%)` | `0% / 0%` |
| | `continuous_batch_b16` | **`182.8` (`4.64x`)** | **`0.95`** | **`18 / 28 ms`** | **`55 / 66 ms`** | `29.5 ms` | **`0.95 rps (100%)`** | **`0% / 0%`** |
| **Busy** (`2.5, 1.0`) | `sequential_fcfs_b1` | `39.3` (`1.00x`) | `0.20` | `88,938 / 158,369 ms` | `88,966 / 158,397 ms` | `25.5 ms` | `0.01 rps (6%)` | `0% / 0%` |
| | `static_batch_b8` | `153.6` (`3.91x`) | `0.80` | `11,086 / 24,155 ms` | `11,407 / 24,450 ms` | `27.4 ms` | `0.09 rps (11%)` | `63.9% / 39.6%` |
| | `static_batch_b16` | `146.7` (`3.74x`) | `0.76` | `8,071 / 26,914 ms` | `8,666 / 26,947 ms` | `36.3 ms` | `0.13 rps (17%)` | `71.0% / 43.9%` |
| | `continuous_batch_b8` | `212.5` (`5.41x`) | `1.10` | `4,893 / 11,950 ms` | `4,930 / 11,989 ms` | `29.9 ms` | `0.34 rps (31%)` | `0% / 0%` |
| | `continuous_batch_b16` | **`270.2` (`6.88x`)** | **`1.40`** | **`26 / 4,193 ms`** | **`62 / 4,242 ms`** | `37.8 ms` | **`0.97 rps (69%)`** | **`0% / 0%`** |
| **Heavy** (`5.0, 1.0`) | `static_batch_b16` | `172.4` (`4.41x`) | `0.90` | `8,211 / 22,115 ms` | `8,880 / 22,572 ms` | `33.4 ms` | `0.20 rps (22%)` | `69.8% / 44.9%` |
| | `continuous_batch_b16` | **`286.3` (`7.31x`)** | **`1.49`** | **`366 / 8,672 ms`** | **`409 / 8,712 ms`** | `38.9 ms` | **`0.79 rps (53%)`** | **`0% / 0%`** |
| **Offline Burst** (`inf`) | `sequential_fcfs_b1` | `39.3` (`1.00x`) | `0.20` | `95,879 / 169,402 ms` | `95,907 / 169,431 ms` | `25.4 ms` | `0.01 rps (6%)` | `0% / 0%` |
| | `static_batch_b8` | `145.3` (`3.70x`) | `0.76` | `19,612 / 38,376 ms` | `19,763 / 38,501 ms` | `27.8 ms` | `0.17 rps (22%)` | `65.2% / 42.9%` |
| | `static_batch_b16` | `164.5` (`4.18x`) | `0.85` | `15,976 / 32,386 ms` | `16,595 / 32,515 ms` | `37.7 ms` | `0.38 rps (44%)` | `74.9% / 46.9%` |
| | `continuous_batch_b8` | `224.5` (`5.71x`) | `1.17` | `9,716 / 21,714 ms` | `9,752 / 21,753 ms` | `29.3 ms` | `0.32 rps (28%)` | `0% / 0%` |
| | `continuous_batch_b16` | **`300.9` (`7.65x`)** | **`1.56`** | **`3,295 / 12,792 ms`** | **`3,339 / 12,831 ms`** | `40.8 ms` | **`0.74 rps (47%)`** | **`0% / 0%`** |

### 2. Sweep B — Varying Burstiness $\beta$ at Fixed Rate $\lambda = 2.5\text{ req/s}$ (`B=16`)

| Arrival Shape ($\lambda = 2.5\text{ req/s}$) | Engine | Out `tok/s` | Peak Queue | TTFT P50 / P99 | ITL P99 | SLO Goodput (`TTFT≤2s, TPOT≤50ms`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Micro-Bursty (`β = 0.30`)** | `static_batch_b16` | `174.8` | `19` | `6,579 / 18,095 ms` | `50.5 ms` | `0.13 rps (14%)` |
| | `continuous_batch_b16` | **`283.0`** | `11` | **`81 / 4,554 ms`** | `52.0 ms` | **`0.90 rps (61%)`** |
| **Standard Poisson (`β = 1.00`)** | `static_batch_b16` | `146.7` | `18` | `8,666 / 26,947 ms` | `48.7 ms` | `0.13 rps (17%)` |
| | `continuous_batch_b16` | **`270.2`** | `11` | **`62 / 4,242 ms`** | `50.5 ms` | **`0.97 rps (69%)`** |
| **Smooth Periodic (`β = 8.00`)** | `static_batch_b16` | `162.6` | `17` | `5,977 / 21,722 ms` | `49.8 ms` | `0.21 rps (25%)` |
| | `continuous_batch_b16` | **`267.8`** | **`6`** | **`61 / 2,538 ms`** | `50.1 ms` | **`1.24 rps (89%)`** |

### 3. Hardware `torch.profiler` Decode Scaling (NVIDIA L4)

| Engine | Tokens / Step | Step Wall (`ms`) | GPU Active (`ms`) | Host CPU Gap (`ms`) | GPU Duty Cycle | Step Decode Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sequential_fcfs_b1` | `1` | `25.51 ms` | `11.22 ms` | `14.29 ms` | `44.0%` | `39.2 tok/s` |
| `static_batch_b8` | `8` | `26.71 ms` | `15.49 ms` | `11.22 ms` | `58.0%` | `299.5 tok/s` |
| `continuous_batch_b8` | `8` | `28.66 ms` | `15.35 ms` | `13.31 ms` | `53.6%` | `279.1 tok/s` |
| `static_batch_b16` | `16` | `26.77 ms` | `18.13 ms` | `8.64 ms` | `67.7%` | `597.7 tok/s` |
| `continuous_batch_b16` | `16` | `27.80 ms` | `18.61 ms` | `9.19 ms` | `66.9%` | `575.5 tok/s` |
