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

## Deep-Dive Documentation

For detailed analysis of GPU execution mechanics and memory systems, see the guides in `docs/`:
- [Unoptimized Attention Breakdown & Memory Physics](file:///home/ayushdeva_google_com/tiny_vllm/docs/unoptimized_attention_breakdown.md)
- [GPU Memory Wait & Scheduling FAQ](file:///home/ayushdeva_google_com/tiny_vllm/docs/GPU_MEMORY_WAIT_AND_SCHEDULING_FAQ.md)
- [Inference Architecture & Hardware Interplay](file:///home/ayushdeva_google_com/tiny_vllm/docs/INFERENCE_ARCHITECTURE_AND_HARDWARE_INTERPLAY.md)
- [Profiling Experiments & Systems Learnings](file:///home/ayushdeva_google_com/tiny_vllm/docs/PROFILING_EXPERIMENTS_AND_SYSTEMS_LEARNINGS.md)



