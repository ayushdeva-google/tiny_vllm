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

## Profiling & Dashboard Workflow

Generating the end-to-end performance analysis is a two-step process:

### Step 1: Run Inference with Profiling

Run [llama_inference_with_profiling.py](file:///home/ayushdeva_google_com/tiny_vllm/llama_inference_with_profiling.py) to generate tokens and output fine-grained execution metrics + Chrome traces into `profile_results/`:

```bash
python llama_inference_with_profiling.py \
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
- `--profile_output_dir`: Target directory for traces and JSON metrics (default: `profile_results`).

> **Tip for quick testing**:
> ```bash
> python llama_inference_with_profiling.py --max_new_tokens 500 --profile_interval 100 --ignore_eos
> ```

---

### Step 2: Generate Interactive HTML Dashboard

Run [profile_visualizer.py](file:///home/ayushdeva_google_com/tiny_vllm/profile_visualizer.py) on the generated `token_metrics.json` file:

```bash
python profile_visualizer.py --json profile_results/token_metrics.json
```

#### Optional Flags:
- `--json`: Path to the input metrics JSON (default: `profile_results/token_metrics.json`).
- `--output`: Output path for the HTML dashboard (default: `profile_results/profile_dashboard.html`).
- `--terminal`: Also print detailed Rich terminal summary tables.

Open the resulting dashboard in any browser:
```text
profile_results/profile_dashboard.html
```

---

## Additional Tools & Evaluations

### Quality Evaluation Suite
Run [eval_quality.py](file:///home/ayushdeva_google_com/tiny_vllm/eval_quality.py) to measure perplexity and ensure accuracy parity:

```bash
python eval_quality.py
```

---

## Deep-Dive Documentation

For detailed analysis of GPU execution mechanics and memory systems, see the guides in `docs/`:
- [GPU Memory Wait & Scheduling FAQ](file:///home/ayushdeva_google_com/tiny_vllm/docs/GPU_MEMORY_WAIT_AND_SCHEDULING_FAQ.md)
- [Inference Architecture & Hardware Interplay](file:///home/ayushdeva_google_com/tiny_vllm/docs/INFERENCE_ARCHITECTURE_AND_HARDWARE_INTERPLAY.md)
- [Profiling Experiments & Systems Learnings](file:///home/ayushdeva_google_com/tiny_vllm/docs/PROFILING_EXPERIMENTS_AND_SYSTEMS_LEARNINGS.md)

