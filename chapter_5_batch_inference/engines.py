"""
Serving Engine Abstraction & Sequential FCFS (B=1) Baseline for Chapter 5.

Provides:
1. `BaseServingEngine`: Common interface shared by Sequential FCFS (`B=1`) and upcoming
   batching engines (Static Batching, Continuous Batching, Chunked Prefill, PagedAttention).
2. `SequentialFCFSEngine`: Processes incoming requests strictly one by one (`batch_size=1`)
   in First-Come-First-Served (FCFS) arrival order using a pre-allocated static `KVCache`.
   Supports:
   - `event-driven` clock (default): Fast-forwards pure GPU-idle periods when the arrival queue
     is empty so low-QPS sweeps don't waste wall-clock time sleeping, while using 100% real
     synchronized GPU execution timings whenever >= 1 request is active.
   - `real-time` clock (`--real-time`): Sleeps until each request's actual wall-clock arrival offset.
"""

import abc
import os
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import torch
from transformers import AutoTokenizer

try:
    from .llama_model import (
        KVCache,
        NativeLlamaTransformer,
        get_stop_token_ids,
        sample_next_tokens,
    )
    from .profiler_metrics import RequestExecutionRecord, analyze_chrome_trace
    from .workloads import InferenceRequest
except ImportError:
    from llama_model import (
        KVCache,
        NativeLlamaTransformer,
        get_stop_token_ids,
        sample_next_tokens,
    )
    from profiler_metrics import RequestExecutionRecord, analyze_chrome_trace
    from workloads import InferenceRequest


@dataclass
class EngineRunOutput:
    """Raw output from executing a stream of incoming requests on a serving engine."""

    records: List[RequestExecutionRecord]
    wall_clock_exec_s: float
    gpu_idle_time_s: float
    peak_vram_allocated_mb: float
    peak_vram_reserved_mb: float


class BaseServingEngine(abc.ABC):
    """Abstract base class for all Chapter 5 serving engines."""

    name: str = "base"

    def __init__(
        self,
        model: NativeLlamaTransformer,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 2048,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.stop_ids = get_stop_token_ids(tokenizer)

    @abc.abstractmethod
    def get_kv_allocated_mb(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def get_kv_bytes_per_token(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def warmup(self, include_profiler: bool = True) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def serve_requests(
        self,
        requests: List[InferenceRequest],
        real_time: bool = False,
        ignore_eos: bool = True,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
    ) -> EngineRunOutput:
        raise NotImplementedError


class SequentialFCFSEngine(BaseServingEngine):
    """
    Baseline Serving Engine: First-Come-First-Served (FCFS) with `batch_size = 1`.

    Systems Characteristics:
    - Head-of-Line (HoL) Blocking: Every newly arrived request sits in the FIFO queue until all
      earlier requests finish BOTH prefill and their entire autoregressive decode loop.
    - Memory-Bandwidth Bound Decode (GEMV): Every decode step streams all 2.47 GB of BF16 model
      weights from HBM to produce just 1 token (arithmetic intensity ~ 1 FLOP/byte).
    - Static KV Reservation: Pre-allocates `max_seq_len` capacity up front for `batch_size = 1`.
    """

    name: str = "sequential_fcfs_b1"

    def __init__(
        self,
        model: NativeLlamaTransformer,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 2048,
        device: str = "cuda",
    ):
        super().__init__(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len, device=device)
        self.kv_cache = KVCache(
            n_layers=model.args.n_layers,
            max_batch_size=1,
            max_seq_len=max_seq_len,
            n_kv_heads=model.args.n_kv_heads,
            head_dim=model.args.head_dim,
            dtype=self.dtype,
            device=device,
        )

    def get_kv_allocated_mb(self) -> float:
        return self.kv_cache.get_allocated_mb()

    def get_kv_bytes_per_token(self) -> int:
        return self.kv_cache.bytes_per_token()

    @torch.inference_mode()
    def warmup(self, include_profiler: bool = True) -> None:
        """Warms up cuBLAS GEMM/GEMV heuristics across prompt scales, allocator caches, and CUPTI."""
        self.model.set_profile_ops(False)
        for p_len in (64, 256, min(512, self.max_seq_len - 8)):
            dummy_prompt = torch.ones((1, p_len), dtype=torch.long, device=self.device)
            logits = self.model(dummy_prompt, start_pos=0, kv_cache=self.kv_cache)
            next_tok = sample_next_tokens(logits)
            _ = int(next_tok.item())
            logits_dec = self.model(next_tok, start_pos=p_len, kv_cache=self.kv_cache)
            _ = int(sample_next_tokens(logits_dec).item())

        if self.device == "cuda":
            torch.cuda.synchronize()

        if include_profiler and self.device == "cuda":
            self.model.set_profile_ops(True)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
            ):
                logits = self.model(dummy_prompt[:, :16], start_pos=0, kv_cache=self.kv_cache)
                _ = int(sample_next_tokens(logits, profile_ops=True).item())
            torch.cuda.synchronize()
            self.model.set_profile_ops(False)

    @torch.inference_mode()
    def serve_requests(
        self,
        requests: List[InferenceRequest],
        real_time: bool = False,
        ignore_eos: bool = True,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
    ) -> EngineRunOutput:
        """Executes incoming requests sequentially (B=1) in FCFS order."""
        self.model.set_profile_ops(False)
        sorted_reqs = sorted(requests, key=lambda r: (r.arrival_time_s, r.request_id))
        n_total = len(sorted_reqs)

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        ready_queue: Deque[InferenceRequest] = deque()
        records: List[RequestExecutionRecord] = []
        next_idx = 0
        sim_clock_s = 0.0
        gpu_idle_time_s = 0.0

        wall_t0 = time.perf_counter()

        def wall_now_s() -> float:
            return time.perf_counter() - wall_t0

        while len(records) < n_total:
            if not real_time:
                # 1. Admit all requests that have arrived up to current virtual clock
                while next_idx < n_total and sorted_reqs[next_idx].arrival_time_s <= sim_clock_s:
                    ready_queue.append(sorted_reqs[next_idx])
                    next_idx += 1

                # 2. If server is idle, fast-forward virtual clock to next incoming arrival
                if not ready_queue and next_idx < n_total:
                    next_arr = sorted_reqs[next_idx].arrival_time_s
                    if next_arr > sim_clock_s:
                        gpu_idle_time_s += next_arr - sim_clock_s
                        sim_clock_s = next_arr
                    while next_idx < n_total and sorted_reqs[next_idx].arrival_time_s <= sim_clock_s:
                        ready_queue.append(sorted_reqs[next_idx])
                        next_idx += 1

                req = ready_queue.popleft()
                dispatch_time_s = sim_clock_s

                # --- Step 0: Prefill ---
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t_step0 = time.perf_counter()
                prompt_tensor = torch.tensor([req.prompt_token_ids], dtype=torch.long, device=self.device)
                logits = self.model(prompt_tensor, start_pos=0, kv_cache=self.kv_cache)
                next_tok = sample_next_tokens(logits, temperature=temperature, top_k=top_k)
                tok_id = int(next_tok.item())  # Stream syncs on DtoH scalar transfer
                t_step1 = time.perf_counter()

                sim_clock_s += t_step1 - t_step0
                first_token_time_s = sim_clock_s
                token_times_s = [first_token_time_s]
                generated_ids = [tok_id]

                # --- Steps 1..M-1: Autoregressive Decode ---
                for step in range(1, req.target_output_len):
                    if not ignore_eos and tok_id in self.stop_ids:
                        break
                    t_d0 = time.perf_counter()
                    logits = self.model(
                        next_tok,
                        start_pos=req.prompt_len + step - 1,
                        kv_cache=self.kv_cache,
                    )
                    next_tok = sample_next_tokens(logits, temperature=temperature, top_k=top_k)
                    tok_id = int(next_tok.item())
                    t_d1 = time.perf_counter()

                    sim_clock_s += t_d1 - t_d0
                    token_times_s.append(sim_clock_s)
                    generated_ids.append(tok_id)

                finish_time_s = sim_clock_s

            else:
                # Real wall-clock arrival replay
                curr_now = wall_now_s()
                while next_idx < n_total and sorted_reqs[next_idx].arrival_time_s <= curr_now:
                    ready_queue.append(sorted_reqs[next_idx])
                    next_idx += 1

                if not ready_queue and next_idx < n_total:
                    target_arr = sorted_reqs[next_idx].arrival_time_s
                    idle_start = wall_now_s()
                    rem = target_arr - idle_start
                    if rem > 0.002:
                        time.sleep(rem - 0.001)
                    while wall_now_s() < target_arr:
                        pass
                    curr_now = wall_now_s()
                    gpu_idle_time_s += max(0.0, curr_now - idle_start)
                    while next_idx < n_total and sorted_reqs[next_idx].arrival_time_s <= curr_now:
                        ready_queue.append(sorted_reqs[next_idx])
                        next_idx += 1

                req = ready_queue.popleft()
                if self.device == "cuda":
                    torch.cuda.synchronize()
                dispatch_time_s = wall_now_s()

                # --- Step 0: Prefill ---
                prompt_tensor = torch.tensor([req.prompt_token_ids], dtype=torch.long, device=self.device)
                logits = self.model(prompt_tensor, start_pos=0, kv_cache=self.kv_cache)
                next_tok = sample_next_tokens(logits, temperature=temperature, top_k=top_k)
                tok_id = int(next_tok.item())
                first_token_time_s = wall_now_s()
                token_times_s = [first_token_time_s]
                generated_ids = [tok_id]

                # --- Steps 1..M-1: Autoregressive Decode ---
                for step in range(1, req.target_output_len):
                    if not ignore_eos and tok_id in self.stop_ids:
                        break
                    logits = self.model(
                        next_tok,
                        start_pos=req.prompt_len + step - 1,
                        kv_cache=self.kv_cache,
                    )
                    next_tok = sample_next_tokens(logits, temperature=temperature, top_k=top_k)
                    tok_id = int(next_tok.item())
                    token_times_s.append(wall_now_s())
                    generated_ids.append(tok_id)

                finish_time_s = token_times_s[-1]

            records.append(
                RequestExecutionRecord(
                    request_id=req.request_id,
                    source_id=req.source_id,
                    prompt_len=req.prompt_len,
                    target_output_len=req.target_output_len,
                    output_len=len(generated_ids),
                    arrival_time_s=req.arrival_time_s,
                    dispatch_time_s=dispatch_time_s,
                    first_token_time_s=first_token_time_s,
                    finish_time_s=finish_time_s,
                    token_times_s=token_times_s,
                    generated_token_ids=generated_ids,
                )
            )

        wall_exec_s = time.perf_counter() - wall_t0
        if self.device == "cuda":
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            peak_res_mb = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
        else:
            peak_alloc_mb, peak_res_mb = 0.0, 0.0

        # Restore original request_id ordering for deterministic downstream diffs
        records.sort(key=lambda r: r.request_id)
        return EngineRunOutput(
            records=records,
            wall_clock_exec_s=wall_exec_s,
            gpu_idle_time_s=gpu_idle_time_s,
            peak_vram_allocated_mb=peak_alloc_mb,
            peak_vram_reserved_mb=peak_res_mb,
        )

    @torch.inference_mode()
    def profile_representative_phases(
        self,
        prefill_seq_len: int = 128,
        decode_context_len: int = 256,
        traces_dir: str = "chapter_5_batch_inference/profile_results/traces",
        native_trials: int = 5,
    ) -> Dict[str, Any]:
        """
        Profiles representative Prefill and Decode steps in isolation:
        1. Measures un-instrumented median wall time (`native_wall_ms`) over `native_trials` runs.
        2. Captures `torch.profiler` Chrome traces with `record_function` scopes enabled to
           decompose GPU kernel active time vs. host CPU dispatch gap and operator shares.
        """
        os.makedirs(traces_dir, exist_ok=True)
        prefill_seq_len = min(max(8, prefill_seq_len), self.max_seq_len - 16)
        decode_context_len = min(max(prefill_seq_len + 1, decode_context_len), self.max_seq_len - 4)

        prompt_ids = torch.ones((1, prefill_seq_len), dtype=torch.long, device=self.device)
        decode_tok = torch.ones((1, 1), dtype=torch.long, device=self.device)

        # Seed KV cache up to decode_context_len - 1
        self.model.set_profile_ops(False)
        seed_ids = torch.ones((1, decode_context_len - 1), dtype=torch.long, device=self.device)
        _ = self.model(seed_ids, start_pos=0, kv_cache=self.kv_cache)
        if self.device == "cuda":
            torch.cuda.synchronize()

        # 1. Measure native un-profiled step wall latencies
        prefill_walls: List[float] = []
        for _ in range(native_trials):
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = self.model(prompt_ids, start_pos=0, kv_cache=self.kv_cache)
            _ = int(sample_next_tokens(logits).item())
            prefill_walls.append((time.perf_counter() - t0) * 1000.0)
        native_prefill_ms = statistics.median(prefill_walls)

        # Re-seed up to decode_context_len - 1 for decode timing
        _ = self.model(seed_ids, start_pos=0, kv_cache=self.kv_cache)
        if self.device == "cuda":
            torch.cuda.synchronize()

        decode_walls: List[float] = []
        for _ in range(native_trials):
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = self.model(decode_tok, start_pos=decode_context_len - 1, kv_cache=self.kv_cache)
            _ = int(sample_next_tokens(logits).item())
            decode_walls.append((time.perf_counter() - t0) * 1000.0)
        native_decode_ms = statistics.median(decode_walls)

        # 2. Capture PyTorch Profiler Chrome traces
        self.model.set_profile_ops(True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        prefill_trace = os.path.join(traces_dir, "b1_prefill_trace.json")
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof_pre:
            with torch.profiler.record_function("step_prefill"):
                logits = self.model(prompt_ids, start_pos=0, kv_cache=self.kv_cache)
                _ = int(sample_next_tokens(logits, profile_ops=True).item())
        if self.device == "cuda":
            torch.cuda.synchronize()
        prof_pre.export_chrome_trace(prefill_trace)

        # Re-seed KV context and profile Decode step
        self.model.set_profile_ops(False)
        _ = self.model(seed_ids, start_pos=0, kv_cache=self.kv_cache)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.model.set_profile_ops(True)

        decode_trace = os.path.join(traces_dir, "b1_decode_trace.json")
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof_dec:
            with torch.profiler.record_function("step_decode"):
                logits = self.model(decode_tok, start_pos=decode_context_len - 1, kv_cache=self.kv_cache)
                _ = int(sample_next_tokens(logits, profile_ops=True).item())
        if self.device == "cuda":
            torch.cuda.synchronize()
        prof_dec.export_chrome_trace(decode_trace)

        self.model.set_profile_ops(False)

        return {
            "prefill_seq_len": prefill_seq_len,
            "decode_context_len": decode_context_len,
            "prefill": analyze_chrome_trace(prefill_trace, "step_prefill", native_wall_ms=native_prefill_ms),
            "decode": analyze_chrome_trace(decode_trace, "step_decode", native_wall_ms=native_decode_ms),
        }
