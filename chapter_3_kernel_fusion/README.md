# Chapter 3: Introduction to Kernel Fusion (Triton)

Welcome to Chapter 3! In this chapter, we transition from pure architecture (like the KV Cache) to **Systems Optimization**. Specifically, we tackle the "Host Launch Starvation" problem we discovered during profiling. 

We do this by writing our first custom GPU kernel using **OpenAI Triton** to perform **Kernel Fusion** on the `RMSNorm` operation.

---

## 1. What are Kernels?

A **Kernel** is a small, highly parallelized function designed specifically to execute on a GPU's compute units (Streaming Multiprocessors). When you write `x + y` in PyTorch on a CUDA tensor, the GPU doesn't literally read "x + y". Instead, PyTorch tells the CUDA Driver to launch an pre-compiled C++/CUDA function (a kernel) named something like `vector_add_kernel` that tells thousands of GPU cores how to load the arrays from memory and add them together.

## 2. Doesn't Python eventually map to CUDA anyway? Why write our own?

This is the most common question in Deep Learning systems. Yes, PyTorch provides thousands of highly optimized, pre-written CUDA kernels. When you write:
```python
# Standard PyTorch RMSNorm
variance = x.pow(2).mean(-1, keepdim=True)
x_normed = x * torch.rsqrt(variance + eps)
out = weight * x_normed
```
PyTorch's **Eager Mode** interprets this line by line and dispatches a sequence of pre-compiled CUDA kernels:
1. Launch `cuda_pow_kernel` (for `x.pow(2)`)
2. Launch `cuda_mean_kernel` (for `.mean()`)
3. Launch `cuda_add_kernel` (for `+ eps`)
4. Launch `cuda_rsqrt_kernel` (for `torch.rsqrt`)
5. Launch `cuda_mul_kernel` (for `* weight`)

### The Flaw: The Dispatch and Memory Bottlenecks
While each individual kernel is perfectly optimized by NVIDIA engineers, **chaining them together from Python is incredibly slow** for two reasons:

1. **Host Launch Starvation (CPU Dispatch Overhead)**: It takes the Python interpreter and the CUDA driver about **~15-30 microseconds (μs)** to prepare and launch a single kernel. However, an L4 GPU can finish doing the math for a small `add` kernel in **2 μs**. Because the CPU is much slower at launching kernels than the GPU is at executing them, the GPU command queue frequently empties out, leaving the GPU silicon **idling while waiting for the CPU**. 
2. **VRAM Roundtrips (The Memory Wall)**: Because these kernels are separate, they cannot communicate directly with each other. `cuda_pow_kernel` has to write its output to a temporary tensor in the GPU's Global VRAM (DRAM). Then `cuda_mean_kernel` has to read that temporary tensor all the way back from VRAM. **Reading and writing to VRAM is hundreds of times slower than doing math.**

## 3. What is Kernel Fusion?

**Kernel Fusion** is the practice of combining multiple mathematical operations into a single, custom GPU kernel. 

Instead of launching 5 separate kernels, we write **one** kernel that does everything.

### The Physics of Kernel Fusion
Modern GPUs have a memory hierarchy, much like a CPU:
- **Global Memory (VRAM)**: Huge (24GB), but "slow" to access (~300 GB/s bandwidth).
- **Registers / SRAM**: Extremely small per thread, but located physically next to the compute units, meaning access is instantaneous (~20,000 GB/s bandwidth).

In our fused kernel, we instruct the GPU to:
1. Load a row of `x` from VRAM into super-fast Registers.
2. Do **all** the math (`pow`, `mean`, `add`, `rsqrt`, `mul`) directly inside the Registers.
3. Write the final output `out` back to VRAM.

**Result**: We collapsed 5 VRAM read/writes down to exactly **1 VRAM read and 1 VRAM write**.

## 4. How We Implemented it (OpenAI Triton)

Writing raw CUDA (in C++) is notoriously difficult because you have to manually manage threads, warps, shared memory bank conflicts, and hardware synchronization.

Instead, we used **OpenAI Triton**, a Python-based language that acts as a compiler for custom GPU kernels. It gives you the performance of raw CUDA but the ergonomics of Python.

### Inside `triton_rmsnorm.py` (The High-Level Abstraction)
We wrote a Triton kernel decorated with `@triton.jit`. Here is the core logic translated to plain English:
1. `row_idx = tl.program_id(0)`: We launch thousands of "programs" in parallel. Each program is assigned exactly one row of the input matrix.
2. `x = tl.load(...)`: The program loads its assigned row from slow VRAM into fast SRAM.
3. `var = tl.sum(x * x) / N`: It computes the variance locally in SRAM.
4. `rstd = 1.0 / tl.sqrt(var + eps)`: Computes the reciprocal standard deviation locally.
5. `y = (x * rstd) * w`: Normalizes the vector and applies the learned weights locally.
6. `tl.store(...)`: Writes the final answer back to VRAM once.

### Diving Deeper: Inside `cuda_rmsnorm/` (The Hardware Reality)
Because our ultimate goal is to understand how the machine physically works, we also implemented the exact same kernel directly in raw C++ and CUDA. 

When you write in Triton, the compiler automatically handles memory coalescing and thread management. In our raw CUDA implementation (`rmsnorm_kernel.cu`), we have to manage the hardware manually:
1. **Warp Primitives**: We explicitly write `warpReduceSum` utilizing `__shfl_down_sync`. A "Warp" is a group of 32 threads. This instruction allows threads to pass variables directly between their registers without touching memory, reducing the sum across 32 threads in 5 clock cycles.
2. **Shared Memory**: We allocate `__shared__ float shared[32]` and explicitly synchronize our threads (`__syncthreads()`) so that the block of 1024 threads can combine their warp-level sums into a single global sum for the row.
3. **C++ Binding**: We wrote a pybind11 wrapper (`rmsnorm_extension.cpp`) to compile the C++ code and expose it directly to Python's PyTorch engine.

By writing the CUDA kernel directly, we learned exactly how block-level and warp-level reductions occur at the silicon level—skills that are mandatory for an ML Systems Engineer building engines like vLLM.

### The Results
When you run the benchmark (`python chapter_3_kernel_fusion/cuda_rmsnorm_benchmark.py`), you will see that:
- PyTorch Native achieves **~130 GB/s** memory bandwidth.
- Our Custom Fused CUDA Kernel achieves **~225 GB/s** memory bandwidth.

By eliminating intermediate VRAM roundtrips, we nearly doubled our memory bandwidth efficiency! We then integrated this directly into our inference engine, making the 33 `RMSNorm` calls per generated token much faster and significantly reducing CPU driver overhead.

