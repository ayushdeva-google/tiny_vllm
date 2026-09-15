import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import triton

# -----------------------------------------------------------------------------
# 1. Compile and Load the Custom CUDA Kernel JIT
# -----------------------------------------------------------------------------
print("Compiling CUDA extension (this might take a few seconds on first run)...")
curr_dir = os.path.dirname(os.path.abspath(__file__))
cuda_rmsnorm_ext = load(
    name="cuda_rmsnorm_ext",
    sources=[
        os.path.join(curr_dir, "cuda_rmsnorm", "rmsnorm_extension.cpp"),
        os.path.join(curr_dir, "cuda_rmsnorm", "rmsnorm_kernel.cu"),
    ],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)
print("CUDA extension compiled successfully!\n")

# -----------------------------------------------------------------------------
# 2. PyTorch Wrapper for the CUDA Module
# -----------------------------------------------------------------------------
class CudaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.view(-1, original_shape[-1])
        # Call our custom C++/CUDA kernel
        y_2d = cuda_rmsnorm_ext.forward(x_2d, self.weight, self.eps)
        return y_2d.view(original_shape)

# -----------------------------------------------------------------------------
# 3. Native PyTorch Reference Implementation
# -----------------------------------------------------------------------------
class NativePyTorchRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# -----------------------------------------------------------------------------
# 4. Correctness & Benchmark
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Setting up benchmark...")
    torch.manual_seed(42)
    
    # Simulating a decoding batch (Batch=32, Seq=1, Dim=4096)
    B, S, D = 32, 1, 4096 
    print(f"Input Shape: Batch={B}, SeqLen={S}, Dim={D}")
    
    x = torch.randn((B, S, D), device='cuda', dtype=torch.bfloat16)
    
    # Initialize both implementations
    cuda_norm = CudaRMSNorm(D).cuda().bfloat16()
    torch_norm = NativePyTorchRMSNorm(D).cuda().bfloat16()
    
    # Validate correctness
    out_cuda = cuda_norm(x)
    out_torch = torch_norm(x)
    
    # Check max difference (relax tolerance for bfloat16 and different math order)
    max_diff = torch.max(torch.abs(out_cuda - out_torch)).item()
    print(f"Correctness validation max difference: {max_diff:.6f}")
    assert max_diff < 5e-2, f"Outputs don't match! Max diff: {max_diff}"
    print("Correctness check passed!\n")

    print("Running Custom CUDA vs PyTorch Native Benchmark (Measuring Memory Bandwidth GB/s)...")
    
    # We use Triton's perf_report just for plotting/benchmarking convenience
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['M'],  # x-axis for the plot (Rows/Tokens)
            x_vals=[128 * i for i in range(1, 21)], # 128 to 2560 rows
            line_arg='provider', 
            line_vals=['pytorch', 'cuda'], 
            line_names=['PyTorch Native', 'Custom CUDA Fused'],
            styles=[('blue', '-'), ('red', '-')],
            ylabel='GB/s',
            plot_name='cuda-rmsnorm-performance',
            args={'N': D}  # Constant arguments
        )
    )
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        if provider == 'pytorch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_norm(x), quantiles=[0.5, 0.2, 0.8])
        elif provider == 'cuda':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: cuda_norm(x), quantiles=[0.5, 0.2, 0.8])
        
        # Calculate theoretical GB/s memory bandwidth
        gb = 2 * x.numel() * x.element_size() / 1e9
        return gb / (ms / 1000), gb / (max_ms / 1000), gb / (min_ms / 1000)

    benchmark.run(print_data=True, show_plots=False)

