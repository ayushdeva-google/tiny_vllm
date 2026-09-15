import torch
import torch.nn as nn
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# 1. Triton Kernel for Fused RMSNorm
# -----------------------------------------------------------------------------
@triton.jit
def _rmsnorm_fwd_fused(
    X_ptr,      # Pointer to input tensor
    Y_ptr,      # Pointer to output tensor
    W_ptr,      # Pointer to weight tensor
    stride_x,   # Stride of the input tensor (how many elements to skip to get to the next row)
    N,          # Hidden dimension (number of columns)
    eps,        # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr, # Number of elements per row to process in parallel
):
    """
    Triton kernel for forward RMSNorm.
    Each program (thread block) processes a single row of the input matrix.
    By doing all math in registers (SRAM), we only read X and write Y to VRAM once!
    """
    # 1. Identify which row this program instance is responsible for
    row_idx = tl.program_id(0)
    
    # 2. Advance the pointers to the start of this specific row
    X_row_ptr = X_ptr + row_idx * stride_x
    Y_row_ptr = Y_ptr + row_idx * stride_x

    # 3. Create a block of memory offsets [0, 1, 2, ..., BLOCK_SIZE - 1]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N  # Protect against out-of-bounds reads if N isn't a power of 2

    # 4. Load the input row and weight row from VRAM into fast SRAM (Registers)
    # We cast input to float32 for higher precision accumulation
    x = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # 5. Compute variance (mean of squares)
    variance = tl.sum(x * x, axis=0) / N
    
    # 6. Compute reciprocal standard deviation (rsqrt)
    rstd = 1.0 / tl.sqrt(variance + eps)

    # 7. Normalize the input and apply the learned weight parameter
    x_hat = x * rstd
    y = x_hat * w

    # 8. Write the final result back to VRAM
    # We cast it back to the original datatype of X (e.g., bfloat16)
    tl.store(Y_row_ptr + col_offsets, y.to(X_ptr.dtype.element_ty), mask=mask)


# -----------------------------------------------------------------------------
# 2. PyTorch Module Wrapper
# -----------------------------------------------------------------------------
class TritonRMSNorm(nn.Module):
    """
    Drop-in replacement for PyTorch RMSNorm, using our custom fused Triton kernel.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The kernel is designed for 2D inputs [M, N]. 
        # If input is [Batch, Seq, Dim], we flatten the first two dimensions.
        original_shape = x.shape
        x_2d = x.view(-1, original_shape[-1])
        M, N = x_2d.shape
        
        # Output buffer
        y_2d = torch.empty_like(x_2d)

        # Find the next power of 2 for BLOCK_SIZE to optimize memory alignment
        BLOCK_SIZE = triton.next_power_of_2(N)
        
        # Triton limit check (usually 64k elements per block max depending on GPU architecture)
        if BLOCK_SIZE > 65536:
            raise RuntimeError(f"Hidden size {N} too large for single block reduction.")

        # Grid defined as a 1D tuple. M programs total, one for each row.
        grid = (M,)

        # Launch the kernel asynchronously
        _rmsnorm_fwd_fused[grid](
            x_2d, y_2d, self.weight,
            x_2d.stride(0), N, self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # Use 4 warps (128 threads) per block
        )
        
        # Reshape back to [Batch, Seq, Dim]
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
        # Decomposes into 5 distinct VRAM roundtrips (pow, mean, add, rsqrt, mul)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# -----------------------------------------------------------------------------
# 4. Benchmark Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Setting up benchmark...")
    torch.manual_seed(42)
    
    # Simulating a decoding batch (Batch=32, Seq=1, Dim=4096)
    B, S, D = 32, 1, 4096 
    print(f"Input Shape: Batch={B}, SeqLen={S}, Dim={D}")
    
    x = torch.randn((B, S, D), device='cuda', dtype=torch.bfloat16)
    
    # Initialize both implementations
    triton_norm = TritonRMSNorm(D).cuda().bfloat16()
    torch_norm = NativePyTorchRMSNorm(D).cuda().bfloat16()
    
    # Validate correctness
    out_triton = triton_norm(x)
    out_torch = torch_norm(x)
    
    # Check max difference (should be extremely small, differences only due to fp32 casting order)
    max_diff = torch.max(torch.abs(out_triton - out_torch)).item()
    print(f"Correctness validation max difference: {max_diff:.6f}")
    assert max_diff < 5e-2, f"Outputs don't match! Max diff: {max_diff}"
    print("Correctness check passed!\\n")

    # Define benchmark parameters using Triton's testing tools
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['M'],  # x-axis for the plot (Rows/Tokens)
            x_vals=[128 * i for i in range(1, 21)], # 128 to 2560 rows
            line_arg='provider', 
            line_vals=['pytorch', 'triton'], 
            line_names=['PyTorch Native', 'Triton Fused'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='GB/s',
            plot_name='rmsnorm-performance',
            args={'N': D}  # Constant arguments
        )
    )
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        if provider == 'pytorch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_norm(x), quantiles=[0.5, 0.2, 0.8])
        elif provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_norm(x), quantiles=[0.5, 0.2, 0.8])
        
        # Calculate theoretical GB/s memory bandwidth
        # We read X (M*N elements) and Weight (N elements), and write Y (M*N elements)
        # We'll simplify to just X and Y transfer volume
        gb = 2 * x.numel() * x.element_size() / 1e9
        return gb / (ms / 1000), gb / (max_ms / 1000), gb / (min_ms / 1000)

    print("Running Triton Benchmark (Measuring Memory Bandwidth GB/s)...")
    benchmark.run(print_data=True, show_plots=False)
