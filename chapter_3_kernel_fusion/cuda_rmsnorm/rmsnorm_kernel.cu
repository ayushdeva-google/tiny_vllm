#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ----------------------------------------------------------------------------
// 1. Warp and Block Reduction Primitives
// ----------------------------------------------------------------------------

// A "Warp" is a group of 32 threads executing in lockstep.
// We can use the __shfl_down_sync primitive to pass variables directly between 
// the registers of these 32 threads without ever touching VRAM or Shared Memory.
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// To reduce across an entire block (e.g., 1024 threads = 32 warps),
// we reduce each warp, write the 32 warp sums to Shared Memory, 
// and then use the first warp to reduce those 32 sums into a single value.
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // 1. Reduce within the warp
    val = warpReduceSum(val);

    // 2. The first thread of each warp writes its result to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    
    // Wait for all warps to finish writing to shared memory
    __syncthreads();

    // 3. The first warp reads from shared memory and does a final reduction
    // (If the block has less than 1024 threads, some shared memory slots are empty (0))
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0) {
        val = warpReduceSum(val);
    }

    return val;
}

// ----------------------------------------------------------------------------
// 2. The Fused RMSNorm CUDA Kernel
// ----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void rmsnorm_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    float eps,
    int N
) {
    // Each block processes exactly one row (one token)
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Shift pointers to the start of this block's assigned row
    const scalar_t* x = input + row * N;
    scalar_t* y = output + row * N;

    // Step 1: Compute local sum of squares
    // If N=4096 and blockDim=1024, each thread handles 4 elements.
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = static_cast<float>(x[i]);
        local_sum += val * val;
    }

    // Step 2: Block-wide reduction to get the total sum of squares for this row
    float total_sum = blockReduceSum(local_sum);

    // Step 3: Compute the reciprocal standard deviation (rsqrt)
    // Only thread 0 does the math, then we broadcast it to all threads via Shared Memory
    __shared__ float rstd;
    if (tid == 0) {
        rstd = rsqrtf((total_sum / N) + eps);
    }
    __syncthreads(); // Ensure all threads see the computed rstd

    // Step 4: Apply the normalization and the learned weight
    // Each thread writes its 4 elements back to VRAM
    for (int i = tid; i < N; i += blockDim.x) {
        float val = static_cast<float>(x[i]);
        float w = static_cast<float>(weight[i]);
        
        float out = (val * rstd) * w;
        y[i] = static_cast<scalar_t>(out);
    }
}

// ----------------------------------------------------------------------------
// 3. PyTorch C++ Dispatcher
// ----------------------------------------------------------------------------
torch::Tensor rmsnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    float eps
) {
    // 1. Allocate output tensor
    auto output = torch::empty_like(input);

    // 2. Determine grid and block dimensions
    // For 2D matrices, M is batch*seqlen (rows), N is hidden_dim (cols)
    int M = input.numel() / input.size(-1);
    int N = input.size(-1);

    int threads = 1024; // Max threads per block
    int blocks = M;     // 1 block per row

    // 3. Launch the kernel!
    // We use AT_DISPATCH_FLOATING_TYPES_AND2 to support float, double, float16, and bfloat16
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "rmsnorm_forward_cuda", ([&] {
        rmsnorm_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            eps,
            N
        );
    }));

    return output;
}

