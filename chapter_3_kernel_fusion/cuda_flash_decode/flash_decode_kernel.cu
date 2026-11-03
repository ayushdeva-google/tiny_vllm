#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

#define CHUNK_SIZE 128

// Utility for warp reduction of max
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Utility for warp reduction of sum
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Flash Decode CUDA Kernel
 * Grid:  1 block per Head
 * Block: 1024 threads (32 warps)
 */
__global__ void flash_decode_kernel(
    const float* __restrict__ q_ptr_base,
    const float* __restrict__ k_ptr_base,
    const float* __restrict__ v_ptr_base,
    float* __restrict__ out_ptr_base,
    int N,         // Sequence length
    int d,         // Head dimension (e.g. 64)
    float scale    // 1.0 / sqrt(d)
) {
    int head_idx = blockIdx.x;
    
    // Pointers for this specific head
    const float* q_ptr = q_ptr_base + head_idx * d;
    const float* k_ptr = k_ptr_base + head_idx * N * d;
    const float* v_ptr = v_ptr_base + head_idx * N * d;
    float* out_ptr     = out_ptr_base + head_idx * d;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Shared memory allocations
    __shared__ float x_p[CHUNK_SIZE];
    __shared__ float warp_maxes[32];

    // Running statistics in registers
    float global_max = -FLT_MAX;
    float global_sum = 0.0f;
    
    // Running output O in registers. 
    // Warp w computes Column w and Column w+32. 
    // Only lane 0 of the warp needs to hold the running totals for these columns!
    float my_O1 = 0.0f; // For column: warp_id
    float my_O2 = 0.0f; // For column: warp_id + 32

    // Load Q into shared memory for fast broadcast (optional, but good since d is small)
    __shared__ float q_smem[128]; // max d=128
    if (tid < d) {
        q_smem[tid] = q_ptr[tid];
    }
    __syncthreads();

    // Iterate over KV cache in chunks of CHUNK_SIZE
    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
        
        // -------------------------------------------------------------
        // Step 1: Q * K^T (Dot product)
        // 32 warps process 4 keys each (32 * 4 = 128 keys)
        // -------------------------------------------------------------
        for (int iter = 0; iter < 4; ++iter) {
            int key_idx = chunk_start + warp_id + iter * 32;
            float dot = 0.0f;
            
            if (key_idx < N) {
                // Thread loops over head dimension (d=64)
                for (int k_idx = lane_id; k_idx < d; k_idx += 32) {
                    dot += q_smem[k_idx] * k_ptr[key_idx * d + k_idx];
                }
            }
            
            // Warp reduction for the dot product
            dot = warpReduceSum(dot);
            
            // Lane 0 writes the result to shared memory x_p
            if (lane_id == 0) {
                if (key_idx < N) {
                    x_p[warp_id + iter * 32] = dot * scale;
                } else {
                    x_p[warp_id + iter * 32] = -FLT_MAX;
                }
            }
        }
        __syncthreads();

        // -------------------------------------------------------------
        // Step 2: Find chunk_max
        // 4 warps reduce 128 elements to 4 warp_maxes, then 1 warp reduces to chunk_max
        // -------------------------------------------------------------
        float local_val = -FLT_MAX;
        if (tid < CHUNK_SIZE) {
            local_val = x_p[tid];
        }
        
        float w_max = warpReduceMax(local_val);
        if (lane_id == 0 && warp_id < 4) {
            warp_maxes[warp_id] = w_max;
        }
        __syncthreads();
        
        float chunk_max = -FLT_MAX;
        if (warp_id == 0) {
            float val = (lane_id < 4) ? warp_maxes[lane_id] : -FLT_MAX;
            chunk_max = warpReduceMax(val);
            // Broadcast chunk_max to all threads in shared memory
            if (lane_id == 0) {
                warp_maxes[0] = chunk_max;
            }
        }
        __syncthreads();
        chunk_max = warp_maxes[0];

        // -------------------------------------------------------------
        // Step 3: Online Scaling factors
        // -------------------------------------------------------------
        float new_global_max = fmaxf(global_max, chunk_max);
        float scale_old = expf(global_max - new_global_max);
        float scale_new = expf(chunk_max - new_global_max);

        // Compute exponentiated scores and sum
        if (tid < CHUNK_SIZE) {
            if (chunk_start + tid < N) {
                x_p[tid] = expf(x_p[tid] - chunk_max);
            } else {
                x_p[tid] = 0.0f;
            }
        }
        __syncthreads();

        // Compute chunk_sum (128 threads do this)
        float chunk_sum = 0.0f;
        if (tid < CHUNK_SIZE) {
            chunk_sum = x_p[tid];
        }
        float w_sum = warpReduceSum(chunk_sum);
        
        // We can reuse warp_maxes array for summing
        if (lane_id == 0 && warp_id < 4) {
            warp_maxes[warp_id] = w_sum;
        }
        __syncthreads();
        
        chunk_sum = 0.0f;
        if (warp_id == 0) {
            float val = (lane_id < 4) ? warp_maxes[lane_id] : 0.0f;
            chunk_sum = warpReduceSum(val);
            if (lane_id == 0) {
                warp_maxes[0] = chunk_sum;
            }
        }
        __syncthreads();
        chunk_sum = warp_maxes[0];

        // -------------------------------------------------------------
        // Step 4: P * V (Numerator update)
        // Warp w computes Column w and Column w+32
        // -------------------------------------------------------------
        float chunk_O1 = 0.0f;
        float chunk_O2 = 0.0f;

        // Each warp processes its assigned columns over the 128 tokens
        // Lane ID processes elements lane_id, lane_id+32, lane_id+64, lane_id+96
        int col1 = warp_id;
        int col2 = warp_id + 32;

        for (int i = lane_id; i < CHUNK_SIZE; i += 32) {
            int key_idx = chunk_start + i;
            if (key_idx < N) {
                float p_val = x_p[i];
                if (col1 < d) {
                    chunk_O1 += p_val * v_ptr[key_idx * d + col1];
                }
                if (col2 < d) {
                    chunk_O2 += p_val * v_ptr[key_idx * d + col2];
                }
            }
        }

        chunk_O1 = warpReduceSum(chunk_O1);
        chunk_O2 = warpReduceSum(chunk_O2);

        // -------------------------------------------------------------
        // Step 5: Update Running Totals in Registers
        // -------------------------------------------------------------
        global_max = new_global_max;
        
        // All threads update the global sum (though only lane 0 actually needs it for final division)
        global_sum = global_sum * scale_old + chunk_sum * scale_new;
        
        // Lane 0 of each warp holds the running output for its columns
        if (lane_id == 0) {
            my_O1 = my_O1 * scale_old + chunk_O1 * scale_new;
            my_O2 = my_O2 * scale_old + chunk_O2 * scale_new;
        }
        __syncthreads();
    }

    // -------------------------------------------------------------
    // Step 6: Final Write to VRAM
    // -------------------------------------------------------------
    if (lane_id == 0) {
        if (warp_id < d) {
            out_ptr[warp_id] = my_O1 / global_sum;
        }
        if (warp_id + 32 < d) {
            out_ptr[warp_id + 32] = my_O2 / global_sum;
        }
    }
}

torch::Tensor flash_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    auto q_f = q.to(torch::kFloat32).contiguous();
    auto k_f = k.to(torch::kFloat32).contiguous();
    auto v_f = v.to(torch::kFloat32).contiguous();

    // Original shapes:
    // q: (B, n_heads, 1, d)
    // k: (B, n_heads, N, d)
    // v: (B, n_heads, N, d)
    int B = q_f.size(0);
    int n_heads = q_f.size(1);
    int N = k_f.size(2);
    int d = k_f.size(3);
    
    // Flatten batch and heads into a single block dimension
    int num_blocks = B * n_heads;

    float scale = 1.0f / std::sqrt((float)d);

    auto out = torch::zeros_like(q_f);

    int threads_per_block = 1024; // 32 warps
    dim3 grid(num_blocks);
    dim3 block(threads_per_block);

    flash_decode_kernel<<<grid, block>>>(
        q_f.data_ptr<float>(),
        k_f.data_ptr<float>(),
        v_f.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        d,
        scale
    );

    return out.to(q.dtype());
}
