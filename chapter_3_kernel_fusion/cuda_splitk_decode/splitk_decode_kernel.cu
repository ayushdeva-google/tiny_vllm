#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

__global__ void splitk_stage1(
    const float* Q, const float* K, const float* V,
    float* partial_O, float* partial_m, float* partial_l,
    int B, int H_q, int H_kv, int N, int d, float scale, int N_chunk) 
{
    int chunk_idx = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = blockDim.x / 32;
    
    int kv_h = h / (H_q / H_kv);
    int start_t = chunk_idx * N_chunk;
    int end_t = min(start_t + N_chunk, N);
    
    // Load Q into shared memory (64 floats)
    __shared__ float sh_q[64];
    if (tid < 64) {
        sh_q[tid] = Q[b * H_q * d + h * d + tid];
    }
    __syncthreads();
    
    float warp_m = -1e20f;
    float warp_l = 0.0f;
    float warp_O[2] = {0.0f, 0.0f}; // lane processes dim `lane` and `lane + 32`
    
    for (int t = start_t + wid; t < end_t; t += num_warps) {
        // Dot product Q * K_t
        float k0 = K[b * H_kv * N * d + kv_h * N * d + t * d + lane];
        float k1 = K[b * H_kv * N * d + kv_h * N * d + t * d + lane + 32];
        float dot = sh_q[lane] * k0 + sh_q[lane + 32] * k1;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        float score = __shfl_sync(0xffffffff, dot, 0) * scale;
        
        // Online Softmax
        float m_prev = warp_m;
        warp_m = max(warp_m, score);
        float exp_prev = expf(m_prev - warp_m);
        float exp_curr = expf(score - warp_m);
        
        warp_l = warp_l * exp_prev + exp_curr;
        
        // Accumulate V
        float v0 = V[b * H_kv * N * d + kv_h * N * d + t * d + lane];
        float v1 = V[b * H_kv * N * d + kv_h * N * d + t * d + lane + 32];
        warp_O[0] = warp_O[0] * exp_prev + exp_curr * v0;
        warp_O[1] = warp_O[1] * exp_prev + exp_curr * v1;
    }
    
    // Reduce warps in block
    __shared__ float sh_m[32];
    __shared__ float sh_l[32];
    __shared__ float sh_O[32][64];
    
    if (lane == 0) {
        sh_m[wid] = warp_m;
        sh_l[wid] = warp_l;
    }
    sh_O[wid][lane] = warp_O[0];
    sh_O[wid][lane + 32] = warp_O[1];
    __syncthreads();
    
    // Warp 0 finalizes the block chunk
    if (wid == 0) {
        float block_m = -1e20f;
        for (int i = 0; i < num_warps; ++i) {
            block_m = max(block_m, sh_m[i]);
        }
        
        float final_O0 = 0.0f;
        float final_O1 = 0.0f;
        float final_l = 0.0f;
        
        for (int i = 0; i < num_warps; ++i) {
            float factor = expf(sh_m[i] - block_m);
            if (lane == 0) final_l += sh_l[i] * factor;
            final_O0 += sh_O[i][lane] * factor;
            final_O1 += sh_O[i][lane + 32] * factor;
        }
        
        int out_idx = b * H_q * gridDim.x + h * gridDim.x + chunk_idx;
        partial_O[out_idx * d + lane] = final_O0;
        partial_O[out_idx * d + lane + 32] = final_O1;
        
        if (lane == 0) {
            partial_m[out_idx] = block_m;
            partial_l[out_idx] = final_l;
        }
    }
}

__global__ void splitk_stage2(
    const float* partial_O, const float* partial_m, const float* partial_l,
    float* final_O,
    int B, int H_q, int num_chunks, int d)
{
    int h = blockIdx.x;
    int b = blockIdx.y;
    int lane = threadIdx.x; // 0 to 63
    
    int base_idx = b * H_q * num_chunks + h * num_chunks;
    
    // Find global max
    float global_m = -1e20f;
    for (int i = 0; i < num_chunks; ++i) {
        global_m = max(global_m, partial_m[base_idx + i]);
    }
    
    // Find global sum and aggregate O
    float global_l = 0.0f;
    float my_O = 0.0f;
    for (int i = 0; i < num_chunks; ++i) {
        float factor = expf(partial_m[base_idx + i] - global_m);
        if (lane == 0) global_l += partial_l[base_idx + i] * factor;
        my_O += partial_O[(base_idx + i) * d + lane] * factor;
    }
    
    // Broadcast global_l from thread 0 to all threads
    global_l = __shfl_sync(0xffffffff, global_l, 0);
    
    // Write final normalized output
    final_O[b * H_q * d + h * d + lane] = my_O / global_l;
}

torch::Tensor splitk_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int N_chunk)
{
    auto q_f = q.to(torch::kFloat32).contiguous();
    auto k_f = k.to(torch::kFloat32).contiguous();
    auto v_f = v.to(torch::kFloat32).contiguous();

    int B = q_f.size(0);
    int H_q = q_f.size(1);
    int d = q_f.size(3);
    int H_kv = k_f.size(1);
    int N = k_f.size(2);
    
    float scale = 1.0f / sqrtf(d);
    int num_chunks = (N + N_chunk - 1) / N_chunk;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    torch::Tensor partial_O = torch::empty({B, H_q, num_chunks, d}, options);
    torch::Tensor partial_m = torch::empty({B, H_q, num_chunks}, options);
    torch::Tensor partial_l = torch::empty({B, H_q, num_chunks}, options);
    torch::Tensor final_O = torch::empty({B, H_q, d}, options);
    
    dim3 grid1(num_chunks, H_q, B);
    dim3 block1(128); // 4 warps
    splitk_stage1<<<grid1, block1>>>(
        q_f.data_ptr<float>(), k_f.data_ptr<float>(), v_f.data_ptr<float>(),
        partial_O.data_ptr<float>(), partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
        B, H_q, H_kv, N, d, scale, N_chunk
    );
    
    dim3 grid2(H_q, B);
    dim3 block2(64); // exactly d threads
    splitk_stage2<<<grid2, block2>>>(
        partial_O.data_ptr<float>(), partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
        final_O.data_ptr<float>(),
        B, H_q, num_chunks, d
    );
    
    return final_O.view({B, H_q, 1, d}).to(q.dtype());
}