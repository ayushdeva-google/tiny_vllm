import os
import torch
from torch.utils.cpp_extension import load
import time
import math

os.makedirs('cuda_warp_decode', exist_ok=True)

with open('cuda_warp_decode/warp_decode.cu', 'w') as f:
    f.write("""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void splitk_decode_stage1_warp(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ partial_O,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int seq_len, int max_seq_len, int d, int num_heads_q, int num_heads_kv, int chunk_size
) {
    int chunk_idx = blockIdx.x;
    int head_idx_q = blockIdx.y;
    int batch_idx = blockIdx.z;
    int head_idx_kv = head_idx_q / (num_heads_q / num_heads_kv);
    int tx = threadIdx.x; // 0 to 127
    
    // 1. Broadcast Q to all 128 threads
    float q_reg[64];
    int q_offset = batch_idx * (num_heads_q * 1 * d) + head_idx_q * d;
    for(int k = 0; k < d; ++k) {
        q_reg[k] = Q[q_offset + k];
    }
    
    int token_idx = chunk_idx * chunk_size + tx;
    float scale = 1.0f / sqrtf((float)d);
    
    // 2. Compute dot product directly from Global Memory (Bypassing Shared Memory!)
    float dot = 0.0f;
    if (token_idx < seq_len) {
        int kv_offset_base = batch_idx * (num_heads_kv * max_seq_len * d) + head_idx_kv * (max_seq_len * d);
        int key_offset = kv_offset_base + token_idx * d;
        for(int k = 0; k < d; ++k) {
            dot += q_reg[k] * K_cache[key_offset + k];
        }
    }
    float score = (token_idx < seq_len) ? dot * scale : -1e20f;
    
    // 3. Block Reduction to find Max
    __shared__ float smem_reduce[128];
    smem_reduce[tx] = score;
    __syncthreads();
    
    for (int stride = 64; stride > 0; stride /= 2) {
        if (tx < stride) {
            smem_reduce[tx] = max(smem_reduce[tx], smem_reduce[tx + stride]);
        }
        __syncthreads();
    }
    float m_i = smem_reduce[0];
    
    // 4. Block Reduction to find Sum
    float p = (token_idx < seq_len) ? expf(score - m_i) : 0.0f;
    smem_reduce[tx] = p;
    __syncthreads();
    
    for (int stride = 64; stride > 0; stride /= 2) {
        if (tx < stride) {
            smem_reduce[tx] += smem_reduce[tx + stride];
        }
        __syncthreads();
    }
    float l_i = smem_reduce[0];
    
    // 5. Weight V directly from Global Memory and Transpose in Shared Memory
    __shared__ float smem_V_weighted[128][64];
    if (token_idx < seq_len) {
        int kv_offset_base = batch_idx * (num_heads_kv * max_seq_len * d) + head_idx_kv * (max_seq_len * d);
        int val_offset = kv_offset_base + token_idx * d;
        for(int k = 0; k < d; ++k) {
            smem_V_weighted[tx][k] = p * V_cache[val_offset + k];
        }
    } else {
        for(int k = 0; k < d; ++k) {
            smem_V_weighted[tx][k] = 0.0f;
        }
    }
    __syncthreads();
    
    // 6. Threads 0..63 sum their assigned dimension across the 128 tokens
    if (tx < d) {
        float o_sum = 0.0f;
        for(int i = 0; i < 128; ++i) {
            o_sum += smem_V_weighted[i][tx];
        }
        
        // Write to Partial_O
        if (tx == 0) { // Thread 0 writes the scalar stats
            int out_idx = batch_idx * (num_heads_q * gridDim.x) + head_idx_q * gridDim.x + chunk_idx;
            partial_m[out_idx] = m_i;
            partial_l[out_idx] = l_i;
        }
        int out_idx = batch_idx * (num_heads_q * gridDim.x) + head_idx_q * gridDim.x + chunk_idx;
        partial_O[out_idx * d + tx] = o_sum;
    }
}

__global__ void splitk_decode_stage2(
    const float* __restrict__ partial_O,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    float* __restrict__ O,
    int num_chunks, int d, int num_heads_q
) {
    int head_idx_q = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tx = threadIdx.x;
    if (tx >= d) return;
    
    float m_global = -1e20f;
    float l_global = 0.0f;
    float o_global = 0.0f;
    
    int base_idx = batch_idx * (num_heads_q * num_chunks) + head_idx_q * num_chunks;
    for (int i = 0; i < num_chunks; ++i) {
        float m_i = partial_m[base_idx + i];
        if (m_i > -1e19f && m_i > m_global) m_global = m_i;
    }
    
    for (int i = 0; i < num_chunks; ++i) {
        float m_i = partial_m[base_idx + i];
        float l_i = partial_l[base_idx + i];
        if (m_i > -1e19f) {
            float exp_diff = expf(m_i - m_global);
            l_global += l_i * exp_diff;
            o_global += partial_O[(base_idx + i) * d + tx] * exp_diff;
        }
    }
    
    int out_idx = batch_idx * (num_heads_q * 1 * d) + head_idx_q * d + tx;
    O[out_idx] = o_global / l_global;
}

void run_decode_cuda(torch::Tensor Q, torch::Tensor K_cache, torch::Tensor V_cache, torch::Tensor O, int seq_len) {
    int B = Q.size(0);
    int H_q = Q.size(1);
    int d = Q.size(3);
    int H_kv = K_cache.size(1);
    int max_seq_len = K_cache.size(2);
    
    int chunk_size = 128;
    int num_chunks = (seq_len + chunk_size - 1) / chunk_size;
    
    auto partial_O = torch::zeros({B, H_q, num_chunks, d}, Q.options());
    auto partial_m = torch::empty({B, H_q, num_chunks}, Q.options());
    auto partial_l = torch::empty({B, H_q, num_chunks}, Q.options());
    
    dim3 grid1(num_chunks, H_q, B);
    dim3 block1(128); // Exactly 128 threads
    splitk_decode_stage1_warp<<<grid1, block1>>>(
        Q.data_ptr<float>(), K_cache.data_ptr<float>(), V_cache.data_ptr<float>(),
        partial_O.data_ptr<float>(), partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
        seq_len, max_seq_len, d, H_q, H_kv, chunk_size
    );
    
    dim3 grid2(H_q, B);
    dim3 block2(64);
    splitk_decode_stage2<<<grid2, block2>>>(
        partial_O.data_ptr<float>(), partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
        O.data_ptr<float>(), num_chunks, d, H_q
    );
}
""")

with open('cuda_warp_decode/warp_extension.cpp', 'w') as f:
    f.write("""
#include <torch/extension.h>

void run_decode_cuda(torch::Tensor Q, torch::Tensor K_cache, torch::Tensor V_cache, torch::Tensor O, int seq_len);

torch::Tensor forward(torch::Tensor Q, torch::Tensor K_cache, torch::Tensor V_cache, int seq_len) {
    auto O = torch::zeros_like(Q);
    run_decode_cuda(Q, K_cache, V_cache, O, seq_len);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp Optimized Decode");
}
""")

print("[*] Compiling Warp Optimized Decode...")
warp_ext = load(
    name='cuda_warp_decode_ext',
    sources=['cuda_warp_decode/warp_extension.cpp', 'cuda_warp_decode/warp_decode.cu'],
    verbose=False
)

B, H_q, H_kv, d = 1, 32, 8, 64
PREFILL_TOKENS = 8192
DECODE_TOKENS = 32
MAX_SEQ_LEN = PREFILL_TOKENS + DECODE_TOKENS

def native_pytorch_attention(q, k_cache, v_cache, seq_len):
    k_expanded = k_cache.repeat_interleave(H_q // H_kv, dim=1)
    v_expanded = v_cache.repeat_interleave(H_q // H_kv, dim=1)
    k = k_expanded[:, :, :seq_len, :]
    v = v_expanded[:, :, :seq_len, :]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)

Q_decode = torch.randn(B, H_q, 1, d, device='cuda', dtype=torch.float32)
K_cache = torch.randn(B, H_kv, MAX_SEQ_LEN, d, device='cuda', dtype=torch.float32)
V_cache = torch.randn(B, H_kv, MAX_SEQ_LEN, d, device='cuda', dtype=torch.float32)

# Warmup
for _ in range(5):
    _ = native_pytorch_attention(Q_decode, K_cache, V_cache, PREFILL_TOKENS)
    _ = warp_ext.forward(Q_decode, K_cache, V_cache, PREFILL_TOKENS)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("\n--- DECODE PHASE (32 Tokens @ Context 8192) ---")

# PyTorch
pt_decode_total = 0
for step in range(DECODE_TOKENS):
    current_seq_len = PREFILL_TOKENS + step
    Q_decode = torch.randn(B, H_q, 1, d, device='cuda', dtype=torch.float32)
    start.record()
    _ = native_pytorch_attention(Q_decode, K_cache, V_cache, current_seq_len)
    end.record()
    torch.cuda.synchronize()
    pt_decode_total += start.elapsed_time(end)
print(f"PyTorch Decode         : {pt_decode_total:.2f} ms")

# Warp Decode
warp_decode_total = 0
for step in range(DECODE_TOKENS):
    current_seq_len = PREFILL_TOKENS + step
    Q_decode = torch.randn(B, H_q, 1, d, device='cuda', dtype=torch.float32)
    start.record()
    _ = warp_ext.forward(Q_decode, K_cache, V_cache, current_seq_len)
    end.record()
    torch.cuda.synchronize()
    warp_decode_total += start.elapsed_time(end)
print(f"Warp-Optimized Decode  : {warp_decode_total:.2f} ms")
print(f"Speedup vs PyTorch     : {pt_decode_total / warp_decode_total:.2f}x")
