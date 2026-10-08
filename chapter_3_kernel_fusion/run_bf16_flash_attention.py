import os
import torch
from torch.utils.cpp_extension import load
import math
import time

os.makedirs('cuda_bf16_flash_attention', exist_ok=True)

with open('cuda_bf16_flash_attention/bf16_flash_attn_kernel.cu', 'w') as f:
    f.write("""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

__global__ void prefill_kernel_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    int q_len, int max_seq_len, int d, int num_heads_q, int num_heads_kv
) {
    int q_chunk_idx = blockIdx.x;
    int head_idx_q  = blockIdx.y;
    int batch_idx   = blockIdx.z;
    int head_idx_kv = head_idx_q / (num_heads_q / num_heads_kv);
    int tx = threadIdx.x; // 0 to 127
    int q_global_idx = q_chunk_idx * 128 + tx;
    
    float q_reg[64];
    float o_reg[64] = {0.0f};
    float m_prev = -1e20f;
    float l_prev = 0.0f;
    
    int q_offset = batch_idx * (num_heads_q * q_len * d) + head_idx_q * (q_len * d) + q_global_idx * d;
    if (q_global_idx < q_len) {
        for(int k = 0; k < d; ++k) {
            q_reg[k] = __bfloat162float(Q[q_offset + k]);
        }
    }
    
    // 32 KB Shared Memory buffer
    __shared__ float smem[128][64];
    float scale = 1.0f / sqrtf((float)d);
    
    for (int k_chunk_idx = 0; k_chunk_idx <= q_chunk_idx; ++k_chunk_idx) {
        int k_global_start = k_chunk_idx * 128;
        int k_global_idx = k_global_start + tx;
        int kv_offset = batch_idx * (num_heads_kv * max_seq_len * d) + head_idx_kv * (max_seq_len * d) + k_global_idx * d;
        
        // 1. Load K into shared memory & upcast on the fly
        if (k_global_idx < q_len) {
            for(int k = 0; k < d; ++k) {
                smem[tx][k] = __bfloat162float(K_cache[kv_offset + k]);
            }
        }
        __syncthreads();
        
        // 2. Compute attention scores
        float scores[128];
        float m_curr = m_prev;
        if (q_global_idx < q_len) {
            for (int c = 0; c < 128; ++c) {
                int key_global_idx = k_global_start + c;
                if (key_global_idx > q_global_idx || key_global_idx >= q_len) {
                    scores[c] = -1e20f;
                } else {
                    float dot = 0.0f;
                    for (int k = 0; k < d; ++k) {
                        dot += q_reg[k] * smem[c][k];
                    }
                    scores[c] = dot * scale;
                    if (scores[c] > m_curr) m_curr = scores[c];
                }
            }
        }
        __syncthreads(); // Done with K
        
        // 3. Load V into same shared memory buffer
        if (k_global_idx < q_len) {
            for(int k = 0; k < d; ++k) {
                smem[tx][k] = __bfloat162float(V_cache[kv_offset + k]);
            }
        }
        __syncthreads();
        
        // 4. Online Softmax update
        if (q_global_idx < q_len) {
            float exp_diff = expf(m_prev - m_curr);
            float l_curr = l_prev * exp_diff;
            
            for (int c = 0; c < 128; ++c) {
                int key_global_idx = k_global_start + c;
                if (key_global_idx <= q_global_idx && key_global_idx < q_len) {
                    float p = expf(scores[c] - m_curr);
                    l_curr += p;
                    scores[c] = p;
                }
            }
            
            for (int k = 0; k < d; ++k) {
                o_reg[k] *= exp_diff;
                for (int c = 0; c < 128; ++c) {
                    int key_global_idx = k_global_start + c;
                    if (key_global_idx <= q_global_idx && key_global_idx < q_len) {
                        o_reg[k] += scores[c] * smem[c][k];
                    }
                }
            }
            m_prev = m_curr;
            l_prev = l_curr;
        }
        __syncthreads(); // Done with V
    }
    
    // Write out native bfloat16
    if (q_global_idx < q_len) {
        int o_offset = batch_idx * (num_heads_q * q_len * d) + head_idx_q * (q_len * d) + q_global_idx * d;
        for(int k = 0; k < d; ++k) {
            O[o_offset + k] = __float2bfloat16(o_reg[k] / l_prev);
        }
    }
}

__global__ void splitk_decode_stage1_warp_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
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
    
    float q_reg[64];
    int q_offset = batch_idx * (num_heads_q * 1 * d) + head_idx_q * d;
    for(int k = 0; k < d; ++k) {
        q_reg[k] = __bfloat162float(Q[q_offset + k]);
    }
    
    int token_idx = chunk_idx * chunk_size + tx;
    float scale = 1.0f / sqrtf((float)d);
    
    // 1. Direct memory dot product (Bypassing Shared Memory!)
    float dot = 0.0f;
    if (token_idx < seq_len) {
        int kv_offset_base = batch_idx * (num_heads_kv * max_seq_len * d) + head_idx_kv * (max_seq_len * d);
        int key_offset = kv_offset_base + token_idx * d;
        for(int k = 0; k < d; ++k) {
            dot += q_reg[k] * __bfloat162float(K_cache[key_offset + k]);
        }
    }
    float score = (token_idx < seq_len) ? dot * scale : -1e20f;
    
    // 2. Tree reduction for max
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
    
    // 3. Tree reduction for sum
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
    
    // 4. Weight V directly from Global Memory
    __shared__ float smem_V_weighted[128][64];
    if (token_idx < seq_len) {
        int kv_offset_base = batch_idx * (num_heads_kv * max_seq_len * d) + head_idx_kv * (max_seq_len * d);
        int val_offset = kv_offset_base + token_idx * d;
        for(int k = 0; k < d; ++k) {
            smem_V_weighted[tx][k] = p * __bfloat162float(V_cache[val_offset + k]);
        }
    } else {
        for(int k = 0; k < d; ++k) {
            smem_V_weighted[tx][k] = 0.0f;
        }
    }
    __syncthreads();
    
    // 5. Threads 0..63 sum their assigned dimension
    if (tx < d) {
        float o_sum = 0.0f;
        for(int i = 0; i < 128; ++i) {
            o_sum += smem_V_weighted[i][tx];
        }
        
        if (tx == 0) {
            int out_idx = batch_idx * (num_heads_q * gridDim.x) + head_idx_q * gridDim.x + chunk_idx;
            partial_m[out_idx] = m_i;
            partial_l[out_idx] = l_i;
        }
        int out_idx = batch_idx * (num_heads_q * gridDim.x) + head_idx_q * gridDim.x + chunk_idx;
        partial_O[out_idx * d + tx] = o_sum;
    }
}

__global__ void splitk_decode_stage2_bf16(
    const float* __restrict__ partial_O,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ O,
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
    O[out_idx] = __float2bfloat16(o_global / l_global);
}

void run_forward_cuda_bf16(
    torch::Tensor Q, torch::Tensor K_cache, torch::Tensor V_cache, torch::Tensor O, int seq_len
) {
    int B = Q.size(0);
    int H_q = Q.size(1);
    int q_len = Q.size(2);
    int d = Q.size(3);
    int H_kv = K_cache.size(1);
    int max_seq_len = K_cache.size(2);
    
    if (q_len > 1) { // PREFILL
        int num_chunks = (q_len + 127) / 128;
        dim3 grid(num_chunks, H_q, B);
        dim3 block(128);
        prefill_kernel_bf16<<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(K_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(V_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(O.data_ptr<at::BFloat16>()),
            q_len, max_seq_len, d, H_q, H_kv
        );
    } else { // DECODE
        int chunk_size = 128;
        int num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        
        auto partial_O = torch::zeros({B, H_q, num_chunks, d}, torch::dtype(torch::kFloat32).device(Q.device()));
        auto partial_m = torch::empty({B, H_q, num_chunks}, torch::dtype(torch::kFloat32).device(Q.device()));
        auto partial_l = torch::empty({B, H_q, num_chunks}, torch::dtype(torch::kFloat32).device(Q.device()));
        
        dim3 grid1(num_chunks, H_q, B);
        dim3 block1(128);
        splitk_decode_stage1_warp_bf16<<<grid1, block1>>>(
            reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(K_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(V_cache.data_ptr<at::BFloat16>()),
            partial_O.data_ptr<float>(), partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
            seq_len, max_seq_len, d, H_q, H_kv, chunk_size
        );
        
        dim3 grid2(H_q, B);
        dim3 block2(64);
        splitk_decode_stage2_bf16<<<grid2, block2>>>(
            partial_O.data_ptr<float>(), partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(O.data_ptr<at::BFloat16>()),
            num_chunks, d, H_q
        );
    }
}
""")

with open('cuda_bf16_flash_attention/bf16_flash_attn_extension.cpp', 'w') as f:
    f.write("""
#include <torch/extension.h>

void run_forward_cuda_bf16(torch::Tensor Q, torch::Tensor K_cache, torch::Tensor V_cache, torch::Tensor O, int seq_len);

torch::Tensor forward(torch::Tensor Q, torch::Tensor K_cache, torch::Tensor V_cache, int seq_len) {
    auto O = torch::zeros_like(Q);
    run_forward_cuda_bf16(Q, K_cache, V_cache, O, seq_len);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Native BFloat16 Flash Attention Forward");
}
""")

print("[*] Compiling Native BFloat16 Flash Attention Extension...")
bf16_flash_ext = load(
    name='cuda_bf16_flash_attn_ext',
    sources=['cuda_bf16_flash_attention/bf16_flash_attn_extension.cpp', 'cuda_bf16_flash_attention/bf16_flash_attn_kernel.cu'],
    extra_cuda_cflags=['-O3'],
    verbose=False
)
print("[*] Compilation finished successfully!")

B, H_q, H_kv, d = 1, 32, 8, 64
PREFILL_TOKENS = 8192
DECODE_TOKENS = 32
MAX_SEQ_LEN = PREFILL_TOKENS + DECODE_TOKENS

print(f"[*] Benchmarking with NATIVE BFLOAT16: Prefill = {PREFILL_TOKENS} tokens, Decode = {DECODE_TOKENS} tokens")

def native_pytorch_attention_bf16(q, k_cache, v_cache, seq_len):
    q_len = q.shape[2]
    k_expanded = k_cache.repeat_interleave(H_q // H_kv, dim=1)
    v_expanded = v_cache.repeat_interleave(H_q // H_kv, dim=1)
    k = k_expanded[:, :, :seq_len, :]
    v = v_expanded[:, :, :seq_len, :]
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    if q_len > 1:
        mask = torch.triu(torch.ones(q_len, seq_len, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)

# Tensors natively in torch.bfloat16
Q_prefill = torch.randn(B, H_q, PREFILL_TOKENS, d, device='cuda', dtype=torch.bfloat16)
K_cache = torch.randn(B, H_kv, MAX_SEQ_LEN, d, device='cuda', dtype=torch.bfloat16)
V_cache = torch.randn(B, H_kv, MAX_SEQ_LEN, d, device='cuda', dtype=torch.bfloat16)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("\n--- PREFILL PHASE (8192 Tokens · NATIVE BFLOAT16) ---")
try:
    _ = native_pytorch_attention_bf16(Q_prefill, K_cache, V_cache, PREFILL_TOKENS)
    start.record()
    for _ in range(5):
        _ = native_pytorch_attention_bf16(Q_prefill, K_cache, V_cache, PREFILL_TOKENS)
    end.record()
    torch.cuda.synchronize()
    print(f"PyTorch BF16 Prefill     : {start.elapsed_time(end) / 5:.2f} ms")
except torch.cuda.OutOfMemoryError:
    print(f"PyTorch BF16 Prefill     : CRASHED! CUDA Out of Memory (OOM)!")
    torch.cuda.empty_cache()

# Flash Attention Native BF16
for _ in range(3):
    _ = bf16_flash_ext.forward(Q_prefill, K_cache, V_cache, PREFILL_TOKENS)
torch.cuda.synchronize()

start.record()
for _ in range(5):
    _ = bf16_flash_ext.forward(Q_prefill, K_cache, V_cache, PREFILL_TOKENS)
end.record()
torch.cuda.synchronize()
fa_bf16_prefill_time = start.elapsed_time(end) / 5
print(f"FlashAttn Native BF16    : {fa_bf16_prefill_time:.2f} ms (0.0 GB extra memory!)")

print("\n--- DECODE PHASE (32 Tokens · NATIVE BFLOAT16) ---")
pt_decode_total = 0
for step in range(DECODE_TOKENS):
    current_seq_len = PREFILL_TOKENS + step
    Q_decode = torch.randn(B, H_q, 1, d, device='cuda', dtype=torch.bfloat16)
    start.record()
    _ = native_pytorch_attention_bf16(Q_decode, K_cache, V_cache, current_seq_len)
    end.record()
    torch.cuda.synchronize()
    pt_decode_total += start.elapsed_time(end)
print(f"PyTorch BF16 Decode      : {pt_decode_total:.2f} ms")

fa_decode_total = 0
for step in range(DECODE_TOKENS):
    current_seq_len = PREFILL_TOKENS + step
    Q_decode = torch.randn(B, H_q, 1, d, device='cuda', dtype=torch.bfloat16)
    start.record()
    _ = bf16_flash_ext.forward(Q_decode, K_cache, V_cache, current_seq_len)
    end.record()
    torch.cuda.synchronize()
    fa_decode_total += start.elapsed_time(end)
print(f"Warp-Optimized BF16 Decode: {fa_decode_total:.2f} ms")
print(f"Decode Speedup vs PyTorch: {pt_decode_total / fa_decode_total:.2f}x")
