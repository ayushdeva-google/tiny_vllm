import os
import torch
from torch.utils.cpp_extension import load
import time
import math

os.makedirs('cuda_flash_attention', exist_ok=True)

with open('cuda_flash_attention/flash_attn_kernel.cu', 'w') as f:
    f.write("""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void prefill_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ O,
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
        for(int k=0; k<d; ++k) {
            q_reg[k] = Q[q_offset + k];
        }
    }
    
    // Use ONLY 32KB shared memory, reusing it for K and then V!
    __shared__ float smem[128][64];
    
    float scale = 1.0f / sqrtf((float)d);
    
    for (int k_chunk_idx = 0; k_chunk_idx <= q_chunk_idx; ++k_chunk_idx) {
        int k_global_start = k_chunk_idx * 128;
        int k_global_idx = k_global_start + tx;
        int kv_offset = batch_idx * (num_heads_kv * max_seq_len * d) + head_idx_kv * (max_seq_len * d) + k_global_idx * d;
        
        // --- 1. Load K into shared memory ---
        if (k_global_idx < q_len) {
            for(int k=0; k<d; ++k) {
                smem[tx][k] = K_cache[kv_offset + k];
            }
        }
        __syncthreads();
        
        // --- 2. Compute Attention Scores ---
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
        __syncthreads(); // Done using smem for K
        
        // --- 3. Load V into the EXACT SAME shared memory buffer ---
        if (k_global_idx < q_len) {
            for(int k=0; k<d; ++k) {
                smem[tx][k] = V_cache[kv_offset + k];
            }
        }
        __syncthreads();
        
        // --- 4. Online Softmax and Accumulate ---
        if (q_global_idx < q_len) {
            float exp_diff = expf(m_prev - m_curr);
            float l_curr = l_prev * exp_diff;
            
            for (int c = 0; c < 128; ++c) {
                int key_global_idx = k_global_start + c;
                if (key_global_idx <= q_global_idx && key_global_idx < q_len) {
                    float p = expf(scores[c] - m_curr);
                    l_curr += p;
                    scores[c] = p; // reuse scores array
                }
            }
            
            for (int k = 0; k < d; ++k) {
                o_reg[k] *= exp_diff;
                for (int c = 0; c < 128; ++c) {
                    int key_global_idx = k_global_start + c;
                    if (key_global_idx <= q_global_idx && key_global_idx < q_len) {
                        o_reg[k] += scores[c] * smem[c][k]; // smem is now V
                    }
                }
            }
            m_prev = m_curr;
            l_prev = l_curr;
        }
        __syncthreads(); // Done using smem for V
    }
    
    // Write out
    if (q_global_idx < q_len) {
        int o_offset = batch_idx * (num_heads_q * q_len * d) + head_idx_q * (q_len * d) + q_global_idx * d;
        for(int k=0; k<d; ++k) {
            O[o_offset + k] = o_reg[k] / l_prev;
        }
    }
}

__global__ void splitk_decode_stage1(
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
    int tx = threadIdx.x;
    
    float q_reg[64];
    int q_offset = batch_idx * (num_heads_q * 1 * d) + head_idx_q * d;
    for(int k = 0; k < d; ++k) {
        q_reg[k] = Q[q_offset + k];
    }
    
    // Reuse shared memory trick here too
    __shared__ float smem[128][64];
    
    int start_token = chunk_idx * chunk_size;
    int end_token = min(start_token + chunk_size, seq_len);
    
    float m_i = -1e20f;
    float l_i = 0.0f;
    float o_i[64] = {0.0f};
    float scale = 1.0f / sqrtf((float)d);
    
    int kv_offset_base = batch_idx * (num_heads_kv * max_seq_len * d) + head_idx_kv * (max_seq_len * d);
    
    // 1. Load K
    if (start_token + tx < end_token) {
        for(int k = 0; k < d; ++k) smem[tx][k] = K_cache[kv_offset_base + (start_token + tx) * d + k];
    }
    __syncthreads();
    
    float scores[128];
    if (tx == 0) {
        for (int c = 0; c < (end_token - start_token); ++c) {
            float dot = 0.0f;
            for (int k = 0; k < d; ++k) dot += q_reg[k] * smem[c][k];
            scores[c] = dot * scale;
            if (scores[c] > m_i) m_i = scores[c];
        }
    }
    __syncthreads();
    
    // 2. Load V
    if (start_token + tx < end_token) {
        for(int k = 0; k < d; ++k) smem[tx][k] = V_cache[kv_offset_base + (start_token + tx) * d + k];
    }
    __syncthreads();
    
    if (tx == 0) {
        for (int c = 0; c < (end_token - start_token); ++c) {
            float p = expf(scores[c] - m_i);
            l_i += p;
            for (int k = 0; k < d; ++k) o_i[k] += p * smem[c][k];
        }
        
        int out_idx = batch_idx * (num_heads_q * gridDim.x) + head_idx_q * gridDim.x + chunk_idx;
        partial_m[out_idx] = m_i;
        partial_l[out_idx] = l_i;
        for (int k = 0; k < d; ++k) partial_O[out_idx * d + k] = o_i[k];
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

// C++ Launcher wrapper inside the .cu file
void run_forward_cuda(
    torch::Tensor Q, torch::Tensor K_cache, torch::Tensor V_cache, torch::Tensor O,
    int seq_len
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
        prefill_kernel<<<grid, block>>>(
            Q.data_ptr<float>(), K_cache.data_ptr<float>(), V_cache.data_ptr<float>(), O.data_ptr<float>(),
            q_len, max_seq_len, d, H_q, H_kv
        );
    } else { // DECODE
        int chunk_size = 128;
        int num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        
        auto partial_O = torch::zeros({B, H_q, num_chunks, d}, Q.options());
        auto partial_m = torch::empty({B, H_q, num_chunks}, Q.options());
        auto partial_l = torch::empty({B, H_q, num_chunks}, Q.options());
        
        dim3 grid1(num_chunks, H_q, B);
        dim3 block1(128);
        splitk_decode_stage1<<<grid1, block1>>>(
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
}
""")

with open('cuda_flash_attention/flash_attn_extension.cpp', 'w') as f:
    f.write("""
#include <torch/extension.h>

void run_forward_cuda(torch::Tensor Q, torch::Tensor K_cache, torch::Tensor V_cache, torch::Tensor O, int seq_len);

torch::Tensor forward(torch::Tensor Q, torch::Tensor K_cache, torch::Tensor V_cache, int seq_len) {
    auto O = torch::zeros_like(Q);
    run_forward_cuda(Q, K_cache, V_cache, O, seq_len);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention Forward");
}
""")

print("[*] Compiling FlashAttention Extension (this takes ~10 seconds)...")
flash_ext = load(
    name='cuda_flash_attention_ext',
    sources=['cuda_flash_attention/flash_attn_extension.cpp', 'cuda_flash_attention/flash_attn_kernel.cu'],
    verbose=False
)
print("[*] Compilation finished.")

B, H_q, H_kv, d = 1, 32, 8, 64
PREFILL_TOKENS = 8192
DECODE_TOKENS = 32
MAX_SEQ_LEN = PREFILL_TOKENS + DECODE_TOKENS

print(f"[*] Benchmarking with Prefill = {PREFILL_TOKENS} tokens, Decode = {DECODE_TOKENS} tokens")

def native_pytorch_attention(q, k_cache, v_cache, seq_len):
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

Q_prefill = torch.randn(B, H_q, PREFILL_TOKENS, d, device='cuda', dtype=torch.float32)
K_cache = torch.randn(B, H_kv, MAX_SEQ_LEN, d, device='cuda', dtype=torch.float32)
V_cache = torch.randn(B, H_kv, MAX_SEQ_LEN, d, device='cuda', dtype=torch.float32)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("\n--- PREFILL PHASE (8192 Tokens) ---")
# Warmup
for _ in range(3):
    _ = native_pytorch_attention(Q_prefill, K_cache, V_cache, PREFILL_TOKENS)
    _ = flash_ext.forward(Q_prefill, K_cache, V_cache, PREFILL_TOKENS)
torch.cuda.synchronize()

start.record()
for _ in range(5):
    _ = native_pytorch_attention(Q_prefill, K_cache, V_cache, PREFILL_TOKENS)
end.record()
torch.cuda.synchronize()
pt_prefill_time = start.elapsed_time(end) / 5
print(f"PyTorch Prefill   : {pt_prefill_time:.2f} ms")

start.record()
for _ in range(5):
    _ = flash_ext.forward(Q_prefill, K_cache, V_cache, PREFILL_TOKENS)
end.record()
torch.cuda.synchronize()
fa_prefill_time = start.elapsed_time(end) / 5
print(f"FlashAttn Prefill : {fa_prefill_time:.2f} ms")
print(f"Prefill Speedup   : {pt_prefill_time / fa_prefill_time:.2f}x")

print("\n--- DECODE PHASE (32 Tokens) ---")
pt_decode_total = 0
for step in range(DECODE_TOKENS):
    current_seq_len = PREFILL_TOKENS + step
    Q_decode = torch.randn(B, H_q, 1, d, device='cuda', dtype=torch.float32)
    start.record()
    _ = native_pytorch_attention(Q_decode, K_cache, V_cache, current_seq_len)
    end.record()
    torch.cuda.synchronize()
    pt_decode_total += start.elapsed_time(end)
print(f"PyTorch Decode    : {pt_decode_total:.2f} ms")

fa_decode_total = 0
for step in range(DECODE_TOKENS):
    current_seq_len = PREFILL_TOKENS + step
    Q_decode = torch.randn(B, H_q, 1, d, device='cuda', dtype=torch.float32)
    start.record()
    _ = flash_ext.forward(Q_decode, K_cache, V_cache, current_seq_len)
    end.record()
    torch.cuda.synchronize()
    fa_decode_total += start.elapsed_time(end)
print(f"FlashAttn Decode  : {fa_decode_total:.2f} ms")
print(f"Decode Speedup    : {pt_decode_total / fa_decode_total:.2f}x")
