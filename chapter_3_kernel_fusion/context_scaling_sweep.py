import torch
from torch.utils.cpp_extension import load
import math
import os

print("[*] Loading compiled BF16 FlashAttention extension...")
bf16_flash_ext = load(
    name='cuda_bf16_flash_attn_ext',
    sources=['chapter_3_kernel_fusion/cuda_bf16_flash_attention/bf16_flash_attn_extension.cpp', 
             'chapter_3_kernel_fusion/cuda_bf16_flash_attention/bf16_flash_attn_kernel.cu'],
    extra_cuda_cflags=['-O3'],
    verbose=False
)

B, H_q, H_kv, d = 1, 32, 8, 64
context_lengths = [2048, 4096, 8192, 12288, 16384]

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

results = []

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print(f"\n{'='*75}")
print(f"{'Context':<10} | {'PyTorch Latency':<18} | {'FlashAttn Latency':<20} | {'PyTorch Intermediate VRAM':<22}")
print(f"{'='*75}")

for seq_len in context_lengths:
    Q = torch.randn(B, H_q, seq_len, d, device='cuda', dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, seq_len, d, device='cuda', dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, seq_len, d, device='cuda', dtype=torch.bfloat16)
    
    # Theoretical intermediate VRAM for PyTorch: scores (N*N*2) + probs (N*N*2) across H_q heads
    vram_gb = (B * H_q * seq_len * seq_len * 2 * 2) / (1024**3)
    
    # 1. Test PyTorch
    torch.cuda.empty_cache()
    pt_time_str = "CRASHED (OOM)"
    try:
        # Warmup
        _ = native_pytorch_attention_bf16(Q, K, V, seq_len)
        torch.cuda.synchronize()
        start.record()
        for _ in range(3):
            _ = native_pytorch_attention_bf16(Q, K, V, seq_len)
        end.record()
        torch.cuda.synchronize()
        pt_time = start.elapsed_time(end) / 3
        pt_time_str = f"{pt_time:.2f} ms"
    except torch.cuda.OutOfMemoryError:
        pt_time_str = "CRASHED (OOM)"
        torch.cuda.empty_cache()
        
    # 2. Test FlashAttention Native BF16
    torch.cuda.empty_cache()
    # Warmup
    _ = bf16_flash_ext.forward(Q, K, V, seq_len)
    torch.cuda.synchronize()
    start.record()
    for _ in range(3):
        _ = bf16_flash_ext.forward(Q, K, V, seq_len)
    end.record()
    torch.cuda.synchronize()
    fa_time = start.elapsed_time(end) / 3
    fa_time_str = f"{fa_time:.2f} ms"
    
    print(f"{seq_len:<10} | {pt_time_str:<18} | {fa_time_str:<20} | {vram_gb:.2f} GB")
    results.append({
        "seq_len": seq_len,
        "pytorch": pt_time_str,
        "flash_attn": fa_time_str,
        "intermediate_vram_gb": round(vram_gb, 2)
    })

print(f"{'='*75}\n")
