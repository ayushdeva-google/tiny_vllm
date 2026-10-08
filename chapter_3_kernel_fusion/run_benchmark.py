import torch
from torch.utils.cpp_extension import load
import time
import math

flash_ext = load(name='cuda_flash_attention_ext', sources=['cuda_flash_attention/flash_attn_extension.cpp', 'cuda_flash_attention/flash_attn_kernel.cu'], verbose=False)

B, H_q, H_kv, d = 1, 32, 8, 64
PREFILL_TOKENS = 8192
DECODE_TOKENS = 32
MAX_SEQ_LEN = PREFILL_TOKENS + DECODE_TOKENS

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

# Try Naive PyTorch
try:
    _ = native_pytorch_attention(Q_prefill, K_cache, V_cache, PREFILL_TOKENS)
    start.record()
    for _ in range(5):
        _ = native_pytorch_attention(Q_prefill, K_cache, V_cache, PREFILL_TOKENS)
    end.record()
    torch.cuda.synchronize()
    print(f"PyTorch Prefill   : {start.elapsed_time(end) / 5:.2f} ms")
except torch.cuda.OutOfMemoryError as e:
    print(f"PyTorch Prefill   : CRASHED! CUDA Out of Memory (OOM) trying to allocate 8.5 GB for the NxN matrix!")
    torch.cuda.empty_cache() # Clear it so flash attention can run

# Flash Attention
for _ in range(3):
    _ = flash_ext.forward(Q_prefill, K_cache, V_cache, PREFILL_TOKENS)
torch.cuda.synchronize()
start.record()
for _ in range(5):
    _ = flash_ext.forward(Q_prefill, K_cache, V_cache, PREFILL_TOKENS)
end.record()
torch.cuda.synchronize()
print(f"FlashAttn Prefill : {start.elapsed_time(end) / 5:.2f} ms (0 GB extra memory!)")

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
