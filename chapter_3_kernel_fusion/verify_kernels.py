import torch
import math
import os
import time
from torch.utils.cpp_extension import load

print("[*] JIT Compiling Kernels for Verification...")
curr_dir = os.path.dirname(os.path.abspath(__file__))
flash_ext = load(
    name='cuda_flash_decode_ext',
    sources=[
        os.path.join(curr_dir, 'cuda_flash_decode', 'flash_decode_extension.cpp'),
        os.path.join(curr_dir, 'cuda_flash_decode', 'flash_decode_kernel.cu')
    ], verbose=False
)
splitk_ext = load(
    name='cuda_splitk_decode_ext',
    sources=[
        os.path.join(curr_dir, 'cuda_splitk_decode', 'splitk_decode_extension.cpp'),
        os.path.join(curr_dir, 'cuda_splitk_decode', 'splitk_decode_kernel.cu')
    ], verbose=False
)

B, H_q, H_kv, N, d = 1, 32, 8, 2048, 64
scale = 1.0 / math.sqrt(d)
iters = 200

print(f"[*] Benchmarking at N={N} tokens...")
q = torch.randn(B, H_q, 1, d, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, H_q, N, d, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, H_q, N, d, device='cuda', dtype=torch.bfloat16)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# 1. Flash Decode Original
for _ in range(20): _ = flash_ext.forward(q, k, v)
torch.cuda.synchronize()
start.record()
for _ in range(iters): _ = flash_ext.forward(q, k, v)
end.record()
torch.cuda.synchronize()
print(f"Raw C++ FlashDecode (Original) : {start.elapsed_time(end) / iters:.4f} ms per layer")

# 2. Flash Decode Split-K
for _ in range(20): _ = splitk_ext.forward(q, k, v, 128)
torch.cuda.synchronize()
start.record()
for _ in range(iters): _ = splitk_ext.forward(q, k, v, 128)
end.record()
torch.cuda.synchronize()
print(f"Raw C++ Split-K FlashDecode  : {start.elapsed_time(end) / iters:.4f} ms per layer")

# 3. Native PyTorch cuBLAS
for _ in range(20):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
torch.cuda.synchronize()
start.record()
for _ in range(iters):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
end.record()
torch.cuda.synchronize()
print(f"Native PyTorch cuBLAS        : {start.elapsed_time(end) / iters:.4f} ms per layer")

# 4. Hidden BFloat16 Cast Overhead (The Bug)
k_bf16 = torch.randn(1, 32, 2048, 64, device='cuda', dtype=torch.bfloat16)
v_bf16 = torch.randn(1, 32, 2048, 64, device='cuda', dtype=torch.bfloat16)
for _ in range(10): 
    _ = k_bf16.to(torch.float32).contiguous()
torch.cuda.synchronize()
start.record()
for _ in range(16): # 16 layers
    _k = k_bf16.to(torch.float32).contiguous()
    _v = v_bf16.to(torch.float32).contiguous()
end.record()
torch.cuda.synchronize()
print(f"PyTorch BFloat16 -> Float32 copy overhead (across 16 layers): {start.elapsed_time(end):.4f} ms")
