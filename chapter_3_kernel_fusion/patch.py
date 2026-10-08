import re

with open("cuda_splitk_decode/splitk_decode_kernel.cu", "r") as f:
    content = f.read()

replacement = """torch::Tensor splitk_decode_cuda(
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
}"""

content = re.sub(r'torch::Tensor splitk_decode_cuda_inner\(.*$', replacement, content, flags=re.DOTALL)
with open("cuda_splitk_decode/splitk_decode_kernel.cu", "w") as f:
    f.write(content)
