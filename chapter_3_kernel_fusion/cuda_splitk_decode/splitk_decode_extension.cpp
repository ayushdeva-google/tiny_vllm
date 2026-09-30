#include <torch/extension.h>

torch::Tensor splitk_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int N_chunk
);

torch::Tensor splitk_decode_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int N_chunk
) {
    TORCH_CHECK(q.device().is_cuda(), "q must be on CUDA");
    TORCH_CHECK(k.device().is_cuda(), "k must be on CUDA");
    TORCH_CHECK(v.device().is_cuda(), "v must be on CUDA");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");

    return splitk_decode_cuda(q, k, v, N_chunk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &splitk_decode_forward, "Split-K Flash Decode Forward (CUDA)");
}
