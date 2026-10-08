#include <torch/extension.h>

// Forward declaration of the CUDA launcher from flash_decode_kernel.cu
torch::Tensor flash_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
);

// C++ entry point with validation checks
torch::Tensor flash_decode_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    TORCH_CHECK(q.device().is_cuda(), "q must be on CUDA");
    TORCH_CHECK(k.device().is_cuda(), "k must be on CUDA");
    TORCH_CHECK(v.device().is_cuda(), "v must be on CUDA");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");

    // Expecting shapes:
    // q: (batch_size, n_heads, 1, head_dim) -> flattened to (B*n_heads, head_dim) inside launcher
    // k: (batch_size, n_heads, seq_len, head_dim)
    // v: (batch_size, n_heads, seq_len, head_dim)
    TORCH_CHECK(q.size(-1) == k.size(-1), "Head dimensions of Q and K must match");
    TORCH_CHECK(k.size(-1) == v.size(-1), "Head dimensions of K and V must match");
    TORCH_CHECK(k.size(-2) == v.size(-2), "Sequence length N of K and V must match");

    return flash_decode_cuda(q, k, v);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_decode_forward, "Flash Decode Forward (CUDA)");
}
