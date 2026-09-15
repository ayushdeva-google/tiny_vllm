#include <torch/extension.h>

// Declaration of the CUDA forward function from rmsnorm_kernel.cu
torch::Tensor rmsnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    float eps
);

// C++ frontend to Python
torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, float eps) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.size(-1) == weight.size(0), "Hidden dimension must match weight size");

    return rmsnorm_forward_cuda(input, weight, eps);
}

// Bind the C++ function to Python via pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
}

