
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
