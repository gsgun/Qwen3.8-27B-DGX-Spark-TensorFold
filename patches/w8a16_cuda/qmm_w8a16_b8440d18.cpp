#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void w8a16_qmm_cuda(const at::Tensor& x, const at::Tensor& w8, const at::Tensor& scales,
                    const at::Tensor& biases, at::Tensor& out, bool f32);

// Host wrapper for dequant kernel (defined in qmm_kernel.cu)
void dequant_to_bf16_host(const uint8_t* w8, const void* scales, const void* biases,
                          void* out, int64_t N, int64_t K, int64_t KG);

// x: (M, K) bf16, w8: (N, K) uint8, scales: (N, K/64) bf16, biases: (N, K/64) bf16
// Returns: (M, N) bf16 or fp32
at::Tensor w8a16_cuda_forward(const at::Tensor& x, const at::Tensor& w8,
                              const at::Tensor& scales, const at::Tensor& biases,
                              bool f32) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2,
                "x must be CUDA bf16 2D");
    TORCH_CHECK(w8.is_cuda() && w8.scalar_type() == at::kByte && w8.dim() == 2,
                "w8 must be CUDA uint8 2D");
    const int64_t M = x.size(0), K = x.size(1), N = w8.size(0);
    TORCH_CHECK(K % 64 == 0, "K must be multiple of 64");
    const int64_t KG = K / 64;
    TORCH_CHECK(w8.size(1) == K, "w8 K mismatch");
    TORCH_CHECK(scales.size(0) == N && scales.size(1) == KG, "scales shape mismatch");
    TORCH_CHECK(biases.size(0) == N && biases.size(1) == KG, "biases shape mismatch");

    auto out = at::zeros({M, N}, x.options().dtype(f32 ? at::kFloat : at::kBFloat16));
    w8a16_qmm_cuda(x, w8, scales, biases, out, f32);
    return out;
}

// Dequantize w8 to bf16 using CUDA kernel
at::Tensor dequant_to_bf16(const at::Tensor& w8, const at::Tensor& scales,
                           const at::Tensor& biases) {
    TORCH_CHECK(w8.is_cuda() && w8.scalar_type() == at::kByte && w8.dim() == 2,
                "w8 must be CUDA uint8 2D");
    const int64_t N = w8.size(0), K = w8.size(1);
    TORCH_CHECK(K % 64 == 0, "K must be multiple of 64");
    const int64_t KG = K / 64;
    TORCH_CHECK(scales.size(0) == N && scales.size(1) == KG, "scales shape mismatch");
    TORCH_CHECK(biases.size(0) == N && biases.size(1) == KG, "biases shape mismatch");

    auto out = at::empty({N, K}, w8.options().dtype(at::kBFloat16));
    dequant_to_bf16_host(
        w8.data_ptr<uint8_t>(),
        scales.data_ptr(),
        biases.data_ptr(),
        out.data_ptr(),
        N, K, KG);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &w8a16_cuda_forward, "W8A16 QMM forward (x, w8, scales, biases, f32)");
    m.def("dequant_to_bf16", &dequant_to_bf16, "Dequantize w8 to bf16 (w8, scales, biases)");
}
