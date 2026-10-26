#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>

namespace {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp16(void* dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(read ? 16 : 0));
}

__device__ __forceinline__ void cp4(void* dst, const void* src, bool read) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(read ? 4 : 0));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint16_t u8_to_bf16_u16(uint8_t b) {
    __nv_bfloat16 h = __int2bfloat16_rn(static_cast<int>(b));
    return *reinterpret_cast<uint16_t*>(&h);
}

__device__ __forceinline__ uint32_t pack_bf16x2(uint16_t lo, uint16_t hi) {
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

template <int CHUNKS>
__device__ __forceinline__ int swz(int r, int c) { return c ^ (r % CHUNKS); }

template <int GS, int BM, int BN, int WM, int WN, int STAGES>
struct T8 {
    static constexpr int TH = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int XROW = GS * 2;
    static constexpr int WROW = GS;
    static constexpr int XCH = XROW / 16;
    static constexpr int WCH = WROW / 16;
    static constexpr int X = BM * XROW;
    static constexpr int W = BN * WROW;
    static constexpr int SB = BN * 2;
    static constexpr int XS = BM * 4;
    static constexpr int STAGE = X + W + 2 * SB + XS;
    static constexpr int PART = MT * NT * 4 * TH * 4;
    static constexpr int SMEM = STAGES * STAGE > PART ? STAGES * STAGE : PART;
};

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32>
__global__ void __launch_bounds__(WM * WN * 32) w8a16_qmm_kernel(
        const __nv_bfloat16* __restrict__ x,
        const uint8_t* __restrict__ w8,
        const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases,
        void* __restrict__ out,
        int M, int N, int K) {
    using T = T8<GS, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;

    const int cols_t = (N + BN - 1) / BN;
    const int m0 = (blockIdx.x / cols_t) * BM;
    const int n0 = (blockIdx.x % cols_t) * BN;

    auto stage = [&](int s) { return buf + s * T::STAGE; };

    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::XCH; c += T::TH) {
            const int r = c / T::XCH, ch = c % T::XCH;
            const bool ok = m0 + r < M;
            cp16(p + r * T::XROW + ch * 16,
                 x + static_cast<size_t>(min(m0 + r, M - 1)) * K + g * GS + ch * 8, ok);
        }
        unsigned char* pw = p + T::X;
        for (int c = tid; c < BN * T::WCH; c += T::TH) {
            const int r = c / T::WCH, ch = c % T::WCH;
            const bool ok = n0 + r < N;
            cp16(pw + r * T::WROW + ch * 16,
                 w8 + static_cast<size_t>(min(n0 + r, N - 1)) * K + g * GS + ch * 16, ok);
        }
        __nv_bfloat16* ps = reinterpret_cast<__nv_bfloat16*>(pw + T::W);
        for (int c = tid; c < BN; c += T::TH) {
            const int col = n0 + c;
            ps[c] = col < N ? scales[static_cast<size_t>(col) * KG + g] : __float2bfloat16_rn(0.0f);
        }
        __nv_bfloat16* pb = ps + BN;
        for (int c = tid; c < BN; c += T::TH) {
            const int col = n0 + c;
            pb[c] = col < N ? biases[static_cast<size_t>(col) * KG + g] : __float2bfloat16_rn(0.0f);
        }
        cp_commit();
    };

    for (int s = 0; s < STAGES - 1; ++s) load(s, s);

    float acc[T::MT][T::NT][4];
    #pragma unroll
    for (int i = 0; i < T::MT; ++i)
        #pragma unroll
        for (int j = 0; j < T::NT; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    for (int g = 0; g < KG; ++g) {
        const int cur = g % STAGES;
        const int nxt = (g + STAGES - 1) % STAGES;
        cp_wait<STAGES - 2>();
        __syncthreads();

        if (g + STAGES - 1 < KG) load(nxt, g + STAGES - 1);

        // Compute row sums
        unsigned char* psum = stage(cur);
        __nv_bfloat16* sxsum = reinterpret_cast<__nv_bfloat16*>(psum);
        float* pxs = reinterpret_cast<float*>(psum + T::X + T::W + 2 * T::SB);
        for (int r = tid; r < BM; r += T::TH) {
            float s = 0.0f;
            for (int k = 0; k < GS; ++k) s += __bfloat162float(sxsum[r * GS + k]);
            pxs[r] = s;
        }
        __syncthreads();

        unsigned char* p = stage(cur);
        __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(p);
        uint8_t* pw = reinterpret_cast<uint8_t*>(p + T::X);
        __nv_bfloat16* ps = reinterpret_cast<__nv_bfloat16*>(p + T::X + T::W);
        __nv_bfloat16* pb = ps + BN;
        pxs = reinterpret_cast<float*>(p + T::X + T::W + 2 * T::SB);

        #pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[T::MT][4];
            #pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix_x4(a[i], &sx[r * GS + ch * 8]);
            }

            #pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int ncol = wn * (BN / WN) + j * 8;
                const int n_idx = ncol + (lane >> 2);
                const int k_base = kt * 16;
                const int k0 = k_base + 2 * (lane & 3);

                const uint8_t* wrow = &pw[n_idx * GS + k_base];
                const uint16_t b0_lo = u8_to_bf16_u16(wrow[k0]);
                const uint16_t b0_hi = u8_to_bf16_u16(wrow[k0 + 1]);
                const uint16_t b1_lo = u8_to_bf16_u16(wrow[k0 + 8]);
                const uint16_t b1_hi = u8_to_bf16_u16(wrow[k0 + 9]);
                const uint32_t b0 = pack_bf16x2(b0_lo, b0_hi);
                const uint32_t b1 = pack_bf16x2(b1_lo, b1_hi);

                #pragma unroll
                for (int i = 0; i < T::MT; ++i)
                    mma_bf16(acc[i][j], a[i], b0, b1);
            }
        }

        #pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            const float s0 = __bfloat162float(ps[col]);
            const float b0v = __bfloat162float(pb[col]);
            const float s1 = __bfloat162float(ps[col + 1]);
            const float b1v = __bfloat162float(pb[col + 1]);
            #pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                const float xs0 = pxs[row];
                const float xs1 = pxs[row + 8];
                acc[i][j][0] = acc[i][j][0] * s0 + xs0 * b0v;
                acc[i][j][1] = acc[i][j][1] * s0 + xs1 * b0v;
                acc[i][j][2] = acc[i][j][2] * s1 + xs0 * b1v;
                acc[i][j][3] = acc[i][j][3] * s1 + xs1 * b1v;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < T::MT; ++i) {
        #pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int r0 = m0 + wm * (BM / WM) + i * 16 + (lane >> 2);
            const int c0 = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            if (F32) {
                float* o = reinterpret_cast<float*>(out);
                if (r0 < M && c0 < N)
                    *reinterpret_cast<uint2*>(o + static_cast<size_t>(r0) * N + c0) =
                        make_uint2(__float_as_uint(acc[i][j][0]), __float_as_uint(acc[i][j][1]));
                if (r0 + 8 < M && c0 < N)
                    *reinterpret_cast<uint2*>(o + static_cast<size_t>(r0 + 8) * N + c0) =
                        make_uint2(__float_as_uint(acc[i][j][2]), __float_as_uint(acc[i][j][3]));
            } else {
                __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(out);
                if (r0 < M && c0 < N)
                    *reinterpret_cast<uint32_t*>(o + static_cast<size_t>(r0) * N + c0) =
                        pack_bf16x2(__bfloat16_as_ushort(__float2bfloat16_rn(acc[i][j][0])),
                                    __bfloat16_as_ushort(__float2bfloat16_rn(acc[i][j][1])));
                if (r0 + 8 < M && c0 < N)
                    *reinterpret_cast<uint32_t*>(o + static_cast<size_t>(r0 + 8) * N + c0) =
                        pack_bf16x2(__bfloat16_as_ushort(__float2bfloat16_rn(acc[i][j][2])),
                                    __bfloat16_as_ushort(__float2bfloat16_rn(acc[i][j][3])));
            }
        }
    }
}

template <int GS, bool F32>
void launch(const at::Tensor& x, const at::Tensor& w8, const at::Tensor& scales,
            const at::Tensor& biases, at::Tensor& out, int N, int K, int M) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* wp = reinterpret_cast<const uint8_t*>(w8.data_ptr());
    const auto* sp = reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr());
    const auto* bp = reinterpret_cast<const __nv_bfloat16*>(biases.data_ptr());
    void* op = out.data_ptr();

    if (M <= 16) {
        using T = T8<GS, 16, 64, 1, 4, 4>;
        dim3 grid(((M + 15) / 16) * ((N + 63) / 64));
        auto* k = w8a16_qmm_kernel<GS, 16, 64, 1, 4, 4, F32>;
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        k<<<grid, T::TH, T::SMEM, stream>>>(xp, wp, sp, bp, op, M, N, K);
    } else if (M <= 32) {
        using T = T8<GS, 32, 64, 1, 4, 4>;
        dim3 grid(((M + 31) / 32) * ((N + 63) / 64));
        auto* k = w8a16_qmm_kernel<GS, 32, 64, 1, 4, 4, F32>;
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        k<<<grid, T::TH, T::SMEM, stream>>>(xp, wp, sp, bp, op, M, N, K);
    } else {
        using T = T8<GS, 64, 64, 1, 4, 3>;
        dim3 grid(((M + 63) / 64) * ((N + 63) / 64));
        auto* k = w8a16_qmm_kernel<GS, 64, 64, 1, 4, 3, F32>;
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        k<<<grid, T::TH, T::SMEM, stream>>>(xp, wp, sp, bp, op, M, N, K);
    }
}

} // anonymous namespace

void w8a16_qmm_cuda(const at::Tensor& x, const at::Tensor& w8, const at::Tensor& scales,
                    const at::Tensor& biases, at::Tensor& out, bool f32) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
                "x: (M, K) bf16 contiguous");
    const int M = x.size(0), K = x.size(1);
    TORCH_CHECK(K % 64 == 0, "K must be a multiple of 64");
    const int KG = K / 64;
    TORCH_CHECK(w8.is_cuda() && w8.scalar_type() == at::kByte && w8.dim() == 2 && w8.is_contiguous(),
                "w8: (N, K) uint8 contiguous");
    const int N = w8.size(0);
    TORCH_CHECK(w8.size(1) == K, "w8 K mismatch");
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == at::kBFloat16 && scales.dim() == 2,
                "scales: (N, KG) bf16");
    TORCH_CHECK(scales.size(0) == N && scales.size(1) == KG, "scales shape mismatch");
    TORCH_CHECK(biases.is_cuda() && biases.scalar_type() == at::kBFloat16 && biases.dim() == 2,
                "biases: (N, KG) bf16");
    TORCH_CHECK(biases.size(0) == N && biases.size(1) == KG, "biases shape mismatch");
    TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.size(0) == M && out.size(1) == N,
                "out: (M, N)");

    c10::cuda::CUDAGuard guard(x.device());
    if (f32) launch<64, true>(x, w8, scales, biases, out, N, K, M);
    else launch<64, false>(x, w8, scales, biases, out, N, K, M);
}
