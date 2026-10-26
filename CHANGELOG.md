# Changelog

## 2026-10-25 — Native CUDA W8A16 Extension (c1 8-bit TP2)

### Added
- **`patches/w8a16_cuda/`**: native CUDA C++ PyTorch extension (`w8a16_cuda_qmm`)
  - `qmm_kernel.cu`: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` kernel,
    `sm_121a` / compute capability 12.1, shared-memory tiling (BM=32, BN=32),
    cooperative global-to-shared loading, vectorized dequantize-on-the-fly
    (int8 → bf16 per group of 64 along K), per-group `dot` accumulator
    pattern (`acc += dot * s + xs * b`) for multi-group correctness
  - `qmm_w8a16.cpp`: Pybind11 bindings via `torch::extension`,
    `w8a16_cuda_qmm.forward(x_bf16[M,K], w8_uint8[N,K], scales_bf16[N,K/64],
    biases_bf16[N,K/64], bf16_out_bool)`
  - `setup.py`: gencode `compute_121/sm_121`, `-O3 --use_fast_math -std=c++17
    -lineinfo --expt-relaxed-constexpr`
  - `test_parity.py`: parity test vs Triton `affine_kernels` reference
    (atol=1e-3, rtol=1e-2, max_diff ≤ 0.0312 across all shapes)
- **`patches/tp2w8a16.py`**: W8A16 dispatch module with native CUDA fast path
  (`bits==8 && gs==64` → `w8a16_cuda_qmm.forward`), Triton fallback intact
- **`patches/0006-fused-w8a16-affine.patch`**: W8A16 affine quantization path
- **`patches/0007-tp2-multistream-batch.patch`**: TP2 multi-stream batch verification
- **`patches/0008a-replicated-dflash2.patch`**: replicated DFlash2 drafter
- **`patches/tp2draftgraph.py`** (v10e, `3ccfeb2b`): CUDA graph decode path
- **`patches/tp2batchverify.py`**: batch verification for multi-stream decode
- **`patches/tp2draftrepl.py`**: draft replication for TP2
- **`patches/tp2slotcap.py`**: slot KV cap
- **`patches/tp2cudagraphs.py`**: CUDA graph management
- **`patches/tp2widthprobe.py`**: draft width probe
- **`patches/sitecustomize.py`**: NCCL timeout clamp (60 s)

### Performance (c1, 8-bit TP2, fixed-seed benchmark, thinking off, draft on)

| Leg | Native CUDA | v10e Triton | 4-bit MLX | Δ vs Triton |
| --- | ---: | ---: | ---: | ---: |
| c1d (code n=1) | **100.9** | 94.1 | 88.0 | +7.3% |
| c4d (code n=4) | **227.3** | 208.0 | 268.8 | +9.3% |
| p1d (prose n=1) | 45.5 | 45.4 | 55.4 | +0.3% |
| p4d (prose n=4) | 119.7 | 121.5 | 137.8 | -1.4% |
| m4d (mixed n=4) | 127.0 | 125.3 | — | +1.4% |
| Prefill 8K | 407 | 372 | 2049 | +9.5% |

- **n=1 code 100.9 tok/s beats 4-bit MLX (88.0) by +15%**
- 14/14 byte-exact parity (C=4, SHAs `d5589b366369` / `39191bf72982`)
- 0007 gate: target-colls/round = 128.0, draft-colls/round = 0.0
- Warm multi-turn agent TTFT: 102–340 ms (32 checkpoint slots)

### Architecture notes
- `--prefill-fp8` is gated to checkpoints with FP8 prompt kernels
  (`f8f8` matmul). The 8-bit MLX format does not include them; the flag is a
  no-op for the 8-bit checkpoint. Parked: W8-to-FP8 `torch._scaled_mm`
  prefill bridge for future cold >32K prefill workloads.
- `_symmetric_memory` is not used (requires P2P NVLink; c1 uses RoCEv2).
  Standard `dist.all_gather_into_tensor` is natively CUDA-graph-capturable
  on NCCL 2.30.7.
- `.so` deployed to both nodes' container site-packages and
  `/patches/w8a16_cuda/` (persistent mount). MD5 `3490ac1f6084c0a5a87d8ff461f02e28`.
- Kernel fixes (9 total): scale/bias lane pairing, per-group `dot` accumulator,
  A-frag manual load (no ldmatrix), B-frag single k_base, W8/X swizzle removal.

## 2026-10-14 — Live delta patch

- `patches/0099-live-delta.patch`: accumulated engine deltas

## 2026-10-02 — Initial release

- Patches `0001`–`0005`: image/video, YaRN, FP8 KV, memory reserve, KV pool
- `start.sh` / `stop.sh`: one-command setup
- 4-bit MLX checkpoint, single DGX Spark, 8 concurrent requests
