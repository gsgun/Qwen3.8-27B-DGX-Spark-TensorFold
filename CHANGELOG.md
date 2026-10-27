## 2026-10-27 — Repository realigned to the boot79 8-bit TP=2 hybrid

- **README rewritten**: the repo is now 100% dedicated to the dual-DGX-Spark 8-bit `TP=2`
  hybrid (`boot79`), with MiaAI-Lab-style benchmark tables (decode prose/code, prefill,
  quality/KL, 1×4-bit vs 2×4-bit vs 2×8-bit memory comparison, patch stack, settings).
  Every number is a measured scorecard value; the 1× Spark 4-bit recipe is credited, not
  carried.
- **Tree organized**: single-node 4-bit toolchain (`start.sh`, `stop.sh`, `scripts/`) and the
  MiaAI-Lab funding badge purged from the index; scorecards consolidated under
  `bench/scorecards/`; the production CUDA extension renamed to its importable name
  `patches/w8a16_cuda_qmm.cpython-312-aarch64-linux-gnu.so` (`b8440d18`) at the `patches/`
  root (the `/patches/`-root load rule is documented in the README); Gen-1 kernel, sources
  and `.so` moved to `patches/w8a16_cuda/archive/`; `qmm_kernel.cu`/`qmm.cpp` are now the
  `b8440d18` production sources.
- **`0008d2-checkpoint-slots32.patch` added**, generated from the live pre-patch backup
  (`tensorfold/__init__.py.bak-pre-0008d2-1024`): the `TF_SLOT_KV_CAP` slot-reservation-cap
  install block plus the B2 hook block.
- **`bench/scorecards/vdrain_matrix.txt` added**: the `b8440d18` determinism record
  (0/0/0/0 flips across 50 L2-flushed calls at M=1 bf16/f32, M=17, M=1024; compute-sanitizer
  0 errors; rel_err 0.189–0.270% vs FP64; RoCE flush matrix 0/0/0/0).
- **`.gitignore`**: `*.bak*`, `*.o`, `build/` added (local `.bak` files stay untracked on disk).
- Live `boot79` containers untouched: health `ok:true` and `.so`/wrapper md5s re-verified on
  both ranks before and after the cleanup.

## [boot80] - 2026-10-27
### Added
- Phase 1 (v11 target graph evaluation): Confirmed TP2_V11_GRAPH=0 winner (v10e 3ccfeb2b eager staging across S in 1..4, c4d 210.55 tok/s, 128.0 target collectives).
- Phase 2 (FP8 prefill AllGather): Patch 0009 rowwise float8_e4m3fn all_gather for M >= 2048 with per-row bfloat16 scaling. Cuts RoCEv2 prefill wire payload ~50% (cosine_sim > 0.9996). 8K-64K prefill accelerated to 1566/1580/1573/1522 tok/s (+7.2% to +19.7% speedup). Zero-restart runtime toggle via /tmp/tp2_fp8_gather_on.
- Phase 3 (Snapshots and NUMA): Mounted persistent snapshot spill storage (/var/tmp/tf-snapshots:/snapshots) with --snapshot-dir /snapshots alongside --checkpoint-slots 32 --prompt-cache-gib 16. NUMA node 0 memory and CPU binding (numactl --cpunodebind=0 --membind=0).
- Phase 4 (Verification battery): Verified 128.0 target collectives/round, 0.0 draft collectives, det500 byte-exact (550 tokens), and 8/8 cross-stream parity (prose 9ae72b23bafa x 4, code 39191bf72982 x 4).
