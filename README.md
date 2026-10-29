# Qwen3.8-27B on 2× DGX Spark — 8-Bit TP=2 Hybrid (`boot91`)

A 27B-parameter LLM serving on **two** NVIDIA DGX Sparks over a direct 200Gbps RoCEv2 link,
running **8-bit affine weights** (`group_size=64`, `8.627 bpw`) with a hand-built CUDA W8A16
kernel, `DFlash2` speculative decoding, and a 1-million-token pooled FP8 KV cache — with
multi-turn prefix caching that answers a warmed agent request in ~100 ms.

**Current production build: `boot91`** = `boot80` (`8b57000`) + the boot90/boot91 line: **PAR=6**
(6 × 262,144 = **1,572,864 pooled tokens**, 24 GiB fp8 KV/rank), the **v11 draft suite** (`S≤8`,
`tp2draftgraph.py` `44e592fd`), RoCE MTU-9000 jumbo self-heal, `RESERVE_GIB=16`, `--prompt-cache-gib 32`,
and `TP2_PREFILL_MODE` passthrough. `boot79`/`boot80` remain the sealed 8-bit hybrid baselines this
file's main tables document; `boot91` is the deployed resident. Every number in this file was measured
on this cluster, not promised.

> **⚠ Operational status (2026-10-28 ~21:xx PDT, live probe):** c1 is **boot91**, healthy —
> `qwen38-tf-rank0`+`rank1` `RestartCount=0`, `.so` md5 `b8440d18…`, draft module `44e592fd…` (v11),
> FP8 gather `ON`, `max streams=6`, `ctx=262,144`/slot (6 slots = 1,572,864 pooled). **Open:** a **~2×
> decode regression vs boot90** (c1d **45.6** vs **85–89** tok/s) attributed to the v6-stride→v11 draft
> swap (pending clean A/B); the RoCE link is at **109.16 Gb/s of 200G** with lane-0 PHY errors; and the
> engine image is `v0.6.0` vs upstream `v1.1.8` (canary gated). boot91 is **not yet committed** to any
> repo. See **Known issues** and **To do** below.

![Qwen3.8-27B on the DGX Spark](.github/image.png)

---

## Summary — measured, not promised

| | |
| --- | --- |
| **Architecture** | `2× NVIDIA DGX Spark (GB10, 200GbE ConnectX-7 RoCEv2)` · `TensorFold v0.6.0 + 8-Bit TP=2 Hybrid Patch Suite` |
| **Weights & drafter** | `TensorFold/Qwen3.8-27B-MLX-8bit` (`group_size=64`, `8.627 bpw`) + `z-lab/Qwen3.8-27B-DFlash2` (rank-replicated, `0` drafter network collectives) |
| **Context & cache** | `4 × 262,144` native windows (`1,048,576` pooled FP8 KV tokens) · `16 GiB` pinned prompt cache (`32` Gated DeltaNet + KV prefix slots) · `PREFILL_FP8=0` (bf16 prompt activations) |
| **Top line** | Code 1-req **94.0 tok/s** (Gen-1 CUDA: 100.9) · Code 4-req **196.1–208.0 tok/s** combined · Prose 1-req **45.4 tok/s** (102 ms warm TTFT) · Prose 4-req **111.4–121.5 tok/s** · Prefill 8K/16K/32K/64K **1,447 / 1,541 / 1,504 / 1,406 tok/s** |
| **Headroom** | `88 GiB` free on `spark1` (32–33 GiB used on spark1, 42 GiB used on spark2) |

This repository is the complete, self-contained recipe for that build: the dual-node launcher,
the nine-engine patch set, the fused W8A16 CUDA extension (production `.so` + sources), the
benchmark battery, and the scorecards behind every number above.

---

## Decode — prose (Table A)

`DFlash2` speculative decoding, `PREFILL_FP8=0` bf16 prompt activations. `boot79` = the
hybrid resident (Mode A CUDA prefill `M≥512` + Triton decode `M<512`).

| Requests | Aggregate (`boot79` / `v10e`) | Per request | Warm prefix TTFT (`32` slots) | Pure CUDA (`w78` `b8440d18`) | Unpatched 8-bit TP2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1 (`p1d`)** | **45.4 tok/s** | **45.4 tok/s** | `102 ms` | `41.6 tok/s` | `10.5 tok/s` |
| **2 (`p2d`)** | **~82.0 tok/s** | **~41.0 tok/s** | `~180 ms` | — | — |
| **4 (`p4d`)** | **111.4 – 121.5 tok/s** | **27.9 – 30.4 tok/s** | `~340 ms` | `96.1 tok/s` | `14.4 tok/s` |

## Decode — code (Table B)

Same conditions, code-generation prompt.

| Requests | Aggregate (`boot79` / `v10e`) | Per request | Pure CUDA (`w78` `b8440d18`) | Gen-1 CUDA (`3490ac1f`) | Serial reference (`c4off`, `draft=false`) | Unpatched 8-bit TP2 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 (`c1d`)** | **94.0 tok/s** | **94.0 tok/s** | `87.2 tok/s` | `100.9 tok/s` | `~21.0 tok/s` | `21.1 tok/s` |
| **4 (`c4d`)** | **196.1 – 208.0 tok/s** | **49.0 – 52.0 tok/s** | `186.0 tok/s` | `227.3 tok/s` | `37.5 – 38.6 tok/s` | `21.5 tok/s` |
| **4 mixed (`m4d`)** | **111.5 – 125.3 tok/s** | **27.9 – 31.3 tok/s** | `104.0 tok/s` | `96.8 tok/s` | — | — |

Notes on the columns:

- **`boot79`** is the resident hybrid: native CUDA for prefill, v10e Triton for decode. It is
  what the cluster serves.
- **`v10e`** (`3ccfeb2b`) is the all-Triton draft-graph variant — its `c4d 208.0` is the
  ceiling for the Triton path on the same image; it is the fallback and the A/B reference.
- **`w78`** is the pure-CUDA `b8440d18` decode (rollback path, `TP2_W8A16_DECODE=1`).
- **Gen-1 `3490ac1f`** is the first native-CUDA kernel generation — binary lost (md5-only),
  source recovered, kept in `patches/w8a16_cuda/archive/` for the record.
- **Unpatched 8-bit TP2** = stock TensorFold dequant-GEMV decode: this is the 29 GB/s wall
  that the fused kernel exists to remove (~4–9× on decode).

## Prefill (Table C)

How fast a prompt is read before the reply starts — `PREFILL_FP8=0` bf16, Mode A CUDA
`dequant_to_bf16` + `cuBLASLt` BF16 tensor cores.

| Prompt length | Unpatched / Triton 8-bit TP2 | **`boot79` Mode A (`dequant_to_bf16` + `cuBLASLt`)** | First token (cold bf16 prefill) | Speedup vs Triton 8-bit |
| :--- | :--- | :--- | :--- | :--- |
| **8,192 tokens (`8K`)** | `407.0 tok/s` (`20.13 s`) | **1,447.0 – 1,476.0 tok/s** | **5.55 – 5.66 s** | **3.6× faster** |
| **16,384 tokens (`16K`)** | `344.0 tok/s` (`47.63 s`) | **1,506.0 – 1,541.0 tok/s** | **10.63 – 10.88 s** | **4.5× faster** |
| **32,768 tokens (`32K`)** | `306.0 tok/s` (`107.08 s`) | **1,487.4 – 1,504.0 tok/s** | **21.79 – 22.03 s** | **4.9× faster** |
| **65,536 tokens (`64K`)** | `270.0 tok/s` (`242.73 s`) | **1,401.2 – 1,406.0 tok/s** | **46.61 – 46.77 s** | **5.2× faster** |

The 8-bit prefill was the structural wall of this project: stock 8-bit TP2 reads the 27B
weights at ~4× the bandwidth of 4-bit, through a dequant-GEMV path that was never meant for
large `M`. Mode A dequantizes the tile once into bf16 registers and hands it to `cuBLASLt`,
which runs on GB10's tensor cores — and at 64K it is *faster than it is at 8K*, because the
kernel is compute-bound, not bandwidth-bound, once `M ≥ 512`.

## What it costs in quality

4,096-token evaluation (wikitext-2 and CPython source, 8 sequences, two 2,048-token chunks so
the second pass reads cached keys) against an all-bf16 reference. The point of the table: this
build locks `PREFILL_FP8=0` — **bf16 prompts, FP8 cache only** — because 8-bit weights carry
`256` levels per group where 4-bit carries `16`, so the prompt path no longer needs to be the
weak link.

| Configuration | Weight quantization | Perplexity (4-bit ref) | Prompt / cache KL divergence | Top-1 token agreement | Used in this repo? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Reference: all `bf16`** | 16-bit / 8-bit | `3.316` | `0.0000` | `100.0%` | Reference baseline |
| **`FP8 cache only` (`PREFILL_FP8=0`, `KV_DTYPE=fp8`)** | **8-bit affine (`gs=64`)** | **`3.315`** | **`0.0031`** | **`98.7%`** | **✅ yes — `boot79` default (`13.5×` lower KL)** |
| `FP8 prompts only` (`PREFILL_FP8=1`, `KV_DTYPE=bf16`) | 4-bit affine (`gs=64`) | `3.397` | `0.0435` | `94.1%` | ❌ no (degrades prompt fidelity) |
| `Both FP8` (upstream 4-bit default) | 4-bit affine (`gs=64`) | `3.387` | `0.0419` | `94.0%` | ❌ no (`+4.1%` code perplexity penalty) |

Greedy open-ended replies diverge from the bf16 reference after a few tokens, as they do for
any numeric change; a 12-question reasoning probe scored 12/12 on the FP8 cache, and the
needle is found at 195k and — with YaRN — at 491k tokens. Deterministic-output parity across
builds: code prompt `39191bf72982` is byte-canonical 4/4 on `boot79`; the prose prompt is
deterministic within a build (4/4) and build-sensitive across builds (argmax ties under
W8A16 + FP8 KV) — `9ae72b23bafa` on `boot79`, `88ed89a747dc` on `w78`.

## Architecture: 1× Spark 4-bit vs 2× Spark 4-bit vs this build (2× Spark 8-bit hybrid)

| Dimension | Upstream `1× Spark` 4-bit (`TP=1`) | Prior `2× Spark` 4-bit (`TP=2`) | **This repo: `2× Spark` 8-bit hybrid (`boot80` `TP=2`)** |
| :--- | :--- | :--- | :--- |
| **Nodes & interconnect** | 1× DGX Spark | 2× DGX Spark (200GbE RoCEv2) | **2× DGX Spark (200GbE ConnectX-7 RoCEv2, MTU 9000)** |
| **Weight precision** | 4-bit MLX (`~15 GiB`) | 4-bit MLX (`~7.5 GiB/rank`) | **8-bit MLX (`group_size=64`, `~14.6 GiB/rank`)** |
| **Prompt prefill precision** | `PREFILL_FP8=1` (`0.0419` KL) | `PREFILL_FP8=1` (`0.0419` KL) | **`PREFILL_FP8=0` (bf16, `0.0031` KL + Mode A `cuBLASLt`)** |
| **`spark1` memory used / free** | `96.7 GiB` used / `~24 GiB` free | `57 GiB` used / `64 GiB` free | **`32.5 GiB` used / `88.0 GiB` free (`+64 GiB` vs 1× Spark!)** |
| **`spark2` memory used / free** | N/A | `69 GiB` used / `52 GiB` free | **`42.0 GiB` used / `79.0 GiB` free** |
| **Active streams & window** | `8 × 262,144` (`78 GiB` pool) | `8 × 262,144` (`39 GiB/rank`) | **`4 × 262,144` (`1.05M` pooled FP8 KV tokens)** |
| **Multi-turn prefix cache** | `12` slots (unpinned) | `12` slots (auto) | **`32` slots (`16 GiB` pinned GDN + KV store, `102 ms` warm TTFT)** |
| **Drafter NCCL collectives** | `0` (single GPU) | `12 AllGather / round` | **`0.0 AllGather / round` (`0008-A` rank-local `DFlash2`)** |

The 8-bit weights cost ~7 GiB more per rank than 4-bit and buy back the prompt-fidelity headroom:
the build serves 4 full 262k-token windows with 88 GiB to spare on `spark1`, against ~24 GiB on
a 1× Spark running the 4-bit 8-stream configuration. Memory measured at the `boot79` seal
(2026-10-26, pool warm); an idle check the next day reads 30/33 GiB used on the two nodes.

## The patch & kernel stack

Everything the engine loads at boot, in load order. The shadow package at `patches/` is first
on `PYTHONPATH` inside both containers; `sitecustomize.py` is the loader and each hook is
env-gated with a kill-switch.

| # | Patch / file | What it does | Verified state |
| :--- | :--- | :--- | :--- |
| 0001 | `0001-image-video.patch` | Images + video content parts (50 images / 2 videos per request) | engine feature, inherited |
| 0002 | `0002-yarn-1m.patch` | YaRN rope rescaling → 1,048,576-token window | needle found @ 491k |
| 0003 | `0003-fp8-kv-cache.patch` | e4m3 KV storage, 32 KiB/token (half of bf16) | parity 6/6 token hashes |
| 0004 | `0004-memory-reserve.patch` | `TENSORFOLD_MEMORY_RESERVE_GIB` budget control | 24 GiB symmetric |
| 0005 | `0005-pinned-kv-pool.patch` | Pinned KV pool reservation | 16 GiB/rank, kept |
| 0006 | `0006-fused-w8a16-affine.patch` + **hybrid wrapper `tp2w8a16.py` (`e3f8d40c`)** | Fused W8A16 dequant GEMM; the wrapper routes `M≥512` → Mode A CUDA, `M<512` → v10e Triton | **resident, this repo's core** |
| 0007 | `0007-tp2-multistream-batch.patch` (`c32e6a1f`) | C=4 multi-stream batch gate + 0007 diagnostics (`fwd/round=1.00`, target-colls/round=128.0) | gate measured 128.0 / 0.0 |
| 0008-A | `0008a-replicated-dflash2.patch` | Rank-replicated `DFlash2` drafter — **0 drafter collectives** | draft-colls/round = 0.0 |
| 0008-D2 | `0008d2-checkpoint-slots32.patch` | `--checkpoint-slots 32` reservation cap (`TF_SLOT_KV_CAP=16384`) + B2 hook block | 32 slots live, 102 ms warm TTFT |
| v10e | `tp2draftgraph.py` (`3ccfeb2b`) | Eager static-buffer draft staging for all S ∈ 1..4 (CUDA graphs parked — boot-state GPU issue) | c4d 208.0 on `3ccfeb2b` engine |
| CUDA | `w8a16_cuda_qmm.cpython-312-aarch64-linux-gnu.so` (**`b8440d18`**, sm_121) | Native `mma.sync` bf16 W8A16 extension: `cp_wait<0>()` K-loop drain + `dequant_to_bf16` (Mode A) | **md5 `b8440d18…` live in both containers** |

### Determinism & numerical verification (`vdrain`, `b8440d18`)

| Check | Result |
| :--- | :--- |
| Call-to-call flips, 50 L2-flushed calls, `M=1` bf16 / `M=1` f32 / `M=17` / `M=1024` | **`0 / 0 / 0 / 0`** |
| `compute-sanitizer` (memory + init) over full probe sweep | **0 errors** |
| `rel_err` vs FP64 reference | **0.189% – 0.270%** |
| RoCE L2-flush divergence, 8K/16K/32K/64K prefill (flush on vs off) | **`0 / 0 / 0 / 0`** identical token streams, 0 dropped packets |

The drain fix (`cp_wait<0>()` at L134 of the K-loop) is numerically inert — it gates readiness,
not accumulation order — which is why the pure-CUDA decode stays ~7–11 tok/s below Gen-1: that
cost is `cp_wait` serialization, not arithmetic. The kernel is kept resident for Mode A
prefill, where it wins 3.6–5.2×. Full matrix: `bench/scorecards/vdrain_matrix.txt`.

## Settings (`boot79` production)

The effective configuration of the resident build. `tp2.env` (loaded by both rank scripts)
plus the `run_rank{0,1}.sh` docker flags:

| Setting | Value | Why |
| :--- | :--- | :--- |
| `TP2_DRAFT_REPL` | `1` | 0008-A replicated drafter → 0 drafter collectives (`0` = stock 12 AllGather/round) |
| `TP2_W8A16` | `1` | Install the 0006 W8A16 hook (`0` = stock dequant GEMV, the ~21 tok/s reference) |
| `TP2_W8A16_DECODE` | `0` | **Hybrid default**: Triton decode `M<512`. `1` = pure-CUDA `b8440d18` decode (rollback, `w78`) |
| `TP2_PREFILL_MODE` / `TP2_PREFILL_THRESH` | `a` / `512` | Mode A CUDA `dequant_to_bf16` + `cuBLASLt` for `M ≥ 512` |
| `TP2_BATCH_VERIFY` | `1` | 0007 multi-stream batch gate + diagnostics |
| `TP2_DRAFT_GRAPH` / `TP2_WIDTH_PROBE` | `1` / `1` | v10e eager draft staging, all S values |
| `TP2_CUDA_GRAPHS` | `0` | CUDA-graph path parked (boot-state GPU issue, 4.6× p1d regression) |
| `TF_SLOT_KV_CAP` | `16384` | 0008-D2: clamp per-slot KV reservation to a 16k boundary |
| `--checkpoint-slots` | `32` | 32 GDN+KV prefix slots (multi-turn agent cache) |
| `--prompt-cache-gib` | `16` | Pinned prompt-cache budget, fixed (no auto-grow) |
| `--no-prefill-fp8` | set | `PREFILL_FP8=0`: bf16 prompt activations (the quality lock) |
| `TENSORFOLD_KV_DTYPE` / `KV_POOL_GB` | `fp8` / `16` | FP8 KV, 16 GiB pinned pool per rank |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `24` | Symmetric hard buffer on both ranks |
| `--parallel` / `--context` | `4` / `262144` | 4 streams × 262,144-token window |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | Single stream context for deterministic ordering |
| `NCCL_BUFFSIZE` | `8388608` (effective; `tp2.env` default `33554432`) | RoCEv2 all-reduce buffer |
| `--cpuset-cpus` | `5-9,15-19` | X925 core pinning, dispatch off the engine cores |
| `--ipc=host`, `--shm-size` | host / `8g` | NCCL shared memory |
| **`.so` load path** | **`/patches/` volume root** | **TensorFold imports `w8a16_cuda_qmm` with `PYTHONPATH=/patches`. The `.so` must sit at the *root* of the `/patches` bind mount in both containers — inside `patches/w8a16_cuda/` it is never importable and the hook silently falls back to stock (~45 tok/s phantom). Container stop wipes the container-layer copy; the host bind mount is the only durable home.** |

## Quick start

Prereqs: two DGX Sparks (GB10, 128 GiB unified), a direct 200Gbps ConnectX-7 RoCEv2 link
(MTU 9000, PFC tuned), Docker with the NVIDIA runtime, the `tensorfold-qwen38-27b:v0.6.0-idlebell`
image, and this repository cloned to `~/qwen38-tf-cluster` on both nodes with `patches/`
bind-mounted to `/patches` (the launcher does it).

```bash
# 1. rank0 (head) on spark1
cd ~/qwen38-tf-cluster && ./launch-tp2.sh          # ROLE defaults to head; $1 = worker on rank1

# 2. rank1 on spark2
ssh gsgun@<spark2> 'cd ~/qwen38-tf-cluster && ./launch-tp2.sh worker'

# 3. verify (spark1)
curl -s http://127.0.0.1:8894/health               # {"ok": true, ...}
md5sum /patches/w8a16_cuda_qmm.cpython-312-aarch64-linux-gnu.so   # b8440d18c2f473a9829088970cb13f43
```

The launcher is per-node and positional (`./launch-tp2.sh [worker]`) — `ROLE=worker ./launch-tp2.sh`
starts a *rank0-named* container on spark2; pass `worker` as `$1`.

In the fleet this model is resident through llama-swap (`qwen3.8-27b-tp2` → `:8894`,
persistent, TTL 0) and routed through LiteLLM as the `auto` leg; a routed request re-adopts
the engine if it ever reports unloaded, so no manual restart is needed after a boot.

### Benchmarking

```bash
./bench/w8a16_c4.sh <label> <draft true|false> [N] [p|c|m]   # C=4 battery (p=prose c=code m=mixed)
./bench/parity_c4.sh                                          # C=1/C=4 greedy parity, 4/4 byte-exact expected
./bench/bench_native.sh                                       # unpatched / Triton reference legs
```

Scorecards: `bench/scorecards/` (`boot79`, `w78`, `v10e`, `vdrain` matrix).

## Requirements

- 2× DGX Spark (GB10, 128 GiB unified memory each), 200GbE ConnectX-7, RoCEv2, MTU 9000.
- Docker with the NVIDIA container runtime; user in the `docker` group.
- ~40 GiB disk per node: checkpoints under `~/.cache/huggingface`, the engine image.
- Optional: `hf` CLI on the hosts (faster, resumable downloads).

## Images and video

Inherited from the upstream engine: up to 50 images (`TENSORFOLD_MAX_IMAGES`, 10 MB each,
16,384 image tokens) and 2 videos per request (2 fps, 256 frames, 16,384 video tokens), as
OpenAI-style content parts. Data URLs by default; `VISION_URLS=1` allows public `https://`
URLs. Media prompts are not kept for prefix reuse.

## Repository layout

```
README.md  CHANGELOG.md  CREDITS.md  LICENSE  NOTICE  LICENSES/
launch-tp2.sh  stop-tp2.sh  run_rank0.sh  run_rank1.sh  tp2.env
patches/
  sitecustomize.py                 # loader (PYTHONPATH-first shadow package)
  tp2w8a16.py                      # 0006 hybrid wrapper (e3f8d40c) — core of boot79
  tp2batchverify.py  tp2draftrepl.py  tp2slotcap.py  tp2widthprobe.py  tp2cudagraphs.py
  tp2draftgraph.py                 # v10e eager draft staging (3ccfeb2b)
  prefill_mode.txt                 # Mode A marker
  0001…0005 (engine build patches)  0006  0007  0008a  0008d2  0009
  w8a16_cuda_qmm.cpython-312-aarch64-linux-gnu.so   # b8440d18 — must sit at /patches/ root
  w8a16_cuda/
    qmm_kernel.cu  qmm.cpp  setup.py  test_parity.py   # b8440d18 production sources
    archive/     # Gen-1: qmm_kernel_gen1_{recovered,initial}.cu (a6d64116), .so.gen1-2f84,
                 #        qmm_w8a16_gen1.cpp, GEN1_RESCUE.md
bench/
  bench_native.sh  w8a16_c4.sh  parity_c4.sh
  scorecards/  boot79_hybrid  w78_native_cuda  v10e  vdrain_matrix
```

## Credits

This repository — the dual-node 8-bit `TP=2` hybrid, the fused W8A16 CUDA extension, the
replicated-drafter and checkpoint-slot patches, and every number in the `boot79`/`boot80`/`w78`/`v10e`
columns — is the work of **gsgun**.

Adapted from the **[MiaAI-Lab](https://github.com/MiaAI-Lab) single-Spark 4-bit recipe** (MIT): the
`start.sh`/`stop.sh`/`scripts/` launcher, the README layout and measurement method, and the video code in
`patches/0001` — its notice is kept in `LICENSES/MIT-MiaAI-Lab.txt`. It runs on
[TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0 from v0.6.0) by Ash Hart and the TensorFold
contributors; the model checkpoints (`TensorFold/Qwen3.8-27B-MLX-8bit`, `z-lab/Qwen3.8-27B-DFlash2`) are © their
respective authors. See `CREDITS.md` and `LICENSE` (Apache-2.0).

## boot80 Production Verification Scorecard (Dual DGX Spark GB10 · 8-Bit TP=2)

Deployed and resident at :8894 across spark1 (rank 0) + spark2 (rank 1).

### Prefill Throughput (Phase 2 FP8 Gather 0009 vs Mode A Baseline)
| Context Length | Stock Mode A (tok/s) | FP8 Gather 0009 (tok/s) | Speedup |
| :--- | :--- | :--- | :--- |
| **8K** | 1308.4 | **1565.9** | **+19.7%** |
| **16K** | 1461.7 | **1579.9** | **+8.1%** |
| **32K** | 1466.6 | **1572.9** | **+7.2%** |
| **64K** | 1416.6 | **1522.0** | **+7.4%** |

### Decode and Multi-Stream Scaling (Phase 1 v10e Eager Staging)
| Metric | Baseline | boot80 Score | Status |
| :--- | :--- | :--- | :--- |
| **c1d (Code, C=1, DFlash2)** | 94.0 tok/s | **93.46 tok/s** | Verified |
| **p1d (Prose, C=1, DFlash2)** | 45.4 tok/s | **48.44 tok/s** | +6.7% |
| **c4d (Code, C=4, DFlash2)** | 196.1 tok/s | **210.55 tok/s** | **+7.3%** |
| **p4d (Prose, C=4, DFlash2)** | 111.4 tok/s | **123.60 tok/s** | **+10.9%** |
| **m4d (Mixed, C=4, DFlash2)** | 111.5 tok/s | **126.36 tok/s** | **+13.4%** |
| **c4off (Drafter Off, C=4)** | 37.5 tok/s | **38.60 tok/s** | Verified |

### Reliability and Verification Battery
- **Target Collectives:** 128.0 / round (2 x 64 layers, zero collective leakage)
- **Draft Collectives:** 0.0 / round (replicated rank-local DFlash2)
- **det500:** BYTE_EXACT PASS (2 x 550 tokens, SHA256 079bd6d2f3c557d1)
- **C=4 Cross-Stream Parity:** 8/8 PASS (Prose 9ae72b23bafa x 4, Code 39191bf72982 x 4)
- **Snapshot Persistence:** --snapshot-dir /snapshots mapped to /var/tmp/tf-snapshots

---

## Live state — 2026-10-28 (current production, `boot91`)

Verified against the running pair: `qwen38-tf-rank0` on spark1 + `qwen38-tf-rank1` on spark2, `:8894`.

| | |
| --- | --- |
| **Image** | `tensorfold-qwen38-27b:v0.6.0-idlebell` (both ranks) |
| **`.so`** | `b8440d18c2f473a9829088970cb13f43` (matches the seal) |
| **Draft module** | `tp2draftgraph.py` `44e592fd240a0c775dcec90378b6aefc` (v11, `S≤8`) |
| **Geometry** | **PAR=6**, `CTX=262,144`/slot → **1,572,864 pooled tokens** (24 GiB fp8 KV/rank) |
| **Pair** | `qwen38-tf-rank0`+`rank1`, `RestartCount=0`, `Up` |
| **Env** | `TP2_PREFILL_FP8_GATHER=1`, `TP2_PREFILL_MODE=a`, `TP2_V11_GRAPH=0` (v10e eager), `TP2_CUDA_GRAPHS=0`, `TP2_DRAFT_REPL=1`, `TP2_W8A16=1` (`fast`), `KV_DTYPE=fp8`, `MEMORY_RESERVE_GIB=16`, `NCCL_IB_QPS_PER_CONNECTION=4` |
| **Health** | `ok:true`, `rounds_total=24,111`, `drafted=234,468`, `accepted=70,170` (**29.9%**), `max streams=6` |

> The live deployment is **PAR=6 × 262,144 = 1,572,864 pooled tokens** at 24 GiB fp8 KV/rank. The
> 1,048,576-token single-slot window in the tables above is the YaRN rope cap (`0002`, needle found @
> 491k); boot91 runs six 262K slots rather than one 1M slot, trading depth-per-request for concurrency.

## boot91 measured scorecard (PAR=6 · CTX=262,144 · 2026-10-28 21:48–21:57 PDT)

| Arm | tok/s | detail |
| --- | ---: | --- |
| **c1d** (prose, 1×512) | **45.59** | 93 rounds, 418 accepted, 11.23 s wall |
| **p1d** (code, 1×512) | **45.69** | 93 rounds, 418 accepted, 11.10 s |
| **c4d** (prose, 4×512) | **108.51** | 112 rounds, 399 accepted, 18.87 s |
| **p4d** (code, 4×512) | **111.32** | 53 rounds, 458 accepted, 18.40 s |
| **c6d** (prose, 6×512) | **154.54** | 115 rounds, 396 accepted, 19.88 s |
| **c4off** (drafter off) | — | 511 rounds, 0 accepted, 56.71 s |

Prefill (first rep, cold): 8K **1668** · 16K **1572** · 32K **1373** · 64K **1321** tok/s (reps 2–3 are
prefix-cached, ~0.13–0.24 s).

> **The headline:** boot91 decode is **~2× slower than boot90** (c1d 45.6 vs 85–89 tok/s; c4d 108.5 vs
> ~201; c6d 154.5 vs ~233). The regression tracks the v6-stride→v11 draft-module swap and is pending a
> clean v6/v6 A/B; the operator kept v11 (`S≤8`).

## Known issues (open, measured)

1. **Uptime rank-divergence churn (#993)** — decode degrades with pair uptime: a clean 5 h warm leg on 2026-10-27 read
   **33–40% of the pre-8-bit baseline** (`prose 46.5 → 34.7 tok/s`, `agg6 84.7 → 25.7 tok/s`, **−70% over 4.5 h**),
   consistent with accumulating rank divergence (no pair rebuild during the window). A rebuild restores near-seal
   numbers (`c1d 83.7`, `c4d 185.9`, `p4d 107.7` post-remediation). **5 flightrecs in 24 h**, still OPEN on 2026-10-28.
   Stopgap: watchdog rebuild on divergence; a durable fix tops the To-do list.
2. **260k needle intermittently FAILS** on every 8-bit-era run (2026-10-09, 10-27 leg 1 + leg 2) vs **PASS** on the
   10-02 pre-8-bit reference — long-context retrieval corruption that appeared with the 8-bit cutover. The 8-bit
   checkpoint is the prime suspect; `v10e`-eager is also in scope. Not reproducible on a fresh rebuild.
3. **Draft acceptance is the weak spec-decode leg** — **28%** at the `boot80` seal (725,183 drafted / 203,460 accepted
   @ 5 h), **39%** live post-rebuild. The DFlash2 drafter proposes well but the target rejects ~2/3 of drafts; this
   caps decode gain from speculation.
4. **`c4off` (drafter off) regressed** to **38.60 tok/s** vs `boot79` 246 tok/s (−84%) — under investigation; the
   serial reference is not the production path, but the drop suggests the batch gate interacts with drafter-off.
5. **Engine image is `v0.6.0` (2026-10-01) vs upstream `v1.1.8` (2026-10-28)** — five patch releases behind. The
   `v0.6.0 → v1.1.8` canary is **gated** and deferred to an operator window (no unattended swap).
6. **boot91 ~2× decode regression vs boot90** — c1d **45.6** vs **85–89** tok/s (c4d 108.5 vs ~201, c6d 154.5 vs
   ~233). The v6-stride→v11 draft-module swap is the prime suspect; a clean v6/v6 A/B (v6 draft + boot91 launch) is
   pending. Operator kept v11 (`S≤8`).
7. **RoCE link degraded** — **109.16 Gb/s of the 200G** link, with lane-0 PHY errors (pre-existing). Caps cross-rank
   collectives; not caused by boot91.
8. **boot91 Phases 2–3 not done** — the router/governor (`decision_router_gguf`, `governor.py`), LiteLLM, llama-swap
   and watchdog were **not touched**; boot91 is the engine geometry only.
9. **boot91 was uncommitted** — spark1's repo is at `f0e2f6f boot90`; the GitHub fork was at boot80 docs until this
   commit. The live `launch-tp2.sh` (`194b3996`) and `tp2draftgraph.py` (`44e592fd`) are now synced into this repo.

## Learned

- **Prefill is compute-bound, not bandwidth-bound** once `M ≥ 512` — Mode A `dequant_to_bf16` + `cuBLASLt` makes 64K
  *faster* than 8K (1,522 vs 1,566 tok/s), and FP8 prefill AllGather (`0009`) cuts the RoCE payload ~50% for another
  +7–20% across 8K–64K.
- **The `.so` must live at the `/patches` mount root.** Inside `patches/w8a16_cuda/` it is never importable and the
  hook silently falls back to stock (~45 tok/s phantom). Container stop wipes the layer copy; the host bind mount is
  the only durable home.
- **Numerical parity is cheap to hold, expensive to lose** — `b8440d18` is 0/0/0/0 flips across 50 L2-flushed calls
  and 0 compute-sanitizer errors; byte-exact det500 (`079bd6d2…`) and 8/8 cross-stream parity are the regression gates.
- **The 8-bit cutover bought quality but exposed long-context instability** — FP8 cache keeps KL at 0.0031 and top-1
  at 98.7%, but the 260k needle regression and #993 churn are the price; both are checkpoint/engine-level, not
  patch-level.
- **Speculative decode is only as good as its acceptance rate** — 28–39% acceptance means most proposals are rejected;
  tune the draft tree / acceptance threshold before adding more streams.
- **The launcher hardcodes `PAR`/`CTX`** — boot91's geometry is edited in `launch-tp2.sh`, not overridable by env;
  changing slots means editing the file on both ranks and relaunching.
- **There is no systemd unit for the pair** — launch is `stop-tp2.sh` + `nohup qwen38-tf-swap-up.sh`; no restart-on-boot.
- **The v11 draft suite costs ~2× decode vs v6-stride** — acceptance alone doesn't explain it; the draft-suite change
  is the variable to A/B before trusting boot91 as production.
- **Benchmark harness pitfalls** — arg order matters (`bench_model.py --model … --base-url …`), and compound ssh
  commands can truncate on quoting; run one command per ssh call.

## To do

- [ ] **Durable #993 fix** — root-cause the rank divergence that degrades decode over uptime; watchdog rebuild is a
      stopgap, not a fix.
- [ ] **260k needle** — bisect the 8-bit checkpoint vs `v10e`-eager to isolate the long-context retrieval regression;
      add a needle gate to the nightly battery.
- [ ] **Draft acceptance** — tune the DFlash2 draft tree / acceptance threshold to push 28–39% toward 60%+;
      re-measure `c1d`/`c4d` with a better drafter.
- [ ] **`c4off` regression** — trace the drafter-off batch-gate interaction (38.60 vs 246 tok/s).
- [ ] **`v0.6.0 → v1.1.8` canary** — staged, gated; run in an operator window, verify `.so` load path + parity + needle
      before promoting.
- [ ] **Commit boot91** — create the boot91 commit in spark1's repo and push to `origin/main` (the synced
      `launch-tp2.sh` + `tp2draftgraph.py` are staged in this fork now).
- [ ] **Clean v6/v6 A/B** — v6-stride draft + boot91 launch, to isolate the ~2× regression before promoting boot91.
- [ ] **boot91 Phases 2–3** — router/governor re-point, LiteLLM, llama-swap, watchdog (untouched).
- [ ] **Re-run parity** — boot91 determinism / cross-stream parity gate.
- [ ] **RoCE fix** — lane-0 PHY errors; restore ~200 Gb/s before trusting cross-rank numbers.
- [ ] **Reconcile docs with live geometry** — PAR=6 × 262,144 (1,572,864 pooled, 24 GiB fp8/rank); `RESERVE_GIB=16`.
