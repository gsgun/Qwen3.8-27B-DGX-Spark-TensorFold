#!/bin/bash
# Qwen3.8-27B TensorFold v0.6.0 Dual-Node TP=2 — 8-Bit production baseline
# (Spark 1 Head + Spark 2 Worker) — non-destructive sibling of launch-tp2.sh.
#
# MiaAI-Lab Qwen3.8-27B-DGX-Spark-TensorFold recipe adaptation:
#   - MODEL: 8-bit checkpoint (~15 GiB weights/rank under TP=2) instead of 4-bit.
#   - --parallel 4 --context 262144 = 1,048,576 pooled tokens (Phase 3 10-23:
#     4 x 262,144 native window). fp8 KV is 16 KiB/token/rank (engine geometry,
#     16 attn layers x 4 kv-heads/rank x 128 dim x K+V x 1 byte), so the pooled
#     slots cost exactly 16 GiB KV/rank on demand -> ~44 GiB used / ~77 GiB
#     free per node at 4x262k (measured 43 GiB at 8x65536 + 8 GiB KV delta).
#   - TENSORFOLD_KV_POOL_GIB: INACTIVE at TP=2 — engine.py:121 pins the
#     pre-faulted pool only when tp==1 (MiaAI Patch 0005 single-GPU gate).
#     At TP=2 the KV grows on demand; the value is kept as the symmetric
#     single-GPU-fallback pin (never auto).
#   - --no-prefill-fp8: bf16 prompt prefill. TF v0.6.0 has no TENSORFOLD_PREFILL_FP8
#     env var — the flag IS the mechanism, and omitting it does NOT turn it off:
#     the Qwen3.8 CUDA family default (serve_options.py) is fp8. The 10-20 8-bit
#     cutover omitted the flag and still ran fp8 prefill; this pins it off.
#   - TENSORFOLD_MEMORY_RESERVE_GIB=24 SYMMETRIC (was 16 worker / 8 head): hard buffer
#     for the co-hosted workloads (d2/d3 decision stacks, llama-swap, surya, kokoro)
#     and OS/RoCE bounce buffers on a node that is NOT the sole tenant.
#   - DISABLE_DRAFTER gate (top of file): default 1 appends --no-drafts (Phase A),
#     isolating whether the 96-collective deficit (2549 enqueued - 2453 completed)
#     is DFlash2 draft verification diverging across ranks. 0 = Phase B, DFlash2 on.
#   - PERF PASS (2026-10-23): NCCL_BLOCKING_WAIT removed (was 1 — forced a CPU
#     sync of the CUDA stream on every collective across all 64 layers +
#     DFlash2 verification, destroying async kernel pipelining); bare
#     NCCL_ASYNC_ERROR_HANDLING dropped in favor of TORCH_NCCL_ASYNC_ERROR_HANDLING=1
#     (the variable torch actually reads); NCCL_PROTO=LL,LL128,Simple enables
#     the low-latency small-message protocols for TP=2 decode all-reduces.
#   - CPU pinning: BOTH ranks -> --cpuset-cpus 5-9,15-19 (10x Cortex-X925
#     3.9GHz). Unpinned dispatch threads migrate to the 2.8GHz A725 efficiency
#     cores (0-4,10-14) and stall collective enqueueing.
#   - The 60s torch timeout clamp (sitecustomize) is kept — fast, bounded
#     failure instead of a 300 s hang.
#   - Prompt cache: --prompt-cache-gib is NOT passed, so the engine auto-sizes
#     it to min(16 GiB, RAM/8) = 16 GiB per node with grow_checkpoints (cli.py:
#     459-462) — multi-turn prefix reuse IS active (verified live: cached_tokens
#     accumulating) and --checkpoint-slots defaults to 3/lane (two ranks each
#     reserve a window). Re-add an explicit --prompt-cache-gib to pin it.
#   - YARN_FACTOR opt-in: unset by default -> TENSORFOLD_YARN_FACTOR empty ->
#     byte-exact unscaled rotary embeddings (yarn.py: factor 1, scale 1.0).
#     YARN_FACTOR=4 stretches the native 262144 window to 1,048,576 for
#     single-stream 1M runs.
#   - CUDA_DEVICE_MAX_CONNECTIONS=1 (Phase 3 10-23 spec): serializes compute
#     and RoCE NET/IB streams in hardware queues. CAVEAT: an earlier A/B on this
#     engine (pre-0006) showed C=4 21.5->8.0 tok/s with it set; re-validated at
#     the new baseline in the Phase 3 pass before locking.
#   - --max-num-batched-tokens: no such flag in TF v0.6.0 (it is a vLLM flag — the
#     "cold-prefill stability fix" carried over from the c2 vLLM stack); CUDA prefill
#     chunking here is engine-internal, so it is not passed.
#
# Same container names as launch-tp2.sh (qwen38-tf-rank0/1): the duplicate-launch
# guard keeps the two variants mutually exclusive, llama-swap / watchdog tracking is
# unchanged, and rollback is one command: stop-tp2.sh, then launch-tp2.sh (4-bit).
set -euo pipefail
# Operator env overlay: tp2.env (if present) is loaded on BOTH ranks — rank1 is
# launched via plain ssh (no SendEnv, see qwen38-tf-swap-up.sh), so a file is
# the only channel that reaches both ranks identically. Kept for the duration
# of a test cycle, removed before the next production boot.
[ -f "$HOME/qwen38-tf-cluster/tp2.env" ] && { set -a; . "$HOME/qwen38-tf-cluster/tp2.env"; set +a; }
ROLE="${1:-head}"
MODEL=TensorFold/Qwen3.8-27B-MLX-8bit
IMAGE=tensorfold-qwen38-27b:v0.6.0-idlebell
PORT=8894
CTX=262144    # Phase 3 10-23: 4 x 262,144 = 1,048,576 pooled slots (16 GiB fp8 KV/rank)
PAR=4
MASTER=10.100.8.2
SOCK_IF=enp1s0f1np1
IB_HCA=rocep1s0f1
# Co-host protection: symmetric hard reserve on BOTH ranks (never auto, never 0).
RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-24}"
# KV pool pin: single-GPU fallback only — engine.py:121 gates the pre-faulted
# pool to tp==1, so at TP=2 this is a no-op (KV grows on demand). Never auto.
KV_POOL_GIB="${TENSORFOLD_KV_POOL_GIB:-16}"
# Phase A drafter isolation gate: 0 (default) enables DFlash2; 1 appends --no-drafts.
# 2026-10-22: default flipped to 0 after Phase B promotion; the llama-swap
# wrapper (qwen38-tf-swap-up.sh) re-boots the pair without env, so the default IS
# the production drafter state.
DISABLE_DRAFTER="${DISABLE_DRAFTER:-0}"
# Opt-in YaRN: unset -> byte-exact native RoPE; YARN_FACTOR=4 -> 1M window.
YARN_FACTOR="${YARN_FACTOR:-}"
# Phase-2 timing probe (0 = off): capture N decode rounds, auto-unhook.
TF_PROF_ROUNDS="${TF_PROF_ROUNDS:-0}"
# 0008-A (2026-10-24): replicated rank-local DFlash2 — TP2_DRAFT_REPL=1 (default)
# forces the stock TP2 path: rank-0-only world=1 drafter (patches/tp2draftrepl.py),
# 0 drafter NCCL collectives/round (was 12: 10 _row + 2 candidate gathers).
# TP2_DRAFT_REPL=0 restores the sharded joint drafter (boot-10 behavior).
# 0008-D: --checkpoint-slots 32 (default 12 = 3/lane) — 32 GDN/KV prefix entries
# in the 16 GiB budget: ~10+ concurrent multi-turn agents, no LRU eviction at 4-5.
# 0008-D2 (2026-10-24): admission sizes each slot at the FULL window (4.072 GiB
# @262144), so keep=32 needs 153.5 GiB and refuses to boot. Cap each slot's
# attn-KV reservation at a 16k boundary (patches/tp2slotcap.py): keep=32 then
# admits at 64.1/62.9 GiB per rank (below the 64.6 GiB boot-10 baseline).
# TF_SLOT_KV_CAP=0 restores the stock full-window reservation (0008-D infeasible).
TF_SLOT_KV_CAP="${TF_SLOT_KV_CAP:-16384}"
HF_CACHE=$HOME/.cache/huggingface
KERNEL_CACHE=$HOME/.cache/tensorfold-qwen38-27b
PATCH_DIR=$HOME/qwen38-tf-cluster/patches
mkdir -p "$KERNEL_CACHE" "$PATCH_DIR"

# Ensure sitecustomize timeout patch exists
cat << 'PYEOF' > "$PATCH_DIR/sitecustomize.py"
import os, datetime
import torch.distributed.constants
import torch.distributed.distributed_c10d

# Clamp default NCCL timeout from 600s down to 60s to prevent 10-minute 100% CPU lockups
timeout_sec = int(os.environ.get('TORCH_NCCL_COLLECTIVE_TIMEOUT_SEC', 60))
torch.distributed.constants.default_pg_nccl_timeout = datetime.timedelta(seconds=timeout_sec)
torch.distributed.distributed_c10d.default_pg_nccl_timeout = datetime.timedelta(seconds=timeout_sec)
PYEOF

if [ "$ROLE" = worker ]; then
  NAME=qwen38-tf-rank1
  RANKARGS=(--rank 1)
  # PERF PASS 10-23: pin to the 10x 3.9 GHz Cortex-X925 cores (was 4-19 —
  # included the 2.8 GHz A725 core 4 where dispatch threads stalled).
  # Co-hosts (d3 sidecar, surya, kokoro) keep the A725 ring.
  CPU_AFFINITY=(--cpuset-cpus 5-9,15-19)
else
  NAME=qwen38-tf-rank0
  RANKARGS=(--rank 0 --name Qwen3.8-27B --host 0.0.0.0 --port "$PORT")
  # PERF PASS 10-23: head was unpinned -> dispatch threads drifted onto the
  # 2.8 GHz A725 efficiency cores (0-4,10-14) and stalled collective enqueue.
  CPU_AFFINITY=(--cpuset-cpus 5-9,15-19)
fi

DRAFT_FLAGS=()
if [ "$DISABLE_DRAFTER" = "1" ]; then
  DRAFT_FLAGS=(--no-drafts)
fi
DRAFTS_ON=$([ "${#DRAFT_FLAGS[@]}" -eq 0 ] && echo yes || echo no)

# Guard against duplicate launch decoupling the rendezvous
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qxE "$NAME"; then
  echo "REFUSED: $NAME is already running — a duplicate launch would kill the TP=2 pair. Use stop-tp2.sh first if a relaunch is intended." >&2
  exit 1
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --gpus all --ipc=host --network host --device /dev/infiniband \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --cap-add SYS_NICE \
  "${CPU_AFFINITY[@]}" \
  -e HF_HUB_OFFLINE=0 -e HF_HUB_ETAG_TIMEOUT=10 \
  -e TENSORFOLD_KV_DTYPE=fp8 \
  -e TENSORFOLD_HOST_RESERVE_MIB=1024 \
  -e TENSORFOLD_MEMORY_RESERVE_GIB="$RESERVE_GIB" \
  -e TENSORFOLD_KV_POOL_GIB="$KV_POOL_GIB" \
  -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TP2_W8A16_DIAG="${TP2_W8A16_DIAG:-0}" \
  -e TP2_W8A16="${TP2_W8A16:-1}" -e TP2_W8A16_MODE="${TP2_W8A16_MODE:-fast}" \
  -e TP2_BATCH_VERIFY="${TP2_BATCH_VERIFY:-1}" -e TP2_BATCH_VERIFY_DIAG="${TP2_BATCH_VERIFY_DIAG:-1}" \
  -e TP2_DRAFT_REPL="${TP2_DRAFT_REPL:-1}" \
  -e TP2_CUDA_GRAPHS="${TP2_CUDA_GRAPHS:-1}" \
  -e TP2_GRAPH_D="${TP2_GRAPH_D:-0}" \
  -e TF_SLOT_KV_CAP="$TF_SLOT_KV_CAP" \
  -e TF_PROF_ROUNDS="$TF_PROF_ROUNDS" \
  -e TP2_WIDTH_PROBE="${TP2_WIDTH_PROBE:-0}" -e TP2_WIDTH_PROBE_OUT="${TP2_WIDTH_PROBE_OUT:-/probe_out/width_probe.jsonl}" \
  -e TF_PROF_KERNELS="${TF_PROF_KERNELS:-0}" \
  -e TENSORFOLD_YARN_FACTOR="$YARN_FACTOR" \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e TENSORFOLD_MAX_IMAGES=50 -e TENSORFOLD_IMAGE_TOKENS=16384 -e TENSORFOLD_VIDEO_TOKENS=16384 \
  -e NCCL_SOCKET_IFNAME=$SOCK_IF -e GLOO_SOCKET_IFNAME=$SOCK_IF \
  -e NCCL_IB_HCA=$IB_HCA -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=0 \
  -e NCCL_BUFFSIZE=8388608 \
  -e NCCL_NET_PLUGIN=none -e NCCL_NET=IB \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_MULTI_RANK_GPU_ENABLE=1 \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60 \
  -e TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
  -e TORCH_NCCL_COLLECTIVE_TIMEOUT_SEC=60 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e OMP_NUM_THREADS=10 -e MKL_NUM_THREADS=10 \
  -e PYTHONPATH=/patches \
  -v "$PATCH_DIR":/patches:ro \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$KERNEL_CACHE":/cache \
  -v "$HOME/qwen38-tf-cluster/probe_out":/probe_out \
  "$IMAGE" \
  numactl --cpunodebind=0 --membind=0 tensorfold serve "$MODEL" --tp 2 --master "$MASTER" "${RANKARGS[@]}" \
    --parallel "$PAR" --context "$CTX" --prompt-cache-gib 16 --no-prefill-fp8 \
    --checkpoint-slots 32 \
    --thinking --temperature 1.0 --top-p 0.95 --top-k 20 "${DRAFT_FLAGS[@]}"   # --vision dropped (text/code/agent traffic only)

echo "launched $NAME ($(date +%T)) ctx=$CTX par=$PAR reserve=${RESERVE_GIB}GiB kvpool=${KV_POOL_GIB}GiB(yarn=${YARN_FACTOR:-native}) drafts=${DRAFTS_ON}"
docker logs --tail 5 "$NAME" 2>&1 | head -8
