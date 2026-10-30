"""Patch 0006 shadow package.

The real TensorFold package lives in dist-packages; this file (found first
because /patches is on PYTHONPATH) re-points __path__ at the real package and
installs the fused W8A16 affine kernel at import time. Submodules still load
from dist-packages. Disable with TP2_W8A16=0.
"""

import os as _os
import sys as _sys

__version__ = "0.6.0"

_REAL = "/usr/local/lib/python3.12/dist-packages/tensorfold"
__path__ = [_REAL]

if _os.environ.get("TP2_W8A16", "1") != "0":
    try:
        from .cuda.kernels import affine as _affine
        from tp2w8a16 import install as _install

        _install(_affine, diag=_os.environ.get("TP2_W8A16_DIAG", "0") == "1")
    except Exception as _e:  # an optional patch must never break the server
        _sys.stderr.write(f"tp2w8a16: install failed: {_e!r}\n")
        _sys.stderr.flush()

if _os.environ.get("TP2_BATCH_VERIFY", "1") != "0":
    try:
        from tp2batchverify import install as _install07

        _install07()
        _sys.stderr.write("tp2batchverify: installed (TP2_BATCH_VERIFY=1)\n")
    except Exception as _e:  # an optional patch must never break the server
        _sys.stderr.write(f"tp2batchverify: install failed: {_e!r}\n")
        _sys.stderr.flush()

# 0008-A: replicated rank-local DFlash2 (0 draft NCCL collectives).
# TP2_DRAFT_REPL=1 (default) forces the stock TP2 path: rank-0-only world=1
# drafter; =0 restores the sharded joint drafter (boot-10 behavior).
if _os.environ.get("TP2_DRAFT_REPL", "1") != "0":
    try:
        from tp2draftrepl import install as _install08a

        _install08a()
    except Exception as _e:  # an optional patch must never break the server
        _sys.stderr.write(f"tp2draftrepl: install failed: {_e!r}\n")
        _sys.stderr.flush()

# 0008-D2: cap checkpoint-slot KV reservation (0008-D 32 slots at 4x262144).
# TF_SLOT_KV_CAP=16384 (default) clamps each slot's attn-KV reservation to a
# 16k boundary; =0 restores the stock full-window reservation.
if int(_os.environ.get("TF_SLOT_KV_CAP", "0") or 0) > 0:
    try:
        from tp2slotcap import install as _install08d2

        _install08d2()
    except Exception as _e:  # an optional patch must never break the server
        _sys.stderr.write(f"tp2slotcap: install failed: {_e!r}\n")
        _sys.stderr.flush()

if int(_os.environ.get("TF_PROF_ROUNDS", "0") or 0) > 0:
    try:
        from tfprof import install as _install_prof

        _install_prof()
    except Exception as _e:  # an optional probe must never break the server
        _sys.stderr.write(f"tfprof: install failed: {_e!r}\n")
        _sys.stderr.flush()

# 0008-B2 Step-1 probe: read-only verify-width / stream distribution.
# TP2_WIDTH_PROBE=1 logs (S, W) per decode round to TP2_WIDTH_PROBE_OUT.
if _os.environ.get("TP2_WIDTH_PROBE", "0") == "1":
    try:
        from tp2widthprobe import install as _install_wprobe

        _install_wprobe()
    except Exception as _e:  # an optional probe must never break the server
        _sys.stderr.write(f"tp2widthprobe: install failed: {_e!r}\n")
        _sys.stderr.flush()

# 0008-B2: two-graph CUDA capture (verify chain + draft launch_blocks).
# TP2_CUDA_GRAPHS=1 (default) installs; =0 is the kill-switch (pure stock).
# TP2_GRAPH_D=1 enables the draft graph; =0 (default) is Graph T only.
if _os.environ.get("TP2_CUDA_GRAPHS", "1") != "0":
    try:
        from tp2cudagraphs import install as _install08b
        _install08b()
    except Exception as _e:  # an optional patch must never break the server
        _sys.stderr.write(f"tp2cudagraphs: install failed: {_e!r}\n")
        _sys.stderr.flush()

# 0008-B2: Graph D — launch_blocks local CUDA graph (rank-0 world=1, 0 NCCL).
# TP2_DRAFT_GRAPH=1 (default), independent of TP2_CUDA_GRAPHS.
if _os.environ.get("TP2_DRAFT_GRAPH", "1") != "0":
    try:
        if _os.environ.get("TP2_V11_GRAPH", "1") == "0":
            from tp2draftgraph import install as _install08d
        else:
            try:
                from tp2draftgraph_v11 import install as _install08d
            except Exception:
                from tp2draftgraph import install as _install08d
        _install08d()
    except Exception as _e:
        _sys.stderr.write(f"tp2draftgraph: install failed: {_e!r}\n")
        _sys.stderr.flush()

# 0009: rowwise FP8 e4m3 all_gather for large prefills (M >= 2048).
# Installs hook with runtime zero-restart toggle (active when TP2_PREFILL_FP8_GATHER=1
# OR /tmp/tp2_fp8_gather_on exists; otherwise takes pure bf16 baseline).
try:
    from tp2fp8gather import install as _install09
    _install09()
except Exception as _e:  # an optional patch must never break the server
    _sys.stderr.write(f"tp2fp8gather: install failed: {_e!r}\n")
    _sys.stderr.flush()

# 0010 (boot92): release kept/viewed-state storage + PAR*GROW admission headroom
# + gate/round telemetry. TP2_KEPTFIX=0 disables.
if _os.environ.get("TP2_KEPTFIX", "1") != "0":
    try:
        from tp2keptfix import install as _install_kept

        _install_kept()
        _sys.stderr.write("tp2keptfix: installed (boot92 kept/headroom/telemetry)\n")
    except Exception as _e:  # an optional patch must never break the server
        _sys.stderr.write(f"tp2keptfix: install failed: {_e!r}\n")
        _sys.stderr.flush()
