# tp2fp8gather.py — 0009: rowwise FP8 e4m3 all_gather for large prefills.
#
# Runtime toggle: active if TP2_PREFILL_FP8_GATHER=1 OR /tmp/tp2_fp8_gather_on exists,
# AND local.shape[0] >= TP2_PREFILL_FP8_THRESH (default 2048).
#
# Zero-restart toggle:
#   docker exec qwen38-tf-rank0 touch /tmp/tp2_fp8_gather_on
#   docker exec qwen38-tf-rank1 touch /tmp/tp2_fp8_gather_on
# Revert to pure bf16:
#   docker exec qwen38-tf-rank0 rm -f /tmp/tp2_fp8_gather_on
#   docker exec qwen38-tf-rank1 rm -f /tmp/tp2_fp8_gather_on
#
# Cuts RoCEv2 all_gather wire payload ~50% (bf16 2 B/elem -> fp8 1 B/elem + [M,1] bf16 scale).
# Cosine similarity > 0.9995 vs pure bf16 baseline.

import os as _os
import sys as _sys
import torch as _torch

_THRESH = int(_os.environ.get("TP2_PREFILL_FP8_THRESH", "2048"))
_FP8 = _torch.float8_e4m3fn
_MAX_FP8 = 448.0  # max magnitude of e4m3fn

# Static staging buffers keyed by (m, n, device). Grown on demand, never freed.
_static = {}


def _stage(m, n, device):
    key = (m, n, device)
    buf = _static.get(key)
    if buf is None:
        qout = _torch.empty((2 * m, n), dtype=_torch.uint8, device=device)
        sout = _torch.empty((2 * m, 1), dtype=_torch.bfloat16, device=device)
        buf = (qout, sout)
        _static[key] = buf
    return buf


def _is_active():
    return _os.environ.get("TP2_PREFILL_FP8_GATHER", "0") == "1" or _os.path.exists("/tmp/tp2_fp8_gather_on")


def install():
    import torch.distributed as dist
    from tensorfold.families.qwen3_5.cuda import distributed as _dist

    orig = _dist.gather_rank_partials

    def fp8_gather(local, group=None, dtype=_torch.bfloat16):
        # Fall back to original (counted) path if toggle is OFF or outside FP8 regime.
        if (not _is_active()
                or local.dtype not in (_torch.float32, _torch.bfloat16)
                or local.ndim != 2
                or local.shape[0] < _THRESH
                or not dist.is_initialized()
                or dist.get_world_size(group) != 2):
            return orig(local, group=group, dtype=dtype)

        m, n = local.shape
        # 1. Per-row scale [M, 1] in bfloat16
        s = (local.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / _MAX_FP8).to(_torch.bfloat16)
        # 2. Quantize local partial to fp8 e4m3fn, view as uint8
        q = (local / s).to(_FP8).view(_torch.uint8)

        # 3. Static staging buffers
        qout, sout = _stage(m, n, local.device)

        # 4. AllGather into static buffers across TP group
        dist.all_gather_into_tensor(qout, q.contiguous(), group=group)
        dist.all_gather_into_tensor(sout, s.contiguous(), group=group)

        # 5. Dequantize rank slices back to bfloat16 and accumulate
        q_slices = qout.view(_FP8).to(_torch.bfloat16).view(2, m, n)
        s_slices = sout.view(2, m, 1)
        res = (q_slices[0] * s_slices[0] + q_slices[1] * s_slices[1]).to(dtype)
        return res

    _dist.gather_rank_partials = fp8_gather
    _sys.stderr.write(f"tp2fp8gather: installed (zero-restart toggle ready, thresh={_THRESH})\n")
    _sys.stderr.flush()
