"""Patch 0006 (TensorFold v0.6.0): fused W8A16 affine decode kernel + prefill accelerator.

Modes (env TP2_W8A16_MODE):
  fast (default)  - native CUDA w8a16_cuda_qmm for decode (M < 512)
  exact           - Triton stock kernel (bit-identical to stock)

Prefill accelerator (env TP2_PREFILL_MODE, M >= TP2_PREFILL_THRESH=512):
  a    - dequant-once to BF16 scratch + torch.matmul (cuBLASLt)
  b    - dequant to FP8 + dynamic x-quant + torch._scaled_mm (f8f8bf16)
  none - no prefill acceleration (default; uses native CUDA for all M)

Prefill threshold (env TP2_PREFILL_THRESH): default 512.
Hybrid routing (b8440d18 + v10e): M >= thresh uses the prefill accelerator
above (mode a: ~1500 tok/s, 3.8x Triton); M < thresh decode routes to
v10e Triton w8a16_mm_fast (c1d 94.1 / c4d 208.0 tok/s vs 87.15 / 186.0
native CUDA). TP2_W8A16_DECODE=1 re-enforces native CUDA for decode
(rollback; default 0).
"""

import os

import torch
import triton
import triton.language as tl


# ── Triton decode kernels (existing, unchanged) ──────────────────────────

@triton.jit
def w8a16_mm_fast(X, W8, S, B, OUT, M, N: tl.constexpr, K: tl.constexpr, GS: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    within = tl.arange(0, GS)
    acc = tl.zeros((16, 32), tl.float32)
    for group in range(K // GS):
        k = group * GS + within
        x = tl.load(X + rows[:, None] * K + k[None, :], mask=rows[:, None] < M, other=0)
        quant = tl.load(W8 + cols[:, None] * K + k[None, :],
                        mask=cols[:, None] < N, other=0).to(tl.bfloat16)
        dot = tl.dot(x, tl.trans(quant), input_precision="ieee")
        scale = tl.load(S + cols * (K // GS) + group, mask=cols < N, other=0).to(tl.float32)
        bias = tl.load(B + cols * (K // GS) + group, mask=cols < N, other=0).to(tl.float32)
        sums = tl.sum(x.to(tl.float32), 1)
        acc = acc + dot * scale[None, :] + sums[:, None] * bias[None, :]
    tl.store(OUT + rows[:, None] * N + cols[None, :], acc,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def w8a16_codes(W, n, k, valid, KW: tl.constexpr, BITS: tl.constexpr):
    bit = k * BITS
    word, shift = bit // 32, bit % 32
    low = tl.load(W + n * KW + word, mask=valid, other=0).to(tl.uint32)
    high = tl.load(W + n * KW + word + 1, mask=valid & (shift + BITS > 32) & (word + 1 < KW), other=0).to(tl.uint32)
    value = (low >> shift) | tl.where(shift + BITS > 32, high << ((32 - shift) % 32), 0)
    return value & ((1 << BITS) - 1)


@triton.jit
def w8a16_mm_exact(X, W, S, B, OUT, M, N: tl.constexpr, K: tl.constexpr,
                   BITS: tl.constexpr, GS: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    within = tl.arange(0, GS)
    acc = tl.zeros((16, 32), tl.float32)
    for group in range(K // GS):
        k = group * GS + within
        x = tl.load(X + rows[:, None] * K + k[None, :], mask=rows[:, None] < M, other=0)
        quant = w8a16_codes(W, cols[:, None], k[None, :], cols[:, None] < N, K * BITS // 32, BITS)
        dot = tl.dot(x, tl.trans(quant.to(tl.bfloat16)), input_precision="ieee")
        scale = tl.load(S + cols * (K // GS) + group, mask=cols < N, other=0).to(tl.float32)
        bias = tl.load(B + cols * (K // GS) + group, mask=cols < N, other=0).to(tl.float32)
        sums = tl.sum(x.to(tl.float32), 1)
        acc = acc + dot * scale[None, :] + sums[:, None] * bias[None, :]
    tl.store(OUT + rows[:, None] * N + cols[None, :], acc,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


# ── Prefill accelerator ──────────────────────────────────────────────────

_prefill_mode = os.environ.get("TP2_PREFILL_MODE", "none").lower()
if _prefill_mode == "none":
    _pm_file = "/patches/prefill_mode.txt"
    if os.path.exists(_pm_file):
        with open(_pm_file) as _f:
            _prefill_mode = _f.read().strip().lower()
_prefill_thresh = int(os.environ.get("TP2_PREFILL_THRESH", "512"))

# Lazily-resized scratch buffers (shared across layers, sequential processing)
_scratch_bf16 = None
_scratch_fp8_t = None   # [K, N] transposed for _scaled_mm
_fp8_row_scale = None   # [N]
_fp8_tok_scale = None   # [M]


def _get_scratch_bf16(n, k, device):
    global _scratch_bf16
    needed = n * k
    if _scratch_bf16 is None or _scratch_bf16.numel() < needed:
        _scratch_bf16 = torch.empty(needed, dtype=torch.bfloat16, device=device)
    return _scratch_bf16[:needed].view(n, k)


def _get_scratch_fp8(n, k, device):
    global _scratch_fp8_t, _fp8_row_scale, _fp8_tok_scale
    needed = n * k
    if _scratch_fp8_t is None or _scratch_fp8_t.numel() < needed:
        _scratch_fp8_t = torch.empty(needed, dtype=torch.float8_e4m3fn, device=device)
    if _fp8_row_scale is None or _fp8_row_scale.numel() < n:
        _fp8_row_scale = torch.empty(max(n, 32768), dtype=torch.float32, device=device)
    if _fp8_tok_scale is None or _fp8_tok_scale.numel() < 65536:
        _fp8_tok_scale = torch.empty(65536, dtype=torch.float32, device=device)
    return _scratch_fp8_t[:needed].view(n, k)  # [N, K] contiguous


def _dequant_bf16(w_u8, scales, biases, n, k, out=None):
    """Dequantize W8A16 → BF16. Uses CUDA kernel if available, else PyTorch fallback."""
    if _w8a16_cuda is not None and hasattr(_w8a16_cuda, "dequant_to_bf16"):
        return _w8a16_cuda.dequant_to_bf16(w_u8[:n, :k], scales[:n, :k//64], biases[:n, :k//64])
    if out is None:
        out = torch.empty(n, k, dtype=torch.bfloat16, device=w_u8.device)
    gs = 64
    w = w_u8[:n, :k].view(n, k // gs, gs).to(torch.float32)
    s = scales[:n, :k // gs].unsqueeze(-1).to(torch.float32)
    b = biases[:n, :k // gs].unsqueeze(-1).to(torch.float32)
    out[:n, :k] = (w * s + b).view(n, k).to(torch.bfloat16)
    return out


def _dequant_fp8(w_u8, scales, biases, n, k, out, row_scale):
    """Dequantize W8A16 → FP8, stored [N, K]. row_scale: [N]."""
    gs = 64
    w = w_u8[:n, :k].view(n, k // gs, gs).to(torch.float32)
    s = scales[:n, :k // gs].unsqueeze(-1).to(torch.float32)
    b = biases[:n, :k // gs].unsqueeze(-1).to(torch.float32)
    w_deq = (w * s + b).view(n, k)  # [N, K] f32
    row_max = w_deq.abs().amax(dim=1)  # [N]
    rs = (row_max / 448.0).clamp(min=1e-12)
    row_scale[:n].copy_(rs)
    out[:n, :k].copy_((w_deq / rs.unsqueeze(1)).to(torch.float8_e4m3fn))


def _quantize_x_fp8(x, out, tok_scale, m, k):
    """Dynamic per-token FP8 quantization of x. tok_scale: [M]."""
    xf = x[:m, :k].float()
    tok_max = xf.abs().amax(dim=1)  # [M]
    ts = (tok_max / 448.0).clamp(min=1e-12)
    tok_scale[:m].copy_(ts)
    out[:m, :k].copy_((xf / ts.unsqueeze(1)).to(torch.float8_e4m3fn))


def _prefill_mode_a(x, q, n, k):
    """Mode A: dequant-once BF16 + cuBLASLt matmul."""
    m = x.shape[0]
    w_dequant = _dequant_bf16(q.weight.view(torch.uint8), q.scales, q.biases, n, k)
    return torch.matmul(x, w_dequant.t())  # [M, N] bf16, cuBLASLt


def _prefill_mode_b(x, q, n, k):
    """Mode B: dequant FP8 + dynamic x-quant + torch._scaled_mm."""
    m = x.shape[0]
    dev = x.device
    scratch = _get_scratch_fp8(n, k, dev)  # [N, K] contiguous
    _dequant_fp8(q.weight.view(torch.uint8), q.scales, q.biases, n, k,
                 scratch, _fp8_row_scale)
    x_fp8 = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=dev)
    _quantize_x_fp8(x, x_fp8, _fp8_tok_scale, m, k)
    # _scaled_mm expects b=[K,N] column-major → pass scratch[:n,:k].T
    return torch._scaled_mm(x_fp8, scratch[:n, :k].T,
                            _fp8_tok_scale[:m].unsqueeze(1),
                            _fp8_row_scale[:n].unsqueeze(0),
                            out_dtype=torch.bfloat16)


# ── Main dispatch ────────────────────────────────────────────────────────

def matmul(x, q, *, f32=False):
    """Fused W8A16 affine matmul; dispatches on M and TP2_PREFILL_MODE."""
    m0 = x.shape[0]
    if m0 == 0:
        n = q.n if q.layout == "tiled" else int(q.weight.shape[0] if q.layout == "dense" else q.scales.shape[0])
        return torch.empty((0, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    n, k = _validate(x, q)
    x = x.contiguous()
    if _w8a16_cuda is not None and q.bits == 8 and q.gs == 64:
        m = x.shape[0]
        if m >= _prefill_thresh and _prefill_mode == "a" and not f32:
            return _prefill_mode_a(x, q, n, k)
        if m >= _prefill_thresh and _prefill_mode == "b" and not f32:
            return _prefill_mode_b(x, q, n, k)
        if m < _prefill_thresh and not f32 and not _w8a16_decode:
            pass  # decode (M < 512) -> v10e Triton w8a16_mm_fast below
        else:
            out = _w8a16_cuda.forward(x, q.weight.view(torch.uint8), q.scales, q.biases, f32)
            return out
    out = torch.empty((x.shape[0], n), dtype=torch.float32 if f32 else torch.bfloat16,
                      device=x.device)
    grid = (triton.cdiv(x.shape[0], 16), triton.cdiv(n, 32))
    if _mode == "exact":
        w8a16_mm_exact[grid](x, q.weight, q.scales, q.biases, out, x.shape[0],
                             N=n, K=k, BITS=q.bits, GS=q.gs,
                             num_warps=4, enable_fp_fusion=False)
    else:
        w8a16_mm_fast[grid](x, q.weight.view(torch.uint8), q.scales, q.biases, out, x.shape[0],
                            N=n, K=k, GS=q.gs,
                            num_warps=4, num_stages=3, enable_fp_fusion=False)
    return out


_validate = None  # bound by install()
_mode = os.environ.get("TP2_W8A16_MODE", "fast").lower()
# Hybrid routing: M < TP2_PREFILL_THRESH decode goes to v10e Triton w8a16_mm_fast
# (beats native CUDA on every decode shape; cp.async pipeline serializes the K-loop).
# TP2_W8A16_DECODE=1 forces native CUDA decode (rollback flag; default 0).
_w8a16_decode = os.environ.get("TP2_W8A16_DECODE", "0") == "1"

# Native CUDA dispatch (Phase 2: bits==8, gs==64)
_w8a16_cuda = None
try:
    import w8a16_cuda_qmm as _w8a16_cuda
    _w8a16_cuda = _w8a16_cuda
except ImportError:
    pass


def install(affine, diag=False):
    """Wrap affine.matmul, routing mlx 8-bit linears through the w8a16 kernels."""
    global _mode, _validate
    _mode = os.environ.get("TP2_W8A16_MODE", "fast").lower()
    if _mode not in ("fast", "exact"):
        _mode = "fast"
    if getattr(affine, "_tp2w8a16", False):
        return
    orig = affine.matmul
    affine._orig_matmul = orig
    _validate = affine._validate
    enabled = os.environ.get("TP2_W8A16", "1") != "0"

    stat = {"calls": 0, "ms": 0.0, "fallback": 0, "shapes": {}, "shape_ms": {}}

    def _flush():
        if not diag:
            return
        try:
            import json as _j
            with open("/tmp/w8a16_stat.json", "w") as _f:
                _j.dump({"calls": stat["calls"], "mode": _mode,
                         "prefill_mode": _prefill_mode,
                         "kernel_ms": round(stat["ms"], 1),
                         "avg_kernel_ms": round(stat["ms"] / max(stat["calls"], 1), 4),
                         "fallback": stat["fallback"],
                         "shapes": {str(kk): v for kk, v in sorted(stat["shapes"].items())},
                         "shape_ms": {str(kk): round(v, 3) for kk, v in sorted(stat["shape_ms"].items())}},
                        _f)
        except Exception:
            pass

    ev0 = torch.cuda.Event(enable_timing=True) if diag else None
    ev1 = torch.cuda.Event(enable_timing=True) if diag else None

    def matmul_wrap(x, q, *, f32=False):
        if not enabled or q.layout != "mlx" or q.bits != 8:
            stat["fallback"] += 1
            return orig(x, q, f32=f32)
        if diag:
            ev0.record()
        out = matmul(x, q, f32=f32)
        if diag:
            ev1.record()
            ev1.synchronize()
            dt = ev0.elapsed_time(ev1)
            n, k = _validate(x, q)
            m = x.shape[0]
            stat["ms"] += dt
            stat["calls"] += 1
            stat["shapes"][(m, n, k)] = stat["shapes"].get((m, n, k), 0) + 1
            stat["shape_ms"][(m, n, k)] = stat["shape_ms"].get((m, n, k), 0.0) + dt
            if stat["calls"] % 50 == 0:
                _flush()
        else:
            stat["calls"] += 1
        return out

    affine.matmul = matmul_wrap
    affine._tp2w8a16 = True
    import sys
    sys.stderr.write("tp2w8a16: installed (mode=%s, prefill=%s, thresh=%d, decode_cuda=%d, diag=%d)\n"
                     % (_mode, _prefill_mode, _prefill_thresh, 1 if _w8a16_decode else 0, 1 if diag else 0))
    sys.stderr.flush()
