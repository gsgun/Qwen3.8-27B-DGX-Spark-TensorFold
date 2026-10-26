"""Patch 0008-B2-v6-stride (TensorFold v0.6.0): Graph D — launch_blocks local CUDA graph.

Root-cause fix (2026-10-24): the v4 _DraftStaged allocated kc_bufs/vc_bufs
with the SHAPE of the sample_snaps (first capture call).  As the drafter's
sliding-window KV cache grew beyond that shape, _fill_kv silently truncated
the copy, and the block_attention Triton kernel derived its per-stream
context length (``lens``) from ``kc.shape[1]`` — i.e. the static buffer
shape, not the live context.  Result: drafter attended a stale, truncated
prefix → ~0 % acceptance at S=1 (the first-captured shape is the smallest).

Fix:
  * kc_bufs / vc_bufs allocated at FULL window size [Hkv, window, D].
  * A per-graph lens_buf (int32 [S]) holds the live per-stream context
    length.  _fill_kv updates it in-place each replay.
  * _dg_block_attention reads lens from lens_buf instead of kc.shape[1].
  * Zero-fill the tail of kc/vc beyond the live context so the Triton
    kernel's out-of-range reads are deterministic (zero keys → zero
    contribution after softmax normalisation over the live lens only).

Gate: TP2_DRAFT_GRAPH=1 (default), independent of TP2_CUDA_GRAPHS (Graph T).
Only rank 0 runs the drafter (TP2_DRAFT_REPL=1 => world=1, zero NCCL).
"""

import os
import sys
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorfold.families.qwen3_5.cuda.glue import embedding, swiglu
from tensorfold.families.qwen3_5.cuda.qmm_fast import matmul, matmul_rows
from tensorfold.families.qwen3_5.cuda.qmm import group_sums
from tensorfold.families.qwen3_5.cuda.draft_attention import _block_attention

_GRAPH_D_ON = os.environ.get("TP2_DRAFT_GRAPH", "1") == "1"
_GRAPH_D_WARMUP = int(os.environ.get("TP2_DRAFT_GRAPH_WARMUP", "8"))
_GRAPH_D_DIAG = os.environ.get("TP2_DRAFT_GRAPH_DIAG", "1") == "1"
_dg_n = [0]


def _log(msg):
    _dg_n[0] += 1
    if _GRAPH_D_DIAG or _dg_n[0] % 500 == 0:
        sys.stderr.write(f"tp2dg {msg}\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# State diagnostic: log snapshot state at each launch_blocks call.
# Tracks context_len / context_end / kc[0] shape+ptr per stream.
# Detects resets (new request on reused slot) vs continuations (decode round).
# ---------------------------------------------------------------------------
_prev_ctx_end = [None]
_state_call_n = [0]


def _log_state(snaps, pendings, tag="pre"):
    _state_call_n[0] += 1
    n = _state_call_n[0]
    if not _GRAPH_D_DIAG:
        return
    if n <= 40 or n % 500 == 0:
        parts = []
        for i, s in enumerate(snaps):
            if s[2] <= 0:
                continue
            kc0 = s[0][0]
            ce = s[3]
            while len(_prev_ctx_end) <= i:
                _prev_ctx_end.append(None)
            prev = _prev_ctx_end[i]
            if prev is None:
                _prev_ctx_end[i] = ce
                reset_tag = "INIT"
            elif ce < prev:
                reset_tag = f"RESET({prev}->{ce})"
            elif ce == prev:
                reset_tag = "STALE"
            else:
                reset_tag = f"+{ce - prev}"
            _prev_ctx_end[i] = ce
            parts.append(
                f"s{i}: n={s[2]} ce={ce} kc0={list(kc0.shape)} "
                f"ptr={kc0.data_ptr() % 1000000:06x} {reset_tag} p={pendings[i]}")
        if parts:
            _log(f"STATE#{n} {tag} " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Patched block_attention: reads lens from a pre-allocated GPU tensor instead
# of deriving it from kc.shape[1].  The lens_buf address is fixed at capture
# time; _fill_kv updates its contents in-place before each replay.
# ---------------------------------------------------------------------------
_dg_lens_buf = [None]          # set to the active graph's lens_buf during refresh
_dg_lens_max = [4]             # max S (also the allocated length of lens_buf)
_dg_buf_len = [0]              # set to the active graph's buf_len during refresh
_dg_kv_bufs = [None]           # per-stream static KV bufs: (kc_bufs, vc_bufs) or None
_dg_orig_lf = [None]  # set in install()



# ---------------------------------------------------------------------------
# Corrected Triton kernel: uses BL (buf_len) for KV head stride instead of s (lens).
# The original _block_attention computes g * s * D which is wrong for static
# buffers of shape [Hkv, buf_len, D].
# ---------------------------------------------------------------------------
@triton.jit
def _block_attention_static(Q, KB, VB, TABLE, LENS, O, scale, window, R,
                            G: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr,
                            LP: tl.constexpr, D: tl.constexpr, BN: tl.constexpr,
                            CAUSAL: tl.constexpr, BL):
    j = tl.program_id(0)
    g = tl.program_id(1)
    M: tl.constexpr = G * LP
    rows = tl.arange(0, M)
    head = g * G + rows // LP
    r = rows % LP
    live = r < L
    d = tl.arange(0, D)
    qrow = (j * L + r).to(tl.int64)
    q = tl.load(Q + head[:, None].to(tl.int64) * R * D + qrow[:, None] * D + d[None, :],
                mask=live[:, None], other=0.0)
    s = tl.load(LENS + j)
    kc = tl.multiple_of(tl.load(TABLE + 2 * j).to(tl.pointer_type(tl.bfloat16)), 16) + g.to(tl.int64) * BL * D
    vc = tl.multiple_of(tl.load(TABLE + 2 * j + 1).to(tl.pointer_type(tl.bfloat16)), 16) + g.to(tl.int64) * BL * D
    m_i = tl.full((M,), float("-inf"), tl.float32)
    l_i = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, D), tl.float32)
    for n0 in range(0, s, BN):
        kidx = n0 + tl.arange(0, BN)
        inside = kidx < s
        k = tl.load(kc + kidx[None, :] * D + d[:, None], mask=inside[None, :], other=0.0)
        qk = tl.dot(q, k) * scale
        seen = inside[None, :] & (s + r[:, None] - kidx[None, :] < window + 1)
        qk = tl.where(seen, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(qk - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vc + kidx[:, None] * D + d[None, :], mask=inside[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    cols = tl.arange(0, LP)
    krow = (j * L + cols).to(tl.int64)
    in_block = cols < L
    kb = tl.load(KB + g.to(tl.int64) * R * D + krow[None, :] * D + d[:, None], mask=in_block[None, :], other=0.0)
    qk = tl.dot(q, kb) * scale
    seen = in_block[None, :]
    if CAUSAL:
        seen = seen & (cols[None, :] <= r[:, None])
    qk = tl.where(seen, qk, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(qk, 1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.exp(qk - m_safe[:, None])
    alpha = tl.exp(m_i - m_safe)
    l_i = l_i * alpha + tl.sum(p, 1)
    vb = tl.load(VB + g.to(tl.int64) * R * D + krow[:, None] * D + d[None, :], mask=in_block[:, None], other=0.0)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)
    out = (acc / l_i[:, None]).to(tl.bfloat16)
    tl.store(O + qrow[:, None] * (G * HKV * D) + head[:, None] * D + d[None, :], out, mask=live[:, None])


def _dg_block_attention(q, k, v, keys, values, length, window, scale,
                        causal=False):
    heads, rows, dim = q.shape
    kv_heads = k.shape[0]
    streams = rows // length
    if len(keys) != streams or len(values) != streams or rows != streams * length \
            or heads % kv_heads:
        raise ValueError("one context a stream, and blocks of equal length")
    # table: data_ptrs (fixed for static buffers, but rebuild to be safe)
    host = torch.tensor(
        [p for kc, vc in zip(keys, values)
         for p in (kc.data_ptr(), vc.data_ptr())],
        dtype=torch.int64).pin_memory()
    dev = host.to(q.device, non_blocking=True)
    table = dev[:2 * streams]
    # lens: from pre-allocated GPU tensor (updated in-place by _fill_kv)
    if _dg_lens_buf[0] is not None:
        lens = _dg_lens_buf[0][:streams]
    else:
        lens = torch.tensor([kc.shape[1] for kc in keys],
                            dtype=torch.int32, device=q.device)
    out = torch.empty((rows, heads * dim), dtype=torch.bfloat16, device=q.device)
    group = heads // kv_heads
    _block_attention_static[(streams, kv_heads)](
        q, k, v, table, lens, out, scale, window, rows,
        G=group, HKV=kv_heads, L=length,
        LP=max(16, triton.next_power_of_2(length)), D=dim,
        BN=64, CAUSAL=causal, BL=keys[0].shape[1],
        num_warps=4, num_stages=2)
    return out


# ---------------------------------------------------------------------------
# _dg_layer_fast: same as DFlash2._layer_fast but calls _dg_block_attention
# ---------------------------------------------------------------------------
def _dg_layer_fast(self, i, x, cos, sin, ctx, length):
    from tensorfold.families.qwen3_5.cuda.dflash2 import _dconv
    w = self.weights
    base = f"layers.{i}."
    normed = F.rms_norm(x, (self.hidden,), w[base + "input_layernorm.weight"],
                        self.eps)
    dyn = self._lin(normed, base + "attention_conv.kernel_projection.weight")
    conv = w[base + "attention_conv.base_kernel"]
    q, k, v = self._prep(
        self._lin(_dconv(normed, dyn, conv, 0, self.group_size, seg=length),
                  base + "self_attn.qkv.weight"),
        i, cos, sin, self.heads_local)
    # Use per-stream static KV buffers (stable data pointers for CUDA graph)
    kv = _dg_kv_bufs[0]
    if kv is not None and len(ctx) == len(kv[0]):
        kc_bufs, vc_bufs = kv
        out = _dg_block_attention(
            q, k, v,
            [kc_bufs[j][i] for j in range(len(ctx))],
            [vc_bufs[j][i] for j in range(len(ctx))],
            length, self.window, self.head_dim ** -0.5, self.is_causal)
    else:
        if _dg_orig_lf[0] is not None:
            return _dg_orig_lf[0](self, i, x, cos, sin, ctx, length)
        out = _dg_block_attention(
            q, k, v,
            [snap[0][i] for snap in ctx],
            [snap[1][i] for snap in ctx],
            length, self.window, self.head_dim ** -0.5, self.is_causal)
    x = _dconv(self._row(out, base + "self_attn.o_proj.weight"),
               dyn, conv, 1, self.group_size, x, seg=length)
    normed = F.rms_norm(x, (self.hidden,),
                        w[base + "post_attention_layernorm.weight"], self.eps)
    dyn = self._lin(normed, base + "mlp_conv.kernel_projection.weight")
    conv = w[base + "mlp_conv.base_kernel"]
    h = _dconv(normed, dyn, conv, 0, self.group_size, seg=length)
    xs = group_sums(h)
    act, act_xs = swiglu(
        matmul(h, self.q4[base + "mlp.gate_proj.weight"], xs),
        matmul(h, self.q4[base + "mlp.up_proj.weight"], xs))
    mlp = self._row(act, base + "mlp.down_proj.weight", act_xs)
    return _dconv(mlp, dyn, conv, 1, self.group_size, x, seg=length)


# ---------------------------------------------------------------------------
# _graph_launch_blocks: identical to v4 except it calls _dg_layer_fast
# ---------------------------------------------------------------------------
def _graph_launch_blocks(self, snaps, pendings, max_nodes, block=None,
                         tokens_buf=None, kc_bufs=None, vc_bufs=None,
                         pos_buf=None):
    out = [None] * len(snaps)
    live = [i for i, snap in enumerate(snaps) if snap[2] > 0]
    if not live or max_nodes < 1:
        return out
    length = min(block or self.block, max_nodes + 1)
    W = len(live) * length

    tokens = tokens_buf[:W]
    x = embedding(tokens, self.target_embed)

    if self.fast:
        ctx = [snaps[i] for i in live]
        rot = []
        for s_idx, snap in enumerate(ctx):
            pos = pos_buf[s_idx]
            phase = torch.arange(0, length, device=self.device,
                                 dtype=torch.float32)[:, None]
            phase = (pos + phase) * self.inv_freq[None, :]
            rot.append((phase.cos().contiguous(), phase.sin().contiguous()))
        cos, sin = torch.cat([c for c, _ in rot]), torch.cat([s for _, s in rot])
        for layer in range(self.layers):
            x = _dg_layer_fast(self, layer, x, cos, sin, ctx, length)
        h = F.rms_norm(
            x.view(len(live), length, -1)[:, 1:].reshape(-1, self.hidden),
            (self.hidden,), self.weights["norm.weight"], self.eps)
    else:
        hs = []
        for j, i in enumerate(live):
            self.restore(snaps[i])
            y = x[j * length:(j + 1) * length]
            for layer in range(self.layers):
                y = self._layer(layer, y)
            yf = y[1:].float()
            hs.append((yf * torch.rsqrt((yf * yf).mean(dim=-1, keepdim=True)
                                        + self.eps)
                       * self.weights["norm.weight"].float()).to(torch.bfloat16))
        h = torch.cat(hs)

    projected = self._lin(h, "candidate_selector.hidden_projection.weight").float()
    if self.sub_parts is not None:
        h = h.contiguous()
        logits = torch.cat([part(h)[:, lo:hi]
                            for part, lo, hi in self.sub_parts], dim=1)
    elif self.head_cols is not None:
        logits = self.sub_head(h.contiguous()).index_select(1, self.head_cols)
    elif self.sub_rows is not None:
        logits = matmul_rows(h, self.sub_rows)
    else:
        logits = matmul(h, self.sub_head)
    values, local_ids = torch.topk(logits.float(), k=16, dim=-1, sorted=False)
    global_ids = self.head_ids[local_ids]
    if self.world == 2:
        import torch.distributed as dist
        both_values = torch.empty((2, *values.shape), dtype=values.dtype,
                                  device=values.device)
        both_ids = torch.empty((2, *global_ids.shape), dtype=global_ids.dtype,
                               device=global_ids.device)
        dist.all_gather_into_tensor(both_values, values.contiguous())
        dist.all_gather_into_tensor(both_ids, global_ids.contiguous())
        merged = torch.cat((both_values[0], both_values[1]), dim=1)
        values, pick = torch.topk(merged, k=16, dim=-1, sorted=False)
        global_ids = torch.cat((both_ids[0], both_ids[1]), dim=1).gather(1, pick)
    shared = [global_ids, torch.cat((values, projected), dim=1), None]
    for j, i in enumerate(live):
        out[i] = (shared, j * (length - 1), length - 1, int(pendings[i]))
    return out


# ---------------------------------------------------------------------------
# _DraftStaged: static buffers + graph for one S value
# ---------------------------------------------------------------------------
class _DraftStaged:
    """Static buffers + graph for one S value.

    kc_bufs / vc_bufs are allocated at FULL window size so that the Triton
    kernel always sees the complete sliding-window context.  lens_buf holds
    the live per-stream context length and is updated in-place each replay.
    """

    def __init__(self, draft, n_live, max_nodes, block,
                 sample_snaps, sample_pendings):
        self.draft = draft
        self.n_live = n_live
        self.max_nodes = max_nodes
        self.block = block
        self.length = min(block or draft.block, max_nodes + 1)
        self.W = n_live * self.length
        self.device = draft.device
        self.replay_count = 0
        self.miss_count = 0
        self.window = draft.window

        # Determine the maximum context length seen in sample_snaps.
        # The live kc buffers can be longer than draft.window (full context,
        # not just the sliding window).  Use the max observed length so the
        # static buffers always fit the live data.
        max_ctx = 0
        for snap in sample_snaps:
            for li in range(len(snap[0])):
                max_ctx = max(max_ctx, snap[0][li].shape[1])
            for li in range(len(snap[1])):
                max_ctx = max(max_ctx, snap[1][li].shape[1])
        self.buf_len = max(max_ctx, self.window)
        _log(f"__init__ S={n_live} window={self.window} max_ctx={max_ctx} "
             f"buf_len={self.buf_len} kc0_shape={sample_snaps[0][0][0].shape} "
             f"n_snap_kv_layers={len(sample_snaps[0][0])}")

        max_W = 4 * self.length
        self.tokens_buf = torch.zeros(max_W, dtype=torch.int32,
                                      device=self.device)
        self.cpu_tokens = torch.zeros(max_W, dtype=torch.int32)

        # lens_buf: allocated at max S=4 so the address is stable across
        # all graphs; only the first n_live entries are used.
        self.lens_buf = torch.zeros(_dg_lens_max[0], dtype=torch.int32,
                                    device=self.device)

        # KV cache: allocated at buf_len (max observed context length)
        self.kc_bufs = []
        self.vc_bufs = []
        self.pos_buf = torch.zeros(n_live, dtype=torch.float32,
                                   device=self.device)
        for s_idx, snap in enumerate(sample_snaps):
            kc_live, vc_live, n, pos = snap
            kc_s, vc_s = [], []
            for li in range(len(kc_live)):
                src = kc_live[li]
                full = (src.shape[0], self.buf_len, src.shape[2])
                b = torch.zeros(full, dtype=src.dtype, device=self.device)
                b[:, :src.shape[1], :].copy_(src)
                kc_s.append(b)
            for li in range(len(vc_live)):
                src = vc_live[li]
                full = (src.shape[0], self.buf_len, src.shape[2])
                b = torch.zeros(full, dtype=src.dtype, device=self.device)
                b[:, :src.shape[1], :].copy_(src)
                vc_s.append(b)
            self.kc_bufs.append(kc_s)
            self.vc_bufs.append(vc_s)
            self.pos_buf[s_idx] = float(pos)
            self.lens_buf[s_idx] = int(n)

        self.static_snaps = [
            (self.kc_bufs[s], self.vc_bufs[s], sample_snaps[s][2], s)
            for s in range(n_live)
        ]
        self.sample_pendings = [int(p) for p in sample_pendings]
        self.graph = None
        self.out_ref = None

    # -- fill helpers (called outside the CUDA graph, before replay) --------

    def _fill_tokens(self, snaps, pendings):
        d = self.draft
        live = [i for i, s in enumerate(snaps) if s[2] > 0][:self.n_live]
        ci = 0
        for i in live:
            self.cpu_tokens[ci] = pendings[i]
            ci += 1
            for _ in range(self.length - 1):
                self.cpu_tokens[ci] = d.mask_id
                ci += 1
        self.tokens_buf[:self.W].copy_(self.cpu_tokens[:self.W],
                                       non_blocking=True)

    def _fill_kv(self, snaps, pendings):
        live = [i for i, s in enumerate(snaps) if s[2] > 0][:self.n_live]
        for s_idx, i in enumerate(live):
            kc_l, vc_l, n, pos = snaps[i]
            for li, buf in enumerate(self.kc_bufs[s_idx]):
                if li < len(kc_l):
                    src = kc_l[li]
                    n_copy = min(src.shape[1], self.buf_len)
                    if n_copy > 0:
                        buf[:src.shape[0], :n_copy, :src.shape[2]] = \
                            src[:src.shape[0], :n_copy, :src.shape[2]]
                    if n_copy < self.buf_len:
                        buf[:src.shape[0], n_copy:, :src.shape[2]].zero_()
            for li, buf in enumerate(self.vc_bufs[s_idx]):
                if li < len(vc_l):
                    src = vc_l[li]
                    n_copy = min(src.shape[1], self.buf_len)
                    if n_copy > 0:
                        buf[:src.shape[0], :n_copy, :src.shape[2]] = \
                            src[:src.shape[0], :n_copy, :src.shape[2]]
                    if n_copy < self.buf_len:
                        buf[:src.shape[0], n_copy:, :src.shape[2]].zero_()
            self.pos_buf[s_idx] = float(pos)
            # lens_buf is updated in-place (address stable for CUDA graph)
            self.lens_buf[s_idx] = int(n)

    def refresh(self, snaps, pendings):
        self._fill_tokens(snaps, pendings)
        self._fill_kv(snaps, pendings)
        # Point the global at this graph's lens_buf so _dg_block_attention
        # reads the correct (in-place-updated) tensor during replay.
        _dg_lens_buf[0] = self.lens_buf
        _dg_buf_len[0] = self.buf_len
        _dg_kv_bufs[0] = (self.kc_bufs, self.vc_bufs)

    def can_handle(self, snaps, pendings):
        live = [i for i, s in enumerate(snaps) if s[2] > 0]
        return len(live) == self.n_live

    def try_replay(self, snaps, pendings, max_nodes, block):
        if not self.can_handle(snaps, pendings):
            self.miss_count += 1
            return None
        if max_nodes != self.max_nodes:
            self.miss_count += 1
            return None
        self.refresh(snaps, pendings)
        if self.graph is not None:
            self.graph.replay()
        else:
            with torch.no_grad():
                self.out_ref = _graph_launch_blocks(self.draft,
                                     self.static_snaps,
                                     self.sample_pendings, self.max_nodes,
                                     self.block,
                                     tokens_buf=self.tokens_buf,
                                     kc_bufs=self.kc_bufs, vc_bufs=self.vc_bufs,
                                     pos_buf=self.pos_buf)
        self.replay_count += 1
        if self.replay_count % 200 == 1:
            _log(f"tp2dg replay S={self.n_live} #{self.replay_count} "
                 f"miss={self.miss_count}")
        out = [None] * len(snaps)
        live = [i for i, s in enumerate(snaps) if s[2] > 0][:self.n_live]
        length = self.length
        for j, i in enumerate(live):
            out[i] = (self.out_ref[i][0], j * (length - 1),
                      length - 1, int(pendings[i]))
        return out


# ---------------------------------------------------------------------------
# _capture: same as v4, uses _graph_launch_blocks (which calls _dg_layer_fast)
# ---------------------------------------------------------------------------
def _capture(draft, S, max_nodes, block, sample_snaps, sample_pendings,
             pool):
    staged = _DraftStaged(draft, S, max_nodes, block,
                          sample_snaps, sample_pendings)
    staged.refresh(snaps=sample_snaps, pendings=sample_pendings)
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(2):
            _graph_launch_blocks(draft, staged.static_snaps,
                                 staged.sample_pendings, max_nodes, block,
                                 tokens_buf=staged.tokens_buf,
                                 kc_bufs=staged.kc_bufs, vc_bufs=staged.vc_bufs,
                                 pos_buf=staged.pos_buf)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            out = _graph_launch_blocks(draft, staged.static_snaps,
                                       staged.sample_pendings, max_nodes, block,
                                       tokens_buf=staged.tokens_buf,
                                       kc_bufs=staged.kc_bufs,
                                       vc_bufs=staged.vc_bufs,
                                       pos_buf=staged.pos_buf)
        staged.graph = g
        staged.out_ref = out
    _log(f"captured S={S} W={staged.W} buf_len={staged.buf_len} window={staged.window}")
    return staged


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
def install():
    if not _GRAPH_D_ON:
        sys.stderr.write("tp2dg: off (TP2_DRAFT_GRAPH=0)\n")
        sys.stderr.flush()
        return

    import tensorfold.families.qwen3_5.cuda.dflash2 as d2
    orig_lb = d2.DFlash2.launch_blocks
    state = {"graphs": {}, "pool": None, "warm": [0], "cache": {}}

    def _patched_lb(self, snaps, pendings, max_nodes, block=None):
        _log_state(snaps, pendings, "pre")
        if state["warm"][0] < 3:
            _log(f"patched_lb called warm={state['warm'][0]} world={getattr(self,'world',1)} "
                 f"S={sum(1 for s in snaps if s[2]>0)}")
        if getattr(self, "world", 1) != 1:
            return orig_lb(self, snaps, pendings, max_nodes, block)
        live = [i for i, s in enumerate(snaps) if s[2] > 0]
        S = len(live)
        if S < 1 or S > 4:
            return orig_lb(self, snaps, pendings, max_nodes, block)
        # Cache this S value's snapshots for staged buffer creation
        if S not in state["cache"]:
            state["cache"][S] = ([snaps[i] for i in live],
                                 [pendings[i] for i in live],
                                 max_nodes, block)
            _log(f"v10e: cached S={S} warm={state['warm'][0]}")
        state["warm"][0] += 1
        if state["warm"][0] <= _GRAPH_D_WARMUP:
            return orig_lb(self, snaps, pendings, max_nodes, block)
        if S in state["graphs"]:
            out = state["graphs"][S].try_replay(snaps, pendings,
                                                max_nodes, block)
            if out is not None:
                return out
        return orig_lb(self, snaps, pendings, max_nodes, block)

    def _patched_lb_with_capture(self, snaps, pendings, max_nodes, block=None):
        out = _patched_lb(self, snaps, pendings, max_nodes, block)
        # Stage all seen S values after warmup (eager, no graph)
        if state["warm"][0] == _GRAPH_D_WARMUP + 1:
            for S_val in sorted(state["cache"].keys()):
                s_snaps, s_pend, mn, bl = state["cache"][S_val]
                if S_val not in state["graphs"]:
                    try:
                        staged = _DraftStaged(self, S_val, mn, bl,
                                              s_snaps, s_pend)
                        staged.refresh(snaps=s_snaps, pendings=s_pend)
                        staged.graph = None  # no graph, eager mode
                        state["graphs"][S_val] = staged
                        _log(f"v10e: staged S={S_val} (eager, no graph)")
                    except Exception as e:
                        import traceback
                        _log(f"stage S={S_val} failed: {e}")
                        for line in traceback.format_exc().splitlines():
                            _log(f"  tb: {line}")
        # If a new S appears post-warmup and isn't staged yet, stage it now
        if state["warm"][0] > _GRAPH_D_WARMUP:
            S_now = sum(1 for s in snaps if s[2] > 0)
            if S_now in state["cache"] and S_now not in state["graphs"]:
                s_snaps, s_pend, mn, bl = state["cache"][S_now]
                try:
                    staged = _DraftStaged(self, S_now, mn, bl,
                                          s_snaps, s_pend)
                    staged.refresh(snaps=s_snaps, pendings=s_pend)
                    staged.graph = None
                    state["graphs"][S_now] = staged
                    _log(f"v10e: staged S={S_now} (eager, late)")
                except Exception as e:
                    _log(f"stage S={S_now} failed: {e}")
        return out

    d2.DFlash2.launch_blocks = _patched_lb_with_capture
    # Replace _layer_fast with _dg_layer_fast so launch_blocks uses static KV buffers
    _dg_orig_lf[0] = d2.DFlash2._layer_fast
    d2.DFlash2._layer_fast = _dg_layer_fast
    sys.stderr.write(
        f"tp2dg: installed v10e-eager-allS (warmup={_GRAPH_D_WARMUP}, shapes S=1..4, "
        f"pool=shared, full-window KV + explicit lens, _layer_fast->_dg_layer_fast)\n")
    sys.stderr.flush()
