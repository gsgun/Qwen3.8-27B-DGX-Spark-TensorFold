"""Patch 0010 v11 (TensorFold v0.6.0): v11 Optimization Suite — boot80.

Built on v10e (3ccfeb2b, eager-staged drafter) + v3 (96fccac9, graph
capture recipes).  Two capture tiers:

Tier 1A — rank-local DFlash2 drafter CUDA graphs (0.0 collectives).
    v10e staged every launch_blocks call in static buffers but ran the
    drafter forward EAGER (graph=None at both staging sites).  v11
    enables v10e's previously-unused _capture(): one torch.cuda.CUDAGraph
    per (S, max_nodes, block) with a shared graph pool; try_replay
    already knows how to replay a captured graph.  Any capture exception
    falls back to the exact v10e eager-staged path for that S.
    Gate: TP2_V11_GRAPH=1 (default 1); TP2_V11_GRAPH=0 -> exact v10e.

Tier 1B — target-verify bucketed CUDA graphs (M < 512).
    Live verify windows are W=15 ([pending] + 14 draft nodes), 97.8%
    non-chain.  The GDN tree kernel dispatches by host-side slot count
    into compile-time templates (gdn.cu dispatch_tree):
        slots == 0    -> TREE(0, 8, 4, C=true)  (chain, plan not read)
        slots <= 2    -> TREE(2, 4, 4)
        slots <= 4    -> TREE(4, 2, 4)          (all live trees land here)
        slots <= 8    -> TREE(8, 2, 4)
        slots <= 16   -> TREE(16, 2, 2)
        else          -> TREE(32, 2, 1)
    The kernel grid depends only on the template, never on plan
    contents, so one captured graph per (state, slot-bucket, W) is
    replayable for every round of that bucket: entries/windows/pos/ids/
    attention tables are refreshed in place (one pinned H2D copy + one
    _paths kernel relaunch) before replay.  Attention plans use the
    engine's padded_host layout (items for every full 512-key chunk up
    to TF_SLOT_KV_CAP); the _shared kernel skips chunks beyond the live
    committed length, so the padded plan is exact for any p <= KV_CAP
    (this is what v3's non-padded chain plan could not do).
    tree_forward / multi_tree_forward are state-pure (commit is a
    separate call), so warmup + capture runs never double-advance state.
    Gate: TP2_V11_GRAPH=1 AND TP2_V11_TARGET_GRAPH=1 (both default 1).

Safety:
  * Every capture/replay is wrapped; any exception falls back to eager
    (logged once per feature), the server never dies on a graph hiccup.
  * Verify graphs are keyed (state-ids, slot-bucket, W, full_logits,
    capture_taps); an LRU cap (TP2_V11_MAX_GRAPHS, default 12) evicts
    the oldest graph and returns its pool memory.
  * Both ranks trigger the same capture in the same round (windows are
    shared before verify), so captured NCCL collectives pair up.

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

# --------------------------------------------------------------------------- #
# v11 config
# --------------------------------------------------------------------------- #
_V11_GRAPH_ON = os.environ.get("TP2_V11_GRAPH", "1") == "1"
_V11_TARGET_ON = _V11_GRAPH_ON and os.environ.get("TP2_V11_TARGET_GRAPH", "1") == "1"
_V11_MAX_GRAPHS = int(os.environ.get("TP2_V11_MAX_GRAPHS", "16"))
_V11_VERIFY_WARMUP = int(os.environ.get("TP2_V11_VERIFY_WARMUP", "50"))
_KV_CAP = int(os.environ.get("TF_SLOT_KV_CAP", "16384"))

_GRAPH_D_ON = os.environ.get("TP2_DRAFT_GRAPH", "1") == "1"
_GRAPH_D_WARMUP = int(os.environ.get("TP2_DRAFT_GRAPH_WARMUP", "8"))
_GRAPH_D_DIAG = os.environ.get("TP2_DRAFT_GRAPH_DIAG", "1") == "1"
_dg_n = [0]


def _log(msg):
    _dg_n[0] += 1
    if _GRAPH_D_DIAG or _dg_n[0] % 500 == 0:
        sys.stderr.write(f"tp2v11 {msg}\n")
        sys.stderr.flush()


_REPLAY_N = [0, 0, 0]  # [tier1a_drafter, tier1b_tf, tier1b_mtf]
def _replay_count(i):
    _REPLAY_N[i] += 1
    if _REPLAY_N[i] % 256 == 0:
        sys.stderr.write(
            f"tp2v11 REPLAY {[ '1A', '1Btf', '1Bmtf'][i]} n={_REPLAY_N[i]}\n")
        sys.stderr.flush()


# --------------------------------------------------------------------------- #
# State diagnostic (v10e verbatim): log snapshot state at each launch_blocks
# call — resets vs continuations per stream.
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Patched block_attention globals (v10e verbatim)
# --------------------------------------------------------------------------- #
_dg_lens_buf = [None]
_dg_lens_max = [8]
_dg_buf_len = [0]
_dg_kv_bufs = [None]
_dg_orig_lf = [None]
# Static KV data-pointer table (v11 fix): the DFlash2 Triton attention reads
# K/V through a per-layer, per-stream pointer table. v10e built that table
# each call via .data_ptr() + an H2D copy of a TEMPORARY pinned host buffer.
# Inside a captured CUDA graph that H2D copy is baked in; on replay the graph
# re-reads the pinned memory AFTER it has been freed/reused -> the table holds
# stale/garbage addresses -> non-deterministic KV reads (deterministic for the
# first ~dozen replays while the pinned slot is untouched, then diverges).
# Fix: the static buffers never move, so the pointer values are CONSTANT —
# build the table once into a persistent device tensor and replay reads it
# directly (no H2D copy in the graph).
_dg_kv_table = [None]


# --------------------------------------------------------------------------- #
# Corrected Triton kernel (v10e verbatim): BL (buf_len) stride for static KV.
# --------------------------------------------------------------------------- #
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
                        causal=False, table=None):
    heads, rows, dim = q.shape
    kv_heads = k.shape[0]
    streams = rows // length
    if len(keys) != streams or len(values) != streams or rows != streams * length \
            or heads % kv_heads:
        raise ValueError("one context a stream, and blocks of equal length")
    if table is None:
        host = torch.tensor(
            [p for kc, vc in zip(keys, values)
             for p in (kc.data_ptr(), vc.data_ptr())],
            dtype=torch.int64).pin_memory()
        dev = host.to(q.device, non_blocking=True)
        table = dev[:2 * streams]
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


# --------------------------------------------------------------------------- #
# _dg_layer_fast (v10e verbatim): DFlash2 layer on static KV buffers.
# --------------------------------------------------------------------------- #
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
    kv = _dg_kv_bufs[0]
    if kv is not None and len(ctx) == len(kv[0]):
        kc_bufs, vc_bufs = kv
        tbl = None
        if _dg_kv_table[0] is not None:
            stride = 2 * len(ctx)
            tbl = _dg_kv_table[0][i * stride:(i + 1) * stride]
        out = _dg_block_attention(
            q, k, v,
            [kc_bufs[j][i] for j in range(len(ctx))],
            [vc_bufs[j][i] for j in range(len(ctx))],
            length, self.window, self.head_dim ** -0.5, self.is_causal,
            table=tbl)
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


# --------------------------------------------------------------------------- #
# _graph_launch_blocks (v10e verbatim)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# _DraftStaged (v10e verbatim): static KV/index buffers for one launch_blocks
# graph (per S).
# --------------------------------------------------------------------------- #
_fillkv_n = [0]
_hooksnap_n = [0]


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

        max_W = self.W
        self.tokens_buf = torch.zeros(max_W, dtype=torch.int32,
                                      device=self.device)
        self.cpu_tokens = torch.zeros(max_W, dtype=torch.int32)

        self.lens_buf = torch.zeros(_dg_lens_max[0], dtype=torch.int32,
                                    device=self.device)

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

        # Static KV data-pointer table (v11 fix). The buffers above are
        # allocated once and never move, so their .data_ptr() values are
        # CONSTANT. Build the table into a persistent device tensor; the
        # captured graph's Triton kernel reads it directly on replay instead
        # of re-running a baked H2D copy from a freed temp pinned buffer.
        # Layout: [layer li][stream j, {K,V}] -> idx = li*2*n_live + j*2 + {0,1}
        n_layers = len(self.kc_bufs[0])
        self.kv_table = torch.tensor(
            [p for li in range(n_layers)
             for j in range(n_live)
             for p in (self.kc_bufs[j][li].data_ptr(),
                       self.vc_bufs[j][li].data_ptr())],
            dtype=torch.int64, device=self.device)
        _log(f"kv_table S={n_live} layers={n_layers} "
             f"entries={n_layers * 2 * n_live}")

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
        if _fillkv_n[0] < 14:
            import sys as _s
            for i in live:
                kc_l, vc_l, n, pos = snaps[i]
                t = kc_l[0].flatten()
                tail = t[-32:].float().cpu().tolist()
                fp = sum(float(x) for x in tail)
                _s.stderr.write(
                    f"tp2v11 SNAP i={i} n={n} pos={pos} "
                    f"len={kc_l[0].shape[1]} kvfp={fp:.6f} "
                    f"tok0={int(pendings[i])}\n")
            _s.stderr.flush()
        _fillkv_n[0] += 1
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
            self.lens_buf[s_idx] = int(n)

    def refresh(self, snaps, pendings):
        self._fill_tokens(snaps, pendings)
        self._fill_kv(snaps, pendings)
        _dg_lens_buf[0] = self.lens_buf
        _dg_buf_len[0] = self.buf_len
        _dg_kv_bufs[0] = (self.kc_bufs, self.vc_bufs)
        _dg_kv_table[0] = self.kv_table

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
            if self.replay_count < 80 or self.replay_count == 200:
                import sys as _s
                if self.replay_count == 200:
                    _s.stderr.write(
                        f"tp2v11 R200 lens=[{self.lens_buf[:self.n_live].tolist()}] "
                        f"pos=[{self.pos_buf[:self.n_live].tolist()}] "
                        f"buf_len={self.buf_len} "
                        f"kvfp={sum(float(x) for x in self.kc_bufs[0][0].flatten()[-32:].float().cpu().tolist()):.6f}\n")
                    _s.stderr.flush()
                with torch.no_grad():
                    eager = _graph_launch_blocks(self.draft,
                                     self.static_snaps,
                                     self.sample_pendings, self.max_nodes,
                                     self.block,
                                     tokens_buf=self.tokens_buf,
                                     kc_bufs=self.kc_bufs,
                                     vc_bufs=self.vc_bufs,
                                     pos_buf=self.pos_buf)
                for ii in range(len(self.out_ref)):
                    if self.out_ref[ii] is not None and eager[ii] is not None:
                        g = self.out_ref[ii][0][0]
                        e = eager[ii][0][0]
                        d = int((g != e).sum())
                        _s.stderr.write(
                            f"tp2v11 T1A_DIFF n={self.replay_count} i={ii} "
                            f"diff={d}/{g.numel()} "
                            f"g0={g.flatten()[:4].tolist()} "
                            f"e0={e.flatten()[:4].tolist()}\n")
                        _s.stderr.flush()
                        break
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


# --------------------------------------------------------------------------- #
# _capture (v10e verbatim): warm up twice, then capture launch_blocks.
# --------------------------------------------------------------------------- #
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
                                       kc_bufs=staged.kc_bufs, vc_bufs=staged.vc_bufs,
                                       pos_buf=staged.pos_buf)
        staged.graph = g
        staged.out_ref = out
    _log(f"TIER1A captured S={S} W={staged.W} buf_len={staged.buf_len} "
         f"window={staged.window}")
    return staged


# --------------------------------------------------------------------------- #
# Tier 1B — target-verify bucketed graphs
# --------------------------------------------------------------------------- #
def _slot_bucket(slots):
    """Exact gdn.cu dispatch_tree template bucket for a slot count."""
    if slots == 0:
        return 0
    if slots <= 2:
        return 1
    if slots <= 4:
        return 2
    if slots <= 8:
        return 3
    if slots <= 16:
        return 4
    return 5


# Representative slot count per bucket: dispatch_tree maps it to the same
# template as any live count inside the bucket, and its template capacity
# (B slots) covers every real topology of that bucket.
_BUCKET_REP = (0, 2, 4, 8, 16, 32)


class _V11Staged:
    """Generalized engine Staged: per-round tree topology within one GDN
    slot bucket, padded attention plan up to _KV_CAP.

    Device layout (int32):
        [0 : W]       ids
        [W : 2W]      pos
        [2W : 5W]     gdn entries (3W)
        [5W : 5W+2]   gdn starts ([0, W] — constant)
        [5W+2 : ...]  windows W*(keep+1)
        [... : end]   attn flat (rows W | streams 4 | items 3n | glob W)
    """

    def __init__(self, w, st, W, bucket):
        from tensorfold.families.qwen3_5.cuda import forward as fwd
        from tensorfold.cuda.kernels import gdn as deltanet
        from tensorfold.cuda.kernels import attention as ta

        c = w.config
        device = w.norm.device
        keep = c.conv_kernel - 1
        group = c.heads // c.kv_heads
        self.W = W
        self.keep = keep
        self.fwd = fwd
        self.deltanet = deltanet
        self.ta = ta
        self.st = st
        self.device = device

        fwd.reserve(st, _KV_CAP)             # pre-grow caches: stable bases

        # Padded attention plan (S=1) over at most _KV_CAP committed keys.
        chain = list(range(-1, W - 1))
        flat, items, chunks = ta.padded_host(chain, _KV_CAP, group)
        self.items_n = items
        self.attn_len = len(flat)

        total = 5 * W + 2 + W * (keep + 1) + self.attn_len
        host = torch.zeros(total, dtype=torch.int32).pin_memory()
        h0 = host.numpy()
        h0[5 * W:5 * W + 2] = [0, W]                        # gdn starts
        h0[5 * W + 2 + W * (keep + 1) + W + 1] = W          # attn rows
        dev = host.to(device)

        self._ids_off = 0
        self._pos_off = W
        self._ent_off = 2 * W
        self._st_off = 5 * W
        self._win_off = 5 * W + 2
        self._apl_off = 5 * W + 2 + W * (keep + 1)

        self.ids = dev[self._ids_off:self._pos_off]
        self.pos = dev[self._pos_off:self._ent_off]
        self.plan = deltanet.Plan(dev[self._ent_off:self._st_off].view(W, 3),
                                  dev[self._st_off:self._win_off],
                                  _BUCKET_REP[bucket], W)
        self.windows = dev[self._win_off:self._apl_off].view(W, keep + 1)
        self.aplan = ta.from_packed(dev[self._apl_off:], 1, W, items, chunks)

        softmax = [i for i, layer in enumerate(w.layers) if not layer.linear]
        self.aoffs = fwd._cache_offsets([st], softmax, device)

        self.host = host
        self.dev = dev
        self._fill_host(chain, 0)

    def _fill_host(self, parents, p):
        fwd, W, keep = self.fwd, self.W, self.keep
        h = self.host.numpy()
        depths, _ = fwd._paths(parents)
        h[self._pos_off:self._pos_off + W] = \
            [p + self.st.rope_delta + d for d in depths]
        entries, _, _, _ = self.deltanet.plan_host([parents])
        h[self._ent_off:self._ent_off + 3 * W] = entries
        wins = fwd._conv_windows(parents, keep)
        h[self._win_off:self._win_off + W * (keep + 1)] = wins.flatten().tolist()
        # attention streams table: (start=0, rows=W, committed p, nch)
        h[self._apl_off + W + 2] = p
        h[self._apl_off + W + 3] = -(-(p + W) // self.ta.CHUNK)
        # attention glob parents (S=1: unchanged local parents)
        g = self._apl_off + W + 4 + 3 * self.items_n
        h[g:g + W] = parents

    def refresh(self, tokens, parents, st):
        self._fill_host([int(p) for p in parents], int(st.pos))
        h = self.host.numpy()
        h[:self.W] = [int(t) for t in tokens]
        self.dev.copy_(self.host, non_blocking=True)
        # Re-derive paths/depths for this round's topology (eager kernel,
        # ordered before the replay on the same stream).
        self.ta._paths[(self.W,)](self.aplan.parents, self.aplan.paths,
                                  self.aplan.depths,
                                  MAXD=self.ta.MAX_NODES, num_warps=1)


class _MultiStagedV11:
    """Padded multi-stream staged buffers for multi_tree_forward (S = 1..8).

    Layout (int32) mirrors v3 plus a padded attention plan:
        positions(W) | sids(W) | entries(3W) | starts(S+1) | ids(W)
        | windows(W*(keep+1)) | attn_flat(padded)
    """

    def __init__(self, w, states, widths, bucket):
        from tensorfold.families.qwen3_5.cuda import forward as fwd
        from tensorfold.cuda.kernels import gdn as deltanet
        from tensorfold.cuda.kernels import attention as ta

        c = w.config
        device = w.norm.device
        keep = c.conv_kernel - 1
        group = c.heads // c.kv_heads
        S = len(states)
        W = sum(widths)
        self.S = S
        self.W = W
        self.keep = keep
        self.fwd = fwd
        self.deltanet = deltanet
        self.ta = ta
        self.st = states
        self.device = device

        for st in states:
            fwd.reserve(st, _KV_CAP)

        # Padded attention plan: per-stream items for every full chunk below
        # _KV_CAP (padded_host's construction, generalized to S streams).
        parents0 = [list(range(-1, ws - 1)) for ws in widths]
        items = []
        for s, ws in enumerate(widths):
            for chunk in range(_KV_CAP // ta.CHUNK):
                for first in range(0, ws * group, ta.QUERY_TILE):
                    items.append([s, first, chunk])
        n_items = len(items)
        self.attn_len = W + 4 * S + 3 * n_items + W
        self.items_n = n_items
        self.chunks = -(-(_KV_CAP + max(widths)) // ta.CHUNK)

        total = 6 * W + (S + 1) + W * (keep + 1) + self.attn_len
        host = torch.zeros(total, dtype=torch.int32).pin_memory()
        h0 = host.numpy()
        starts = [0]
        for ws in widths:
            starts.append(starts[-1] + ws)
        self.starts = starts
        h0[5 * W:5 * W + S + 1] = starts                    # gdn starts
        soff = 6 * W + S + 1 + W * (keep + 1) + W           # streams table
        for s, ws in enumerate(widths):
            h0[soff + 4 * s + 0] = starts[s]
            h0[soff + 4 * s + 1] = ws
        for s in range(S):
            for r in range(widths[s]):
                h0[W + starts[s] + r] = s                   # sids (static)
        dev = host.to(device)

        self.pos_off = 0
        self.sids_off = W
        self.ent_off = 2 * W
        self.st_off = 5 * W
        self.ids_off = 5 * W + S + 1
        self.win_off = 6 * W + S + 1
        self.apl_off = 6 * W + S + 1 + W * (keep + 1)

        self.pos = dev[self.pos_off:W]
        self.sid_t = dev[self.sids_off:self.ent_off]
        self.plan = deltanet.Plan(dev[self.ent_off:self.st_off].view(W, 3),
                                  dev[self.st_off:self.ids_off],
                                  _BUCKET_REP[bucket], max(widths))
        self.ids = dev[self.ids_off:self.win_off]
        self.windows = dev[self.win_off:self.apl_off].view(W, keep + 1)
        self.aplan = ta.from_packed(dev[self.apl_off:], S, W, n_items,
                                    self.chunks)

        softmax = [i for i, layer in enumerate(w.layers) if not layer.linear]
        self.aoffs = fwd._cache_offsets(list(states), softmax, device)

        linear = [i for i, layer in enumerate(w.layers) if layer.linear]
        ptrs = [p for i in linear for p in
                deltanet.pointers([st.rec[i] for st in states])]
        self.tables = dict(
            zip(linear, deltanet.to_device(ptrs, torch.int64, device)
                .view(len(linear), S)))

        # Static conv concatenation for S > 1 (cat allocates per round in the
        # stock path; we copy into one stable buffer instead).
        self.conv_bufs = {}
        if S > 1:
            for i in linear:
                shape = (S, keep, states[0].conv[i].shape[-1])
                self.conv_bufs[i] = torch.zeros(*shape, dtype=states[0].conv[i].dtype,
                                                device=device)

        self.host = host
        self.dev = dev
        self._fill_host(parents0, [0] * S)

    def _fill_host(self, parents_list, ps):
        fwd, W, keep = self.fwd, self.W, self.keep
        h = self.host.numpy()
        local = []
        pos_rows, win_rows, glob = [], [], []
        for s, (parents, p) in enumerate(zip(parents_list, ps)):
            parents = [int(x) for x in parents]
            local.append(parents)
            base = sum(len(x) for x in local[:-1])
            depths, _ = fwd._paths(parents)
            pos_rows += [p + self.st[s].rope_delta + d for d in depths]
            rows = []
            for row, parent in enumerate(parents):
                tail = list(range(keep)) if parent < 0 else rows[parent][1:]
                rows.append(tail + [keep + base + row])
            win_rows += [i for r in rows for i in r]
            glob += [-1 if x < 0 else x + base for x in parents]
        entries, _, _, _ = self.deltanet.plan_host(local)
        h[self.pos_off:W] = pos_rows
        h[self.ent_off:self.ent_off + 3 * W] = entries
        h[self.win_off:self.win_off + W * (keep + 1)] = win_rows
        # attention streams table rows: (start, rows, p, nch) per stream
        off = self.apl_off + W
        for s, (parents, p) in enumerate(zip(parents_list, ps)):
            ws = len(parents)
            h[off + 4 * s + 1] = ws
            h[off + 4 * s + 2] = p
            h[off + 4 * s + 3] = -(-(p + ws) // self.ta.CHUNK)
        g = self.apl_off + W + 4 * self.S + 3 * self.items_n
        h[g:g + W] = glob
        # sids
        for s, parents in enumerate(parents_list):
            for r in range(len(parents)):
                idx = sum(len(x) for x in parents_list[:s]) + r
                h[self.sids_off + idx] = s

    def refresh(self, streams):
        """streams: [(tokens, parents), ...] parallel to self.st."""
        W = self.W
        h = self.host.numpy()
        parents_list, ps, toks = [], [], []
        for s, (tokens, parents) in enumerate(streams):
            parents_list.append([int(p) for p in parents])
            ps.append(int(self.st[s].pos))
            base = sum(len(x) for x in parents_list[:-1])
            toks += [int(t) for t in tokens]
        h[self.ids_off:self.ids_off + W] = toks
        self._fill_host(parents_list, ps)
        self.dev.copy_(self.host, non_blocking=True)
        self.ta._paths[(self.W,)](self.aplan.parents, self.aplan.paths,
                                  self.aplan.depths,
                                  MAXD=self.ta.MAX_NODES, num_warps=1)
        if self.S > 1:
            for i, buf in self.conv_bufs.items():
                for s, st in enumerate(self.st):
                    buf[s].copy_(st.conv[i])


def _multi_tree_forward_v11(w, streams, staged, *, full_logits=True, tp=True,
                            capture_taps=False):
    """multi_tree_forward using staged static buffers (capture/replay body).

    Mirrors the stock function exactly except: static inputs from ``staged``,
    static conv concat for S > 1, padded aplan.  Byte-identical math.
    """
    from tensorfold.families.qwen3_5.cuda import forward as fwd
    from tensorfold.cuda.kernels import gdn as deltanet
    from tensorfold.cuda.kernels import attention as ta
    from tensorfold.families.qwen3_5.cuda import glue
    from tensorfold.cuda import kv_fp8
    from tensorfold.families.qwen3_5.cuda.forward import (
        _mm, _mlp, _row_mm)

    c = w.config
    keep = c.conv_kernel - 1
    S = staged.S
    W = staged.W
    states = staged.st

    dev = staged.dev
    pos = staged.pos
    sid_t = staged.sid_t
    plan = staged.plan
    ids_t = staged.ids
    windows_t = staged.windows
    aplan = staged.aplan
    tables = staged.tables
    aoffs = staged.aoffs

    x = glue.embedding(ids_t, w.embed)
    pending = None
    record = []
    taps = []
    for i, layer in enumerate(w.layers):
        x, h, xs = glue.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        if layer.linear:
            gdn = layer.gdn
            qkv = _mm(h, gdn.qkv, xs)
            if gdn.zba is not None:
                zba = _mm(h, gdn.zba, xs)
                vd = c.v_heads * c.dv
                z = zba[:, :vd].contiguous().reshape(W, c.v_heads, c.dv)
                b = zba[:, vd:vd + c.v_heads].contiguous()
                a = zba[:, vd + c.v_heads:].contiguous()
            else:
                z = _mm(h, gdn.z, xs).reshape(W, c.v_heads, c.dv)
                b = _mm(h, gdn.b, xs)
                a = _mm(h, gdn.a, xs)
            if S == 1:
                conv = states[0].conv[i]
            else:
                conv = staged.conv_bufs[i]
            q, k, v, g, beta = glue.gdn_pre(qkv, conv, gdn.conv, windows_t, a, b,
                                             gdn.A_log, gdn.dt_bias, kh=c.k_heads,
                                             vh=c.v_heads, dk=c.dk,
                                             stream_ids=sid_t, nkeep=keep)
            yr = deltanet.tree(q, k, v, g, beta, plan, table=tables[i])
            out, out_xs = glue.gated_norm(yr, z, gdn.norm, c.eps)
            r = _row_mm(out, gdn.out, tp, out_xs)
            record.append(fwd.GDNRecord(q, k, v, g, beta, qkv))
        else:
            attn = layer.attn
            qg = _mm(h, attn.q, xs)
            if attn.kv is not None:
                kv = _mm(h, attn.kv, xs)
                kd = c.kv_heads * c.head_dim
                key = kv[:, :kd].contiguous()
                value = kv[:, kd:].contiguous().reshape(W, c.kv_heads, c.head_dim)
            else:
                key = _mm(h, attn.k, xs)
                value = _mm(h, attn.v, xs).reshape(W, c.kv_heads, c.head_dim)
            q, key = glue.attn_prep(qg, key, attn.q_norm, attn.k_norm, pos,
                                    w.inv_freq, c.eps, heads=c.heads,
                                    kv_heads=c.kv_heads, head_dim=c.head_dim,
                                    mscale=c.rope_mscale)
            key, value = kv_fp8.round(key), kv_fp8.round(value)
            out = ta.attention(q, key, value, aoffs[i], aplan,
                               scale=c.head_dim ** -0.5)
            gated, out_xs = glue.gate_mul(out, qg, heads=c.heads, head_dim=c.head_dim)
            r = _row_mm(gated, attn.o, tp, out_xs)
            record.append(fwd.AttentionRecord(key, value))
        x, h, xs = glue.add_rmsnorm(x, r, layer.post_norm, c.eps)
        pending = _mlp(layer, h, xs, tp)
        if capture_taps and i in (5, 19, 33, 47, 61):
            taps.append((x.float() + pending.float()).to(torch.bfloat16))
    _, h, xs = glue.add_rmsnorm(x, pending, w.norm, c.eps)
    logits = _mm(h, w.head, xs) if full_logits else h
    starts = staged.starts
    if capture_taps:
        if len(taps) != 5:
            raise ValueError("DFlash2 taps require the complete 64-layer target")
        return logits, record, torch.cat(taps, dim=-1), starts
    return logits, record, None, starts


class _VerifyGraphs:
    """Bucketed CUDA graphs for tree_forward (S=1) and multi_tree_forward."""

    def __init__(self, w):
        self.w = w
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}          # key -> (g, staged, out)
        self.order = []           # LRU order
        self.disabled = False

    def _evict(self):
        while len(self.graphs) >= _V11_MAX_GRAPHS and self.order:
            old = self.order.pop(0)
            if old in self.graphs:
                g, staged, out = self.graphs.pop(old)
                del g, staged, out
                _log(f"TIER1B evicted key={old}")

    # ---- single stream (decode_tp.tree_forward) --------------------------
    def try_replay_tf(self, tokens, parents, st, full_logits, capture_taps):
        from tensorfold.cuda.kernels import gdn as deltanet
        from tensorfold.families.qwen3_5.cuda import forward as fwd
        W = len(parents)
        if W > 16:
            return None
        if int(st.pos) + W > _KV_CAP:
            return None
        try:
            _, _, slots, _ = deltanet.plan_host([list(parents)])
        except Exception:
            return None
        key = ("tf", id(st), _slot_bucket(slots), W, full_logits, capture_taps)
        e = self.graphs.get(key)
        if e is None:
            if self.disabled:
                return None
            try:
                self._capture_tf(key, st, tokens, parents, full_logits,
                                 capture_taps)
            except Exception as ex:
                _log(f"TIER1B tf capture failed key={key}: {ex!r} -> eager")
                if len(self.graphs) >= _V11_MAX_GRAPHS:
                    self.disabled = True
                return None
            e = self.graphs.get(key)
        if e is None:
            return None
        g, staged, out = e
        self._touch(key)
        staged.refresh(tokens, parents, st)
        g.replay()
        return out

    def _capture_tf(self, key, st, tokens, parents, full_logits, capture_taps):
        from tensorfold.families.qwen3_5.cuda import forward as fwd
        W = len(parents)
        self._evict()
        if len(self.graphs) >= _V11_MAX_GRAPHS:
            self.disabled = True
            return
        staged = _V11Staged(self.w, st, W, key[2])
        staged.refresh(tokens, parents, st)
        for _ in range(2):
            self._run_tf(staged, st, parents, full_logits, capture_taps)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = self._run_tf(staged, st, parents, full_logits, capture_taps)
        self.graphs[key] = (g, staged, out)
        self.order.append(key)
        _log(f"TIER1B tf captured key={key} (st={st.pos})")

    def _run_tf(self, staged, st, parents, full_logits, capture_taps):
        from tensorfold.families.qwen3_5.cuda import forward as fwd
        return fwd.tree_forward(self.w, staged.ids, parents, st, tp=True,
                                full_logits=full_logits,
                                capture_taps=capture_taps, staged=staged)

    # ---- multi stream (scheduler multi_tree_forward) ---------------------
    def try_replay_multi(self, streams, full_logits, capture_taps):
        from tensorfold.cuda.kernels import gdn as deltanet
        S = len(streams)
        if S < 1 or S > 8:
            return None
        widths = [len(p) for (_, p, _) in streams]
        if any(ws > 16 for ws in widths):
            return None
        states = [st for (_, _, st) in streams]
        if any(int(st.pos) + ws > _KV_CAP for ws, st in zip(widths, states)):
            return None
        try:
            _, _, slots, _ = deltanet.plan_host([list(p) for (_, p, _) in streams])
        except Exception:
            return None
        key = ("mtf", tuple(id(st) for st in states), _slot_bucket(slots),
               tuple(widths), full_logits, capture_taps)
        e = self.graphs.get(key)
        if e is None:
            if self.disabled:
                return None
            try:
                self._capture_multi(key, streams, full_logits, capture_taps)
            except Exception as ex:
                _log(f"TIER1B mtf capture failed key={key}: {ex!r} -> eager")
                if len(self.graphs) >= _V11_MAX_GRAPHS:
                    self.disabled = True
                return None
            e = self.graphs.get(key)
        if e is None:
            return None
        g, staged, out = e
        self._touch(key)
        staged.refresh([(t, p) for (t, p, _) in streams])
        g.replay()
        return out

    def _capture_multi(self, key, streams, full_logits, capture_taps):
        from tensorfold.cuda.kernels import gdn as deltanet
        S = len(streams)
        widths = [len(p) for (_, p, _) in streams]
        states = [st for (_, _, st) in streams]
        self._evict()
        if len(self.graphs) >= _V11_MAX_GRAPHS:
            self.disabled = True
            return
        staged = _MultiStagedV11(self.w, states, widths, key[2])
        staged.refresh([(t, p) for (t, p, _) in streams])
        for _ in range(2):
            self._run_multi(staged, streams, full_logits, capture_taps)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = self._run_multi(staged, streams, full_logits, capture_taps)
        self.graphs[key] = (g, staged, out)
        self.order.append(key)
        _log(f"TIER1B mtf captured key={key} S={S} W={widths}")

    def _run_multi(self, staged, streams, full_logits, capture_taps):
        return _multi_tree_forward_v11(self.w, streams, staged,
                                       full_logits=full_logits, tp=True,
                                       capture_taps=capture_taps)

    def _touch(self, key):
        if key in self.order:
            self.order.remove(key)
        self.order.append(key)


# --------------------------------------------------------------------------- #
# install
# --------------------------------------------------------------------------- #
def install():
    if not _GRAPH_D_ON and not _V11_GRAPH_ON:
        sys.stderr.write("tp2v11: off (TP2_DRAFT_GRAPH=0)\n")
        sys.stderr.flush()
        return

    import tensorfold.families.qwen3_5.cuda.dflash2 as d2
    orig_lb = d2.DFlash2.launch_blocks
    state = {"graphs": {}, "pool": None, "warm": [0], "cache": {}}

    def _stage_S(self, S_val, mn, bl, s_snaps, s_pend):
        """v10e staging, plus v11 Tier 1A: capture a CUDA graph when enabled."""
        if _V11_GRAPH_ON:
            if state["pool"] is None:
                state["pool"] = torch.cuda.graph_pool_handle()
            try:
                staged = _capture(self, S_val, mn, bl, s_snaps, s_pend,
                                  pool=state["pool"])
                state["graphs"][S_val] = staged
                return
            except Exception as e:
                import traceback
                _log(f"TIER1A capture S={S_val} failed: {e!r} -> eager staged")
                for line in traceback.format_exc().splitlines()[:8]:
                    _log(f"  tb: {line}")
        staged = _DraftStaged(self, S_val, mn, bl, s_snaps, s_pend)
        staged.refresh(snaps=s_snaps, pendings=s_pend)
        staged.graph = None
        state["graphs"][S_val] = staged
        _log(f"v11: staged S={S_val} (eager, no graph)")

    def _patched_lb(self, snaps, pendings, max_nodes, block=None):
        _log_state(snaps, pendings, "pre")
        if _hooksnap_n[0] < 14:
            import sys as _s
            for i, s in enumerate(snaps):
                if s[2] > 0:
                    kc_l, vc_l, n, pos = s
                    t = kc_l[0].flatten()
                    tail = t[-32:].float().cpu().tolist()
                    fp = sum(float(x) for x in tail)
                    _s.stderr.write(
                        f"tp2v11 HS i={i} n={n} pos={pos} "
                        f"len={kc_l[0].shape[1]} kvfp={fp:.6f} "
                        f"tok0={int(pendings[i])}\n")
                    break
            _s.stderr.flush()
        _hooksnap_n[0] += 1
        if getattr(self, "world", 1) != 1:
            return orig_lb(self, snaps, pendings, max_nodes, block)
        live = [i for i, s in enumerate(snaps) if s[2] > 0]
        S = len(live)
        if S < 1 or S > 8:
            return orig_lb(self, snaps, pendings, max_nodes, block)
        if S not in state["cache"]:
            state["cache"][S] = ([snaps[i] for i in live],
                                 [pendings[i] for i in live],
                                 max_nodes, block)
            _log(f"v11: cached S={S} warm={state['warm'][0]}")
        state["warm"][0] += 1
        if state["warm"][0] <= _GRAPH_D_WARMUP:
            return orig_lb(self, snaps, pendings, max_nodes, block)
        if S in state["graphs"]:
            out = state["graphs"][S].try_replay(snaps, pendings,
                                                max_nodes, block)
            if out is not None:
                _replay_count(0)
                return out
        return orig_lb(self, snaps, pendings, max_nodes, block)

    def _patched_lb_with_capture(self, snaps, pendings, max_nodes, block=None):
        out = _patched_lb(self, snaps, pendings, max_nodes, block)
        if state["warm"][0] == _GRAPH_D_WARMUP + 1:
            for S_val in sorted(state["cache"].keys()):
                s_snaps, s_pend, mn, bl = state["cache"][S_val]
                if S_val not in state["graphs"]:
                    try:
                        _stage_S(self, S_val, mn, bl, s_snaps, s_pend)
                    except Exception as e:
                        _log(f"stage S={S_val} failed: {e!r}")
        if state["warm"][0] > _GRAPH_D_WARMUP:
            S_now = sum(1 for s in snaps if s[2] > 0)
            if S_now in state["cache"] and S_now not in state["graphs"]:
                s_snaps, s_pend, mn, bl = state["cache"][S_now]
                try:
                    _stage_S(self, S_now, mn, bl, s_snaps, s_pend)
                except Exception as e:
                    _log(f"stage S={S_now} failed: {e!r}")
        return out

    d2.DFlash2.launch_blocks = _patched_lb_with_capture
    _dg_orig_lf[0] = d2.DFlash2._layer_fast
    d2.DFlash2._layer_fast = _dg_layer_fast

    # ---- Tier 1B: verify bucketed graphs --------------------------------- #
    if _V11_TARGET_ON:
        _install_verify_graphs()

    sys.stderr.write(
        f"tp2v11: installed (Tier1A={'on' if _V11_GRAPH_ON else 'off'}, "
        f"Tier1B={'on' if _V11_TARGET_ON else 'off'}, "
        f"max_graphs={_V11_MAX_GRAPHS}, warmup={_V11_VERIFY_WARMUP}, "
        f"KV_CAP={_KV_CAP})\n")
    sys.stderr.flush()


def _install_verify_graphs():
    import tensorfold.families.qwen3_5.cuda.forward as fwd
    import tensorfold.families.qwen3_5.cuda.multi as multi
    import tensorfold.families.qwen3_5.cuda.decode_tp as dtp
    from tensorfold.cuda.kernels import gdn as deltanet

    # ---- tree_forward (decode_tp — the live C=1 verify path) ------------- #
    orig_tf = dtp.tree_forward
    _tf_n = [0]
    vg = {"obj": None}

    def _patched_tf(w, tokens, parents, st, *, full_logits=True, tp=False,
                    initial=None, finish=True, capture_taps=False,
                    hidden=False, staged=None):
        _tf_n[0] += 1
        if (_tf_n[0] <= 3):
            _log(f"TF_ENTRY tp={tp} W={len(parents)} hidden={hidden} "
                 f"taps={capture_taps} staged={staged is not None} "
                 f"n={_tf_n[0]}")
        if (tp and staged is None and initial is None and not hidden
                and finish and _tf_n[0] > _V11_VERIFY_WARMUP):
            if vg["obj"] is None:
                vg["obj"] = _VerifyGraphs(w)
            try:
                out = vg["obj"].try_replay_tf(tokens, parents, st,
                                              full_logits, capture_taps)
            except Exception as ex:
                if vg["obj"] is not None:
                    vg["obj"].disabled = True
                _log(f"TIER1B tf replay error: {ex!r} -> eager")
                out = None
            if out is not None:
                _replay_count(1)
                return out
        return orig_tf(w, tokens, parents, st, full_logits=full_logits, tp=tp,
                       initial=initial, finish=finish,
                       capture_taps=capture_taps, hidden=hidden, staged=staged)

    dtp.tree_forward = _patched_tf

    # ---- multi_tree_forward (forward source + multi import site) --------- #
    orig_mtf = fwd.multi_tree_forward
    _mtf_n = [0]

    def _patched_mtf(w, streams, *, full_logits=True, tp=False,
                     capture_taps=False, hidden=False):
        alive = all(st.pos > 0 for (_, _, st) in streams)
        if alive:
            _mtf_n[0] += 1
        if _mtf_n[0] <= 3:
            toks, par, st = streams[0]
            _log(f"MTF_ENTRY n={len(streams)} W={len(par)} n={_mtf_n[0]}")
        if (tp and not hidden and alive
                and _mtf_n[0] > _V11_VERIFY_WARMUP):
            if vg["obj"] is None:
                vg["obj"] = _VerifyGraphs(w)
            try:
                out = vg["obj"].try_replay_multi(streams, full_logits,
                                                 capture_taps)
            except Exception as ex:
                if vg["obj"] is not None:
                    vg["obj"].disabled = True
                _log(f"TIER1B mtf replay error: {ex!r} -> eager")
                out = None
            if out is not None:
                _replay_count(2)
                return out
        return orig_mtf(w, streams, full_logits=full_logits, tp=tp,
                        capture_taps=capture_taps, hidden=hidden)

    fwd.multi_tree_forward = _patched_mtf
    multi.multi_tree_forward = _patched_mtf


# --------------------------------------------------------------------------- #
# install() is called by the tensorfold/__init__.py loader shim (v11 block).
# --------------------------------------------------------------------------- #
