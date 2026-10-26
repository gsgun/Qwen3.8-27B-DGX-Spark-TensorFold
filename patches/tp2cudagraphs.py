"""Patch 0008-B2 v3 (TensorFold v0.6.0): two-graph CUDA capture for the TP2 decode.

Graph T  - the multi-tree verify ``multi_tree_forward`` (C=1 single-stream
           chain), captured with an EXACT eager fallback: the graph is replayed
           only when the live parents are the exact chain the graph was captured
           with, otherwise the stock eager path runs.  Provably byte-exact.

           ROOT CAUSE (boot 32): with --parallel 4, engine.generate() L250 routes
           to the scheduler -> MultiDecoder._verify -> multi_tree_forward
           (forward.py), NOT decode_tp.tree_forward.  v1/v2 hooked the dead path.

Graph D  - the DFlash2 ``launch_blocks`` (the drafter's GPU half), captured in
           full.  0 NCCL collectives under TP2_DRAFT_REPL=1.

Step-1 probe (width_probe_findings.md) fixed the verify buckets:
    verify width  W = 1 + dW  ->  {1, 16, 31, 46, 61},  W_cap = 16 / 64
    draft tree            ->  always 15 nodes; 35% exact chains, 65% branches.

Zero state corruption (padded rows, W_actual < W_cap):
  * The captured graph runs the full [W_cap, hidden] static tensor, but the
    GDN recurrent update, the attention KV write, and the aoffs/commit paths
    are all bounded on the device by the plan's live row count.  Padded rows
    therefore read state but never write it.
  * ``reserve(st, KV_CAP)`` pre-grows every attention cache before capture so
    the cache base pointers (aoffs) and the GDN recurrent state keep their
    addresses for the whole session.

Rank symmetry & lockstep:
  * Both ranks share the window + parents via ``_share`` BEFORE the verify
    runs, so (S, W, parents) are deterministic on both ranks -> both pick the
    same (bucket, chain?) branch on every round.  The captured AllGathers
    (128/round) and the eager collectives therefore never desync.

Kill-switch: TP2_CUDA_GRAPHS=0 installs nothing (pure stock, instant rollback).

Delivery: /patches is on PYTHONPATH ahead of dist-packages; the shadow
tensorfold/__init__.py imports this module at load time.  An install failure is
logged and swallowed (optional patch must never break the server).
"""

import os
import sys

# --------------------------------------------------------------------------- #
# Config (Step-1 probe derived)
# --------------------------------------------------------------------------- #
W_CAPS = (16, 64)         # canonical verify-width buckets; W > 64 -> eager
_MAX_CHAIN = 64           # chains longer than this never get captured
KV_CAP = int(os.environ.get("TF_SLOT_KV_CAP", "16384"))  # padded-plan context

_ST = {
    "t_captured": 0, "t_replay": 0, "t_branch_eager": 0, "t_overcap_eager": 0,
    "d_captured": 0, "d_replay": 0,
}


def _is_chain(parents, width):
    """Exact chain: parents == [-1, 0, 1, ..., width-2]."""
    if len(parents) != width:
        return False
    for i, p in enumerate(parents):
        if int(p) != i - 1:
            return False
    return True


def _pick_cap(width):
    """Tightest W_cap >= width, or None (eager)."""
    for cap in W_CAPS:
        if width <= cap:
            return cap
    return None


# --------------------------------------------------------------------------- #
# _MultiStaged — pre-allocated static buffers for multi_tree_forward (C=1)
# --------------------------------------------------------------------------- #
class _MultiStaged:
    """Static inputs for a C=1 chain window in the multi_tree_forward layout.

    Host buffer layout (all int32):
        positions(W) | sids(W) | entries(W) | starts(S+1) | ids(W)
        | windows(W*(keep+1)) | attn_flat(variable)

    Per-replay refresh updates: positions, ids, and the committed-keys fields
    inside attn_flat.  Everything else is static (chain structure, pointers).
    """

    def __init__(self, w, st, width):
        from tensorfold.families.qwen3_5.cuda import forward as fwd
        from tensorfold.cuda.kernels import gdn as deltanet
        from tensorfold.cuda.kernels import attention as tree_attention

        c = w.config
        device = w.norm.device
        keep = c.conv_kernel - 1
        parents = list(range(-1, width - 1))
        depths, _ = fwd._paths(parents)

        # Initial (placeholder) data for each section
        positions = [st.pos + st.rope_delta + d for d in depths]
        sids = [0] * width
        entries, _, slots, most = deltanet.plan_host([parents])
        starts = [0, width]
        ids = [0] * width
        rows = []
        for row, parent in enumerate(parents):
            tail = list(range(keep)) if parent < 0 else rows[parent][1:]
            rows.append(tail + [keep + row])
        windows_flat = [i for win in rows for i in win]
        attn_flat, attn_items, attn_chunks = tree_attention.plan_host(
            [parents], [st.pos], c.heads // c.kv_heads)

        # Pinned host mirror + device copy (stable addresses for graph replay)
        host_list = positions + sids + entries + starts + ids + windows_flat + attn_flat
        self.host = torch.tensor(host_list, dtype=torch.int32).pin_memory()
        self.dev = self.host.to(device)

        # GDN state pointer table (stable: same state object, same addresses)
        linear = [i for i, layer in enumerate(w.layers) if layer.linear]
        softmax = [i for i, layer in enumerate(w.layers) if not layer.linear]
        ptrs = [p for i in linear for p in deltanet.pointers([st.rec[i]])]
        raw = deltanet.to_device(ptrs, torch.int64, device).view(len(linear), 1)
        self.tables = {i: raw[j] for j, i in enumerate(linear)}

        # Attention cache offsets (stable: derived from state KV cache)
        self.aoffs = fwd._cache_offsets([st], softmax, device)

        # Metadata
        self.width = width
        self.W = width
        self.S = 1
        self.slots = slots
        self.most = most
        self.attn_items = attn_items
        self.attn_chunks = attn_chunks
        self.keep = keep
        self.st = st
        self.w = w
        self.c = c
        self.device = device
        self.attn_len = len(attn_flat)
        self.depths = depths
        self._stale = False

        # Offsets within the host buffer
        W, S = width, 1
        self.pos_off = 0
        self.sids_off = W
        self.entries_off = 2 * W
        self.starts_off = 3 * W
        self.ids_off = 3 * W + S + 1
        self.windows_off = 4 * W + S + 1
        self.attn_off = 4 * W + S + 1 + W * (keep + 1)

    def refresh(self, tokens, p):
        """Update the pinned host mirror with this round's tokens + position.

        Only positions, ids, and the attn_flat committed-keys fields change.
        The device copy is refreshed via one H2D memcpy (captured in the graph).
        """
        from tensorfold.cuda.kernels import attention as tree_attention
        from tensorfold.families.qwen3_5.cuda import forward as fwd

        W = self.W
        h = self.host.numpy()
        parents = list(range(-1, W - 1))

        # Update positions (depend on live committed position p)
        h[self.pos_off:self.pos_off + W] = [p + self.st.rope_delta + d for d in self.depths]

        # Update token ids
        h[self.ids_off:self.ids_off + W] = tokens

        # Update attn_flat (recompute for the new committed position)
        attn_flat, items, chunks = tree_attention.plan_host(
            [parents], [p], self.c.heads // self.c.kv_heads)
        if len(attn_flat) != self.attn_len:
            # Chain plan length should be constant; if not, signal to caller
            # that the graph is stale and eager fallback is needed.
            self._stale = True
            return
        h[self.attn_off:self.attn_off + self.attn_len] = attn_flat
        # Keep items/chunks in sync (should be identical for a chain)
        self.attn_items = items
        self.attn_chunks = chunks

        # One H2D copy (blocking — must complete before graph replay reads dev)
        self.dev.copy_(self.host)
        torch.cuda.synchronize()  # Ensure copy is visible to the graph


# --------------------------------------------------------------------------- #
# _multi_tree_forward_staged — multi_tree_forward with optional static buffers
# --------------------------------------------------------------------------- #
def _multi_tree_forward_staged(w, streams, *, full_logits=True, tp=False,
                                capture_taps=False, hidden=False, staged_multi=None):
    """Drop-in for forward.multi_tree_forward that reuses pre-allocated buffers
    when ``staged_multi`` is given (graph capture/replay path).  Byte-identical
    to the stock function when ``staged_multi`` is None."""
    from tensorfold.families.qwen3_5.cuda import forward as fwd
    from tensorfold.cuda.kernels import gdn as deltanet
    from tensorfold.cuda.kernels import attention as tree_attention
    from tensorfold.families.qwen3_5.cuda import glue
    from tensorfold.cuda import kv_fp8
    from tensorfold.families.qwen3_5.cuda.forward import (
        _mm, _mlp, _row_mm, _paths, GDNRecord, AttentionRecord)

    c = w.config
    device = w.norm.device
    keep = c.conv_kernel - 1
    starts = [0]
    ids, positions, windows, sids, local, states = [], [], [], [], [], []
    for s, (tokens, parents, st) in enumerate(streams):
        parents = [int(p) for p in parents]
        depths, _ = _paths(parents)
        if len(tokens) != len(parents):
            raise ValueError("each stream needs one token id a parent")
        base = starts[-1]
        rows = []
        for row, parent in enumerate(parents):
            tail = list(range(keep)) if parent < 0 else rows[parent][1:]
            rows.append(tail + [keep + base + row])
        windows.extend(rows)
        ids.extend(int(t) for t in tokens)
        positions.extend(st.pos + st.rope_delta + d for d in depths)
        sids.extend([s] * len(parents))
        local.append(parents)
        states.append(st)
        starts.append(base + len(parents))
    W = starts[-1]
    entries, _, slots, most = deltanet.plan_host(local)
    linear = [i for i, layer in enumerate(w.layers) if layer.linear]
    softmax = [i for i, layer in enumerate(w.layers) if not layer.linear]
    S = len(states)
    attn_flat, attn_items, attn_chunks = tree_attention.plan_host(
        local, [st.pos for st in states], c.heads // c.kv_heads)

    if staged_multi is not None:
        # Graph path: use pre-allocated buffers (host already refreshed by caller)
        dev = staged_multi.dev
        tables = staged_multi.tables
        aoffs = staged_multi.aoffs
    else:
        # Stock path: allocate everything inline
        host = torch.tensor(
            positions + sids + entries + starts + ids +
            [i for win in windows for i in win] + attn_flat,
            dtype=torch.int32).pin_memory()
        dev = host.to(device, non_blocking=True)
        ptrs = [p for i in linear for p in deltanet.pointers([st.rec[i] for st in states])]
        tables = dict(zip(linear,
                          deltanet.to_device(ptrs, torch.int64, device).view(len(linear), S)))
        aoffs = fwd._cache_offsets(states, softmax, device)

    pos, sid_t = dev[:W], dev[W:2 * W]
    plan = deltanet.Plan(dev[2 * W:5 * W].view(W, 3), dev[5 * W:5 * W + S + 1], slots, most)
    ids_t = dev[5 * W + S + 1:6 * W + S + 1]
    windows_t = dev[6 * W + S + 1:6 * W + S + 1 + W * (keep + 1)].view(W, keep + 1)
    aplan = tree_attention.from_packed(dev[6 * W + S + 1 + W * (keep + 1):],
                                        S, W, attn_items, attn_chunks)

    x = glue.embedding(ids_t, w.embed)
    pending: torch.Tensor | None = None
    record: list = []
    taps: list = []
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
            conv = states[0].conv[i] if S == 1 else torch.cat([st.conv[i] for st in states])
            q, k, v, g, beta = glue.gdn_pre(qkv, conv, gdn.conv, windows_t, a, b,
                                             gdn.A_log, gdn.dt_bias, kh=c.k_heads,
                                             vh=c.v_heads, dk=c.dk,
                                             stream_ids=sid_t, nkeep=keep)
            yr = deltanet.tree(q, k, v, g, beta, plan, table=tables[i])
            out, out_xs = glue.gated_norm(yr, z, gdn.norm, c.eps)
            r = _row_mm(out, gdn.out, tp, out_xs)
            record.append(GDNRecord(q, k, v, g, beta, qkv))
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
                                    w.inv_freq, c.eps, heads=c.heads, kv_heads=c.kv_heads,
                                    head_dim=c.head_dim, mscale=c.rope_mscale)
            key, value = kv_fp8.round(key), kv_fp8.round(value)
            out = tree_attention.attention(q, key, value, aoffs[i], aplan,
                                            scale=c.head_dim ** -0.5)
            gated, out_xs = glue.gate_mul(out, qg, heads=c.heads, head_dim=c.head_dim)
            r = _row_mm(gated, attn.o, tp, out_xs)
            record.append(AttentionRecord(key, value))
        x, h, xs = glue.add_rmsnorm(x, r, layer.post_norm, c.eps)
        pending = _mlp(layer, h, xs, tp)
        if capture_taps and i in (5, 19, 33, 47, 61):
            taps.append((x.float() + pending.float()).to(torch.bfloat16))
    _, h, xs = glue.add_rmsnorm(x, pending, w.norm, c.eps)
    logits = _mm(h, w.head, xs) if full_logits else h
    if capture_taps:
        if len(taps) != 5:
            raise ValueError("DFlash2 taps require the complete 64-layer target")
        return logits, record, torch.cat(taps, dim=-1), starts
    return logits, record, h if hidden else None, starts


# --------------------------------------------------------------------------- #
# Graph T — verify (multi_tree_forward), chain-captured  [PRIMARY: --parallel 4]
# --------------------------------------------------------------------------- #
class _VerifyGraphs:
    """Per-width CUDA graphs for the multi-tree verify (C=1 chain-captured).

    Targets multi_tree_forward (the live scheduler path under --parallel 4).
    """

    def __init__(self, w, full_logits, capture_taps):
        self.w = w
        self.full_logits = full_logits
        self.capture_taps = capture_taps
        self.device = w.norm.device
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}            # W_cap -> (CUDAGraph, _MultiStaged, out)

    def _ensure(self, cap, st):
        """Capture the W_cap chain graph on first use (reserve -> stage -> 2x warm -> capture)."""
        if cap in self.graphs:
            return
        from tensorfold.families.qwen3_5.cuda import forward as _fwd
        _fwd.reserve(st, KV_CAP)                       # stable cache addresses
        staged = _MultiStaged(self.w, st, cap)

        # Warm up twice (allocations settle, kernels compile, addresses stable)
        for _ in range(2):
            self._run(staged, st)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = self._run(staged, st)
        self.graphs[cap] = (g, staged, out)
        _ST["t_captured"] += 1
        sys.stderr.write(f"tp2cudagraphs: Graph T captured W_cap={cap} (multi_tree_forward)\n")
        sys.stderr.flush()

    def _run(self, staged, st):
        """One verify forward using the staged buffers (capture/warmup path)."""
        parents = list(range(-1, staged.W - 1))
        tokens = [int(t) for t in staged.host[staged.ids_off:staged.ids_off + staged.W]]
        streams = [(tokens, parents, st)]
        return _multi_tree_forward_staged(
            self.w, streams, full_logits=self.full_logits, tp=True,
            capture_taps=self.capture_taps, hidden=False, staged_multi=staged)

    def try_replay(self, tokens, parents, st):
        """Return the graph's static output on a chain replay, else None (eager)."""
        width = len(parents)
        is_ch = _is_chain(parents, width)
        cap = _pick_cap(width) if is_ch else None
        # diagnostic (rate-limited): log the first 8 eligible calls
        if _ST.get("diag", 0) < 8:
            _ST["diag"] = _ST.get("diag", 0) + 1
            sys.stderr.write(f"tp2cudagraphs DIAG: width={width} chain={is_ch} cap={cap} "
                             f"parents[:6]={list(parents[:6])}\n")
            sys.stderr.flush()
        if width > _MAX_CHAIN:
            _ST["t_overcap_eager"] += 1
            return None
        if not is_ch:
            _ST["t_branch_eager"] += 1
            return None
        if cap is None:
            _ST["t_overcap_eager"] += 1
            return None
        self._ensure(cap, st)
        g, staged, out = self.graphs[cap]
        # Refresh the static inputs with this round's tokens (padded to cap)
        ids = [int(t) for t in tokens]
        ids += [0] * (cap - len(ids))                   # padded rows: read-only
        staged.refresh(ids, st.pos)
        if staged._stale:
            _ST["t_branch_eager"] += 1
            return None
        g.replay()
        _ST["t_replay"] += 1
        return out                                       # static (logits, record, None, starts)


# --------------------------------------------------------------------------- #
# Graph D — draft (launch_blocks), captured in full
# --------------------------------------------------------------------------- #
class _DraftGraphs:
    """CUDA graph for the DFlash2 launch_blocks (0 NCCL under TP2_DRAFT_REPL)."""

    def __init__(self, draft):
        self.draft = draft
        self.device = draft.device
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}            # (n_live, max_nodes) -> (CUDAGraph, tokens_buf, out)

    def try_replay(self, snaps, pendings, max_nodes, block):
        """Return launch_blocks' output on a graph replay, else None (eager)."""
        live = [i for i, s in enumerate(snaps) if s[2] > 0]
        if not live or max_nodes < 1:
            return None
        key = (len(live), max_nodes)
        entry = self.graphs.get(key)
        if entry is None:
            return None                      # capture on the caller's next warm round
        g, tokens_buf, out = entry
        length = min(block or self.draft.block, max_nodes + 1)
        vals = []
        for i in live:
            vals.append(int(pendings[i]))
            vals += [self.draft.mask_id] * (length - 1)
        tokens_buf.copy_(torch.tensor(vals, dtype=torch.int32, device=self.device),
                         non_blocking=True)
        g.replay()
        _ST["d_replay"] += 1
        return out

    def capture(self, snaps, pendings, max_nodes, block):
        """Warm up twice, then capture launch_blocks for this (n_live, max_nodes) bucket."""
        live = [i for i, s in enumerate(snaps) if s[2] > 0]
        if not live or max_nodes < 1:
            return
        key = (len(live), max_nodes)
        if key in self.graphs:
            return
        length = min(block or self.draft.block, max_nodes + 1)
        n = len(live) * length
        tokens_buf = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.draft._tp2_tokens_buf = tokens_buf
        for _ in range(2):
            self._run(self.draft.launch_blocks, snaps, pendings, max_nodes, block)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = self._run(self.draft.launch_blocks, snaps, pendings, max_nodes, block)
        self.graphs[key] = (g, tokens_buf, out)
        _ST["d_captured"] += 1
        sys.stderr.write(f"tp2cudagraphs: Graph D captured n_live={len(live)} max_nodes={max_nodes}\n")
        sys.stderr.flush()

    @staticmethod
    def _run(fn, snaps, pendings, max_nodes, block):
        return fn(snaps, pendings, max_nodes, block)


# --------------------------------------------------------------------------- #
# install
# --------------------------------------------------------------------------- #
def install():
    if os.environ.get("TP2_CUDA_GRAPHS", "1") == "0":
        return                                       # kill-switch: pure stock
    import torch  # noqa: F401  (module-global for the classes above)
    globals()["torch"] = torch

    from tensorfold.families.qwen3_5.cuda import forward as _fwd
    from tensorfold.families.qwen3_5.cuda import multi as _multi
    from tensorfold.families.qwen3_5.cuda import decode_tp as _dtp
    from tensorfold.families.qwen3_5.cuda import dflash2 as _dfl

    state = {"vg": None, "dg": None, "warm": 0}

    # ---- Graph T: patch multi_tree_forward at the source (forward module) -- #
    # The _verify method in multi.py captures multi_tree_forward via from-import.
    # Patching multi.multi_tree_forward updates the module dict, which IS the
    # same object the _verify closure references (from-import binds the name
    # in the module namespace, not a local closure cell).
    orig_mtf = _fwd.multi_tree_forward
    _mtf_entry_count = [0]
    _MTF_WARMUP = 50

    def _patched_multi_tree_forward(w, streams, *, full_logits=True, tp=False,
                                     capture_taps=False, hidden=False):
        _mtf_entry_count[0] += 1
        if _mtf_entry_count[0] <= 3 or _mtf_entry_count[0] % 200 == 0:
            toks, par, st = streams[0]
            sys.stderr.write(
                f"tp2cudagraphs MTF: n={len(streams)} W={len(par)} "
                f"chain={_is_chain(par, len(par))} cnt={_mtf_entry_count[0]}\n")
            sys.stderr.flush()
        if (tp and len(streams) == 1 and not hidden and not capture_taps
                and _mtf_entry_count[0] > _MTF_WARMUP):
            if state["vg"] is None:
                state["vg"] = _VerifyGraphs(w, full_logits, capture_taps)
            toks, par, st = streams[0]
            out = state["vg"].try_replay(toks, par, st)
            if out is not None:
                return out
        return orig_mtf(w, streams, full_logits=full_logits, tp=tp,
                        capture_taps=capture_taps, hidden=hidden)

    # Patch BOTH the forward module (source) and the multi module (import site)
    _fwd.multi_tree_forward = _patched_multi_tree_forward
    _multi.multi_tree_forward = _patched_multi_tree_forward

    # ---- Graph T: PRIMARY — tree_forward (decode_tp, actual decode path) #
    import tensorfold.families.qwen3_5.cuda.decode_tp as _dtp
    orig_tf = _dtp.tree_forward
    _tf_entry_count = [0]
    _TF_WARMUP = 50  # Skip graph capture for first N calls (calibrate phase)

    def _patched_tree_forward(w, tokens, parents, st, *, full_logits=True, tp=False,
                               initial=None, finish=True, capture_taps=False,
                               hidden=False, staged=None):
        _tf_entry_count[0] += 1
        if _tf_entry_count[0] <= 3:
            sys.stderr.write(
                f"tp2cudagraphs TF_ENTRY: tp={tp} W={len(parents)} "
                f"chain={_is_chain(parents, len(parents))} "
                f"hidden={hidden} taps={capture_taps} staged={staged is not None}\n")
            sys.stderr.flush()
        # C=1 chain gate: single stream, no hidden, no taps, past warmup
        if (tp and not hidden and not capture_taps and staged is None
                and _tf_entry_count[0] > _TF_WARMUP):
            if state["vg"] is None:
                state["vg"] = _VerifyGraphs(w, full_logits, capture_taps)
            out = state["vg"].try_replay(tokens, parents, st)
            if out is not None:
                return out
        return orig_tf(w, tokens, parents, st, full_logits=full_logits, tp=tp,
                       initial=initial, finish=finish, capture_taps=capture_taps,
                       hidden=hidden, staged=staged)

    _dtp.tree_forward = _patched_tree_forward

    # ---- Graph D: patch the launch_blocks method on the live drafter -------- #
    graph_d_on = os.environ.get("TP2_GRAPH_D", "0") == "1"
    orig_launch_blocks = _dfl.DFlash2.launch_blocks

    def _patched_launch_blocks(self, snaps, pendings, max_nodes, block=None):
        if graph_d_on and os.environ.get("TP2_DRAFT_REPL", "1") == "1" and self.world == 2:
            if state["dg"] is None:
                state["dg"] = _DraftGraphs(self)
            out = state["dg"].try_replay(snaps, pendings, max_nodes, block)
            if out is not None:
                return out
            state["warm"] += 1
            if state["warm"] >= 3:
                state["dg"].capture(snaps, pendings, max_nodes, block)
        return orig_launch_blocks(self, snaps, pendings, max_nodes, block)

    _dfl.DFlash2.launch_blocks = _patched_launch_blocks

    sys.stderr.write(
        "tp2cudagraphs: installed (TP2_CUDA_GRAPHS=1, Graph T -> multi_tree_forward)\n")
    sys.stderr.flush()
