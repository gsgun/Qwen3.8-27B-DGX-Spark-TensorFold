"""Patch 0008-D2 (TensorFold v0.6.0): cap checkpoint-slot KV reservation.

Problem: admission sizes each ``--checkpoint-slots`` entry at the FULL
prompt/reply window. In stream_geometry (tensorfold/cuda/geometry.py):

    kv     = attention * capacity * hk * hd * 2 * kv_item     # per cache copy
    caches = (streams + keep + 1) * kv                        # one copy PER slot

At capacity 262144 (fp8, 4 kv-heads/2 = 2/rank, head_dim 256, 16 attn layers)
each slot is 4.072 GiB of rank-local attn KV plus ~150 MiB GDN state (the
``state`` term in ``fixed``, capacity-independent). So keep=32 needs
153.5 GiB of stream geometry vs the ~74.6 GiB rank budget: 0008-D (32 slots)
is infeasible at 4x262144 under the stock full-window reservation.

Fix: cap the per-slot KV reservation at TF_SLOT_KV_CAP tokens (a realistic
multi-turn boundary length) while keeping the GDN-state portion uncapped. The
wrapper is a pure composition of the stock geometry (no dim math):

    base  = stream_geometry(keep=0)
    full  = stream_geometry(keep=K)
    full(c) - base(c) = K * (GDN_state + kv(c))        # slot cost at window c
    bytes_at(c) = base(c) + min(full(c) - base(c), full(CAP) - base(CAP))

Capping the delta at its value for window=CAP keeps all K*GDN_state and only
clamps the KV part to kv(CAP). No bytes are added or dropped below CAP.

Measured (2026-10-24, probe_slotcap.py / probe_admit.py, rank-local fp8):
  per-slot @262144            = 4.072 GiB
  keep=32 stream geometry     = 153.54 GiB stock -> 33.54 GiB @ cap=16384
                               (32 capped slots cost LESS than boot-10's
                                3 full-window slots: 35.46 GiB)
  full admission keep=32 cap=16384, 0008-A mode:
    rank0 64.08 / 84.44 GiB OK, rank1 62.85 / 74.59 GiB OK
    (boot-10 sharded baseline was 64.62 GiB on both ranks)

Kill-switch: TF_SLOT_KV_CAP=0 (default) -> no-op, stock reservation.
No-ops: keep<=0, pinned-pool mode (pool>0, TP=1 only).
Only the admission estimate is touched; runtime KV allocation is unchanged,
so a boundary longer than CAP simply draws its true KV from the (larger)
headroom rather than being pre-reserved at the full window.
"""

import os
import sys


def install():
    try:
        cap = int(os.environ.get("TF_SLOT_KV_CAP", "0") or 0)
    except ValueError:
        cap = 0
    if cap <= 0:
        sys.stderr.write(
            "tp2slotcap: off (TF_SLOT_KV_CAP=0) - stock full-window slot reservation\n")
        sys.stderr.flush()
        return

    from tensorfold.cuda import geometry as _geom

    orig_sg = _geom.stream_geometry

    def _capped(text, world, streams, keep, first=None, kv_item=2, pool=0):
        g = orig_sg(text, world, streams, keep, first=first, kv_item=kv_item, pool=pool)
        if keep <= 0 or pool:
            return g
        base = orig_sg(text, world, streams, 0, first=first, kv_item=kv_item, pool=pool)
        b_at, bb_at = g.bytes_at, base.bytes_at
        cap_cost = b_at(cap) - bb_at(cap)

        def bytes_at(capacity):
            cost = b_at(capacity) - bb_at(capacity)
            return bb_at(capacity) + min(cost, cap_cost)

        return _geom.Geometry(bytes_at, 1)

    _geom.stream_geometry = _capped
    sys.stderr.write(
        f"tp2slotcap: slot KV reservation capped at {cap} tokens "
        f"(stock: full window)\n")
    sys.stderr.flush()
