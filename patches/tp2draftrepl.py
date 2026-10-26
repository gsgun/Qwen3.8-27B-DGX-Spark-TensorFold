"""Patch 0008-A (TensorFold v0.6.0): replicated rank-local DFlash2 — zero draft NCCL.

Audit (2026-10-24, read-only): the qwen3_5 factory (families/qwen3_5/__init__.py:325)
hard-codes tp_draft=tp==2, so at TP=2 BOTH ranks hold a sharded half of the DFlash2
drafter and draft together:
    - DFlash2._row (dflash2.py:476-483)   10 gather_rank_partials/round
      (o_proj + mlp.down_proj across the 5 drafter layers)
    - launch_blocks (dflash2.py:536-544)   2 all_gather_into_tensor/round
      (candidate values + global ids, [rows, 16])
    => 12 drafter collectives/round on the RoCE link (measured 149.7 all_gather
       kernels/round = 128 target + 12 draft + 2 split-sampling + NCCL drift).

Every DFlash2 code path is already world==1-native:
    - __init__ (185-298): world==1 => no q/k/v/o/gate/up/down split,
      heads_local=32, kv_local=8, sub_head a zero-copy row view of the target's
      tiled head (matmul_rows at launch)
    - _row (474-483):           world==1 => local matmul
    - _layer_fast (447-483):    local given _row is local
    - launch_blocks (489-534):  the candidate-gather pair is gated on world==2;
      world==1 runs topk(k=16) locally over the full 98592-wide candidate logits
    - add_taps / _commit:       local projections, no world gating
The merged global top-16 of candidates is mathematically identical to the
sharded merge (each rank's half top-16 covers its half of the global top-16;
the union contains the global top-16, and topk picks the same 16).

TP2_DRAFT_REPL=1 (default): force tp_draft=False at engine construction =>
the stock TensorFold TP2 shape: the drafter exists ONLY on rank 0 as a
world=1 full replica; rank 1 mirrors rank 0's windows (multi.py:98
self.drafts is False there, so it never runs the drafter). Zero NCCL
collectives in the draft pass. Rank 0 pays +1.8 GiB (full drafter vs the
sharded half); rank 1 frees 1.8 GiB.
TP2_DRAFT_REPL=0: no-op => stock sharded joint drafter (boot-10 behavior).

Why safe: rank-0-only world=1 drafter is TensorFold's default TP2 path
(engine.py:154-158 else branch); the joint sharded drafter is the opt-in
deviation the family factory made. No dflash2.py / multi.py / engine.py line
is edited — only the tp_draft kwarg is intercepted at Qwen27Engine.__init__.

Phase-2 synergy: launch_blocks (Graph D) captured under this mode contains
zero NCCL collectives => a pure local single-GPU CUDA graph capture.
"""

import os
import sys


def install():
    if os.environ.get("TP2_DRAFT_REPL", "1") != "1":
        sys.stderr.write("tp2draftrepl: off (TP2_DRAFT_REPL=0) — stock sharded joint DFlash2\n")
        sys.stderr.flush()
        return

    from tensorfold.families.qwen3_5.cuda import engine as _engine

    orig_init = _engine.Qwen27Engine.__init__

    def _counted_init(self, *args, **kwargs):
        if kwargs.get("tp_draft"):
            kwargs["tp_draft"] = False
            sys.stderr.write(
                f"tp2draftrepl: rank {kwargs.get('rank', '?')} tp_draft True->False — "
                f"rank-0 world=1 replicated DFlash2, 0 draft NCCL collectives\n")
            sys.stderr.flush()
        return orig_init(self, *args, **kwargs)

    _engine.Qwen27Engine.__init__ = _counted_init
