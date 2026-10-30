"""boot92: release kept/viewed-state storage + PAR*GROW admission headroom + gate/round telemetry.

Root cause (boot91 NoRoom on c6d's 6th stream):
  - kept(st)/viewed(st) build ``kv[:st.pos]`` VIEWS. A slice's untyped_storage is the
    WHOLE buffer it was sliced from, and prefill grows that buffer by DOUBLING
    (forward.grow: cap = max(need, 2*cap, 1024)). So a checkpoint pins the full
    doubled prefill slab (up to 2x the prefix), not just the rows below pos.
  - _trim() is a no-op on a pinned pool (self.pinned=True), and _room() reserves no
    headroom for the live streams' growth, so 32 slots of long prefixes pin the
    32 GiB pool and the 6th stream's _first(s) can't fit.

Fix:
  1. kept/viewed CLONE the rows below pos -> the checkpoint owns exactly pos rows;
     the stream's (doubled) prefill slab is freed when the stream grows/finishes.
  2. _room reserves PAR*GROW rows of admission headroom before admitting.
  3. Telemetry (kill TP2_KEPTFIX_LOG=0): every N rounds log gate used/pool/base,
     torch_allocated, cache entries, max entry slab, live streams, round ms, and the
     target-forward vs draft-launch vs commit split (p4d breakdown).
Disable all: TP2_KEPTFIX=0.
"""

import os as _os
import time as _time
import sys as _sys

_GROW = 8192
_GIB = 1024 ** 3


def _now() -> float:
    return _time.perf_counter()


class _Acc:
    __slots__ = ("n", "t_target", "t_draft", "t_commit")

    def __init__(self):
        self.n = 0
        self.t_target = 0.0
        self.t_draft = 0.0
        self.t_commit = 0.0


ACC = _Acc()


def install():
    if _os.environ.get("TP2_KEPTFIX", "1") == "0":
        return
    import torch

    from tensorfold.families.qwen3_5.cuda import multi
    from tensorfold.families.qwen3_5.cuda.decode import clone_state
    from tensorfold.cuda.memory_gate import NoRoom

    # ---- 1. kept/viewed: CLONE rows below pos (own storage) -------------------
    def kept(st):
        other = clone_state(st)
        other.kv = [None if kv is None else (kv[0][:st.pos].clone(), kv[1][:st.pos].clone()) for kv in st.kv]
        other.rec = [None if r is None else r.clone() for r in st.rec]
        other.conv = [None if c is None else c.clone() for c in st.conv]
        return other

    def viewed(st):
        st.kv = [None if kv is None else (kv[0][:st.pos].clone(), kv[1][:st.pos].clone()) for kv in st.kv]
        return st

    multi.kept = kept
    multi.viewed = viewed

    # ---- 2. _room: reserve PAR*GROW headroom ----------------------------------
    _HEAD_ROWS = int(_os.environ.get("TP2_HEADROOM_ROWS", "6") or 6)

    def _room(self, s):
        if any(x.waiting for x in self.streams.values()):
            raise NoRoom("streams already wait for memory; a new request waits until one finishes")
        headroom = _HEAD_ROWS * _GROW * self.row_bytes
        while not self.memory_gate.fits(self._first(s) * self.row_bytes + headroom):
            if not self.cache.evict():
                if not self.live():
                    return
                raise NoRoom(f"a {len(s.prompt)}-token prompt waits for memory until a live stream finishes")
            self._trim()

    multi.MultiDecoder._room = _room

    # ---- 3. telemetry: wrap target forward / draft launch / commit ------------
    if _os.environ.get("TP2_KEPTFIX_LOG", "1") != "0":
        _orig_mtf = multi.multi_tree_forward

        def _mtf_timed(*a, **k):
            t0 = _now()
            out = _orig_mtf(*a, **k)
            ACC.t_target += _now() - t0
            ACC.n += 1
            return out

        multi.multi_tree_forward = _mtf_timed

        _orig_commit = multi.commit_streams

        def _commit_timed(*a, **k):
            t0 = _now()
            out = _orig_commit(*a, **k)
            ACC.t_commit += _now() - t0
            return out

        multi.commit_streams = _commit_timed

        # draft launch_blocks: patch the drafter instance's bound method lazily
        def _patch_draft(self):
            if self.draft is not None and not getattr(self.draft, "_tp2kv_timed", False):
                try:
                    _lb = self.draft.launch_blocks

                    def _lb_timed(*a, **k):
                        t0 = _now()
                        out = _lb(*a, **k)
                        ACC.t_draft += _now() - t0
                        return out

                    self.draft.launch_blocks = _lb_timed
                    self.draft._tp2kv_timed = True
                except Exception:
                    pass

        _orig_admit = multi.MultiDecoder.admit

        def _admit(self, s):
            _patch_draft(self)
            return _orig_admit(self, s)

        multi.MultiDecoder.admit = _admit

        _orig_round = multi.MultiDecoder.round
        _cnt = [0]
        _LOG_N = int(_os.environ.get("TP2_KEPTFIX_LOG_N", "200") or 200)

        def _round(self):
            t0 = _now()
            r = _orig_round(self)
            dt = _now() - t0
            _cnt[0] += 1
            if (
                _LOG_N
                and _cnt[0] % _LOG_N == 0
                and _os.environ.get("TP2_KEPTFIX_LOG", "1") != "0"
            ):
                try:
                    mg = self.memory_gate
                    used = mg.used() if mg is not None else 0
                    pool = mg.room if mg is not None else 0
                    base = mg.base if mg is not None else 0
                    ta = torch.cuda.memory_allocated()
                    n_ent = len(self.cache.entries)
                    slab = 0
                    for _e in self.cache.entries:
                        for _kv in _e[1].kv:
                            if _kv is not None:
                                for _t in _kv:
                                    slab = max(slab, _t.untyped_storage().nbytes())
                    n = ACC.n or 1
                    _sys.stderr.write(
                        f"[tp2keptfix] r={_cnt[0]} used={used / _GIB:.2f} pool={pool / _GIB:.2f} "
                        f"base={base / _GIB:.2f} torch={ta / _GIB:.2f} entries={n_ent} "
                        f"slab={slab / 1024 ** 2:.1f}MiB streams={self.live()} round={dt * 1e3:.1f}ms "
                        f"target={ACC.t_target / n * 1e3:.1f} draft={ACC.t_draft / n * 1e3:.1f} "
                        f"commit={ACC.t_commit / n * 1e3:.1f}ms/round\n"
                    )
                    _sys.stderr.flush()
                except Exception as _e:  # telemetry must never break the round
                    _sys.stderr.write(f"[tp2keptfix] log err: {_e!r}\n")
            return r

        multi.MultiDecoder.round = _round

    _sys.stderr.write("tp2keptfix: installed (kept/viewed clone + headroom + telemetry)\n")
    _sys.stderr.flush()
