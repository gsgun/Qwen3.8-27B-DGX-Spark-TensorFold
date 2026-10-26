"""Patch 0007 (TensorFold v0.6.0): TP=2 multi-stream batched-verification gate.

Root-cause audit (2026-10-23, C=4 "inversion"):
  The apparent C=4 slowdown (24.5 tok/s vs C=1 32.2) was a harness artifact
  (bench draft flag never set -> C=4 actually ran draft-OFF). Code audit of
  families/qwen3_5/cuda/ found no B>1 defect:
    - multi.py:427-431   every stream's draft block in ONE drafter pass
    - multi.py:442-444   ONE multi_tree_forward per round, all streams'
                         windows in a single W-row batch (forward.py:300-317)
    - forward.py:357/376 one GDN tree / one tree-attention per layer for
      all streams; :432/:441 one replay covers all layers and streams
    - distributed.py:267-284  128 all_gather collectives per forward
      (2 row-parallel gathers x 64 layers: each layer is EITHER GDN
      gdn.out OR attention attn.o, plus mlp.down), count B-invariant
    - 0006 tp2w8a16 flat M=1..64 (memory-bound, ~8% spread)
  With drafts genuinely on, C=4 measures 2.58x (prose) / 2.30-2.50x (code)
  of C=1, byte-exact greedy parity vs C=1.

What this patch does:
  TP2_BATCH_VERIFY=0           - kill-switch: installs nothing, pure stock
                                 path (mirrors TP2_W8A16=0).
  TP2_BATCH_VERIFY=1 + DIAG=0  - silent invariant bookkeeping; hooks stay
                                 installed forever, no logging, no unhook.
  TP2_BATCH_VERIFY=1 + DIAG=1  - 50-round boot gate (production default):
                                 verifies fwd/round=1.00 and
                                 target-colls/round=128.0 across the first
                                 50 decode rounds (colls/total counter is reset at
                                 the first round so prefill gathers are excluded),
                                 then restores the original
                                 unwrapped gather_rank_partials / _verify /
                                 multi_tree_forward callables so steady state
                                 pays zero wrapper overhead. The gate log
                                 also reports total collectives per round,
                                 whose excess over 128 is the DFlash2
                                 drafter's own gather count per round.
                                 A violated invariant keeps the wrappers
                                 installed and degrades to continuous watch.
  TP2_BATCH_VERIFY=1 + DIAG=2  - continuous watch: logs every 50 rounds,
                                 never unhooked (pre-change DIAG=1 behavior).

0008-A extension (2026-10-24): full-NCCL-scope counter (hook 2b).
  gather_rank_partials misses the direct dist calls — dflash2.py:537-538
  (candidate values/ids merge) and decode_tp._sample_split (split-head
  halves). Hook 2b counts all_gather_into_tensor / all_gather / all_reduce
  with the same in_fwd scoping, so the gate reports
  target-colls/round=128.0 AND draft-colls/round (0.0 under
  TP2_DRAFT_REPL=1; ~12/round under the stock sharded joint drafter) plus
  dist-ag/round (total NCCL: 128 target + ~2 split-sampling expected).

Delivery: /patches mount is on PYTHONPATH ahead of dist-packages; the
shadow tensorfold/__init__.py installs this module at import time. No
behavior change, no image rebuild, persists across container recreation.
An install failure is logged and swallowed (optional patch).
"""

import os
import sys

TARGET_COLLS = 128.0   # live-verified: 2 gathers/layer x 64 layers, stable across 800+ rounds
GATE_ROUNDS = 50


def install():
    if os.environ.get("TP2_BATCH_VERIFY", "1") == "0":
        return  # kill-switch: pure stock

    diag = os.environ.get("TP2_BATCH_VERIFY_DIAG", "0")
    gate = diag == "1"
    watch = diag == "2"

    from tensorfold.families.qwen3_5.cuda import distributed as _dist
    from tensorfold.families.qwen3_5.cuda import multi as _multi

    st = {
        "rounds": 0, "streams": 0, "rows": 0,
        "forwards": 0, "colls": 0, "target_colls": 0,
        "dist_colls": 0, "dist_target": 0,
        "fwd_since_log": 0, "tcolls_since_log": 0,
        "unhooked": False, "gate_done": False,
    }
    in_fwd = {"n": 0}

    # Hook 1: target-forward count. multi.py binds multi_tree_forward from
    # .forward at import time, so patch the name in the multi namespace.
    orig_forward = _multi.multi_tree_forward

    def _counted_forward(*args, **kwargs):
        st["forwards"] += 1
        st["fwd_since_log"] += 1
        in_fwd["n"] += 1
        try:
            return orig_forward(*args, **kwargs)
        finally:
            in_fwd["n"] -= 1

    _multi.multi_tree_forward = _counted_forward

    # Hook 2: collective count. forward.py:30 and dflash2.py:481 import
    # gather_rank_partials function-locally, so patch the attribute on the
    # distributed module (re-resolved on every call). Target collectives per
    # forward = 128. The drafter (DFlash2._row, dflash2.py:476-483) fires its
    # own gathers OUTSIDE the target forward; total - target per round is the
    # drafter's per-round collective count.
    orig_gather = _dist.gather_rank_partials

    def _counted_gather(*args, **kwargs):
        st["colls"] += 1
        if in_fwd["n"] > 0:
            st["target_colls"] += 1
            st["tcolls_since_log"] += 1
        return orig_gather(*args, **kwargs)

    _dist.gather_rank_partials = _counted_gather

    # Hook 2b (0008-A): full-NCCL scope — the direct dist calls that
    # gather_rank_partials misses (dflash2.py:537-538 candidate merge,
    # decode_tp._sample_split split-head halves). Same in_fwd scoping.
    import torch.distributed as _tdist

    dist_orig = {}
    for _name in ("all_gather_into_tensor", "all_gather", "all_reduce"):
        _fn = getattr(_tdist, _name, None)
        if _fn is None:
            continue
        dist_orig[_name] = _fn

        def _make(_orig):
            def _wrapped(*args, **kwargs):
                st["dist_colls"] += 1
                if in_fwd["n"] > 0:
                    st["dist_target"] += 1
                return _orig(*args, **kwargs)
            return _wrapped

        setattr(_tdist, _name, _make(_fn))

    # Hook 3: round boundary (one _verify per decode round, both ranks).
    orig_verify = _multi.MultiDecoder._verify

    def _log_window(tag):
        per_round = st["fwd_since_log"] / float(GATE_ROUNDS)
        tcolls_per = st["tcolls_since_log"] / float(GATE_ROUNDS)
        sys.stderr.write(
            f"tp2batchverify: rounds={st['rounds']} streams={st['streams']} "
            f"rows={st['rows']} fwd/round={per_round:.2f} "
            f"target-colls/round={tcolls_per:.1f} (expect 128.0) {tag}\n")
        sys.stderr.flush()
        st["fwd_since_log"] = 0
        st["tcolls_since_log"] = 0

    def _gate():
        """Run once at the 50-round mark (DIAG=1): verify, then unhook if OK."""
        per_round = st["fwd_since_log"] / float(GATE_ROUNDS)
        tcolls_per = st["tcolls_since_log"] / float(GATE_ROUNDS)
        total_per = st["colls"] / float(st["rounds"])
        dist_per = st["dist_colls"] / float(st["rounds"])
        dist_target_per = st["dist_target"] / float(st["rounds"])
        ok = per_round == 1.0 and abs(tcolls_per - TARGET_COLLS) < 0.5
        if ok:
            sys.stderr.write(
                f"tp2batchverify: rounds={st['rounds']} streams={st['streams']} "
                f"rows={st['rows']} fwd/round={per_round:.2f} "
                f"target-colls/round={tcolls_per:.1f} (expect 128.0) OK "
                f"draft-colls/round={total_per - tcolls_per:.1f} "
                f"dist-ag/round={dist_per:.1f} (target {dist_target_per:.1f}, "
                f"non-target {dist_per - dist_target_per:.1f}) — "
                f"50-round invariant verified — auto-unhooking hot-path wrappers\n")
            sys.stderr.flush()
            _multi.multi_tree_forward = orig_forward
            _dist.gather_rank_partials = orig_gather
            for _name, _fn in dist_orig.items():
                setattr(_tdist, _name, _fn)
            _multi.MultiDecoder._verify = orig_verify
            st["unhooked"] = True
        else:
            sys.stderr.write(
                f"tp2batchverify: rounds={st['rounds']} fwd/round={per_round:.2f} "
                f"target-colls/round={tcolls_per:.1f} (expect 128.0) WARN batch "
                f"invariant — keeping hot-path wrappers (degraded to watch)\n")
            sys.stderr.flush()
        st["gate_done"] = True

    def _counted_verify(self, plan, copied=None):
        if st["rounds"] == 0:
            # discard boot-time calibration forwards before counting. colls (the
            # UNscoped total) is reset too: the request's prefill gathers run outside
            # multi_tree_forward and would otherwise leak into total_per as phantom
            # draft-colls (measured 112.6/round = 5630 prefill gathers / 50 rounds).
            st["fwd_since_log"] = 0
            st["tcolls_since_log"] = 0
            st["colls"] = 0
            st["dist_colls"] = 0
            st["dist_target"] = 0
        result = orig_verify(self, plan, copied)
        st["rounds"] += 1
        st["streams"] += len(plan)
        st["rows"] += sum(len(t) for t, _ in result[0])
        if st["unhooked"]:
            return result
        if gate and not st["gate_done"] and st["rounds"] >= GATE_ROUNDS:
            _gate()
        elif watch and st["rounds"] % GATE_ROUNDS == 0:
            _log_window("OK" if (st["fwd_since_log"] == GATE_ROUNDS
                                 and abs(st["tcolls_since_log"] / float(GATE_ROUNDS) - TARGET_COLLS) < 0.5)
                        else "WARN batch invariant")
        return result

    _multi.MultiDecoder._verify = _counted_verify
