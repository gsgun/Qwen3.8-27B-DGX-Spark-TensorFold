#!/usr/bin/env python3
"""TP2 decode-loop verify-width + draft-row distribution probe (0008-B2 Step 1).
Read-only; gated by TP2_WIDTH_PROBE=1. Appends JSONL (batched) to
TP2_WIDTH_PROBE_OUT (default /probe_out/width_probe.jsonl, host-mounted).

ACTIVE TP2 PATH (verified): engine.step -> decode_tp (per rank/stream) -> per round:
  draft:  draft.propose_tree -> launch_block -> DFlash2.launch_blocks  (Graph D)
  verify: tree_forward on window = [pending] + draft_nodes (W = len(window) <= max_rows=16)
decode_tp (rank 0) returns DecodeResult(..., widths) where widths = per-round len(window).
So wrapping decode_tp and reading result.widths gives the EXACT verify width per round.
The engine re-imports decode_tp at call time (`from .decode_tp import decode_tp` inside
the step functions), so patching the module attr dtp.decode_tp is picked up.

Records (one JSONL line per round, per stream):
  {"fn":"tf","W":<verify_rows>,"t":...}      per decode round (verify width)
  {"fn":"lb","max_nodes":<int>,"dW":<draft_rows>,"t":...}  per launch_blocks call (draft)
"""
import os, sys, time, json, threading

_lock = threading.RLock()
_buf = []
_flush_n = 8
_out = os.environ.get("TP2_WIDTH_PROBE_OUT", "/probe_out/width_probe.jsonl")
_seen = {"tf": 0, "lb": 0, "ft": 0}
_dbg = {"n": 0}


def _drain():
    with _lock:
        if not _buf:
            return
        lines = [json.dumps(r) for r in _buf]
        del _buf[:]
    try:
        with open(_out, "a") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"tp2widthprobe: drain write error {e}", file=sys.stderr)


def _emit(rec):
    with _lock:
        _buf.append(rec)
        n = len(_buf)
    key = rec.get("fn")
    if key in _seen:
        _seen[key] += 1
        if _seen[key] <= 16:
            print(f"tp2widthprobe: {key} #{_seen[key]} {rec}", file=sys.stderr)
        elif _seen[key] % 1024 == 0:
            print(f"tp2widthprobe: tf={_seen['tf']} lb={_seen['lb']} out={_out}",
                  file=sys.stderr)
    if n >= _flush_n:
        _drain()


def install():
    try:
        from tensorfold.families.qwen3_5.cuda import (
            decode_tp as dtp, dflash2,
        )
        # ---- verify width: wrap decode_tp, read result.widths (per-round W) ----
        orig_decode_tp = dtp.decode_tp

        def dtp_probe(*a, **k):
            result = orig_decode_tp(*a, **k)
            try:
                widths = getattr(result, "widths", None)
                with _lock:
                    dn = _dbg["n"]; _dbg["n"] += 1
                if dn <= 4:
                    print(f"tp2widthprobe: dtpHIT #{dn} type={type(result).__name__} "
                          f"widths_len={len(widths) if widths is not None else 'None'} "
                          f"widths_head={list(widths[:6]) if widths else None}", file=sys.stderr)
                if widths:
                    t = int(time.time() * 1000)
                    for W in widths:
                        _emit({"fn": "tf", "W": int(W), "t": t})
            except Exception as _e:
                print(f"tp2widthprobe: dtpHIT EXC {_e!r}", file=sys.stderr)
            return result
        dtp.decode_tp = dtp_probe

        # ---- draft rows: wrap DFlash2.launch_blocks ----
        DFlash2 = dflash2.DFlash2
        orig_lb = DFlash2.launch_blocks

        def lb_probe(self, *a, **k):
            with _lock:
                n = _dbg["n"]; _dbg["n"] += 1
            if n == 1:
                print(f"tp2widthprobe: DEBUG decode_tp patched? "
                      f"{dtp.decode_tp is dtp_probe}", file=sys.stderr)
            try:
                mn = a[2] if len(a) > 2 else k.get("max_nodes")
                _emit({"fn": "lb", "max_nodes": int(mn) if mn is not None else -1,
                       "dW": -1, "t": int(time.time() * 1000)})
            except Exception:
                pass
            out = orig_lb(self, *a, **k)
            try:
                dw = 0
                for tup in out:
                    if isinstance(tup, (tuple, list)) and len(tup) >= 3:
                        dw += int(tup[2])
                with _lock:
                    if _buf and _buf[-1].get("fn") == "lb" and _buf[-1]["dW"] < 0:
                        _buf[-1]["dW"] = dw
            except Exception:
                pass
            return out
        DFlash2.launch_blocks = lb_probe

        # ---- tree structure: wrap DFlash2.finish_tree -> log parents (chain?) ----
        orig_ft = DFlash2.finish_tree
        _ft_n = {"n": 0}

        def ft_probe(self, *a, **k):
            out = orig_ft(self, *a, **k)
            try:
                _ft_n["n"] += 1
                if _ft_n["n"] <= 200:
                    tokens, parents, scores = out[0], out[1], out[2]
                    is_chain = list(parents) == list(range(-1, len(parents) - 1))
                    _emit({"fn": "ft", "n": len(parents), "chain": int(is_chain),
                           "parents": list(parents)[:24], "t": int(time.time() * 1000)})
            except Exception as _e:
                print(f"tp2widthprobe: ftHIT EXC {_e!r}", file=sys.stderr)
            return out
        DFlash2.finish_tree = ft_probe

        print(f"tp2widthprobe: installed -> {_out} "
              f"decode_tp_patched={dtp.decode_tp is dtp_probe} "
              f"lb_patched={DFlash2.launch_blocks is lb_probe} "
              f"ft_patched={DFlash2.finish_tree is ft_probe}", file=sys.stderr)
        # engine-style import check (simulates `from .decode_tp import decode_tp`)
        try:
            from tensorfold.families.qwen3_5.cuda.decode_tp import decode_tp as _eng_check
            print(f"tp2widthprobe: engine-style import is dtp_probe? "
                  f"{_eng_check is dtp_probe} id={id(_eng_check)}", file=sys.stderr)
        except Exception as _e:
            print(f"tp2widthprobe: engine-style import check failed {_e!r}", file=sys.stderr)
    except Exception as e:
        import traceback
        print(f"tp2widthprobe: install failed (optional, continuing): {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
