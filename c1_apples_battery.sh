#!/bin/bash
# Apples-to-apples boot89/90 prompts (arg4=c / arg4=p) on live boot91. Zero restart.
cd ~/qwen38-tf-cluster || exit 1
LOG=/tmp/c1_apples.log
exec >"$LOG" 2>&1
echo "=== apples-to-apples (arg4 c/p) $(date) ==="
run(){ L=$1; D=$2; N=$3; T=$4
  echo "[leg] $L draft=$D N=$N type=$T"
  ./bench/w8a16_c4.sh "$L" "$D" "$N" "$T"
  echo "clock: $(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader)"
  sleep 5
}
run c1d true 1 c
run p1d true 1 p
run c4d true 4 c
run p4d true 4 p
run c6d true 6 c
echo "=== APPLES PARSE ==="
python3 - <<'PY'
import json,re
log=open("/tmp/c1_apples.log").read()
for L,N in [("c1d",1),("p1d",1),("c4d",4),("p4d",4),("c6d",6)]:
    m=re.search(rf"{L} c{N} tokens=\d+ .*agg_tok_s=(\d+\.\d+)",log)
    agg=float(m.group(1)) if m else None
    dec=[];rounds=[];dr=[];ac=[]
    for i in range(N):
        try:
            j=json.load(open(f"/tmp/w8a16_{L}_c{N}_{i}.json"));tf=j.get("tensorfold",{})
            dec.append(tf.get("decode_s",0));rounds.append(tf.get("rounds",0));dr.append(tf.get("drafted",0));ac.append(tf.get("accepted",0))
        except: pass
    if rounds:
        msr=1000*sum(dec)/max(sum(rounds),1)
        print(f"{L}: agg={agg} tok/s | rounds={rounds} drafted={sum(dr)} accepted={sum(ac)} accept={100*sum(ac)/max(sum(dr),1):.1f}% ms/round={msr:.1f} per-stream dec={[round(x,1) for x in dec]}")
PY
echo APPLES_DONE
