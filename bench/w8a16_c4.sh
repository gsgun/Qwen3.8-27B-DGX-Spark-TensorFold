#!/bin/bash
# C=N x 512-token bench. Usage: w8a16_c4.sh <label> <draft:true|false> [N]
L=$1; D=$2; N=${3:-4}
P1="Explain how a hash table works, including collision resolution strategies such as chaining and open addressing."
P2="Write a Python function that implements a binary search tree with insert and in-order traversal methods, and include a demo."
REQ=/tmp/w8a16_req_${L}
rm -f ${REQ}_*.json /tmp/w8a16_${L}_c${N}_*.json
i=0
while [ $i -lt $N ]; do
  case "${4:-m}" in
    p) CP=$P1;; c) CP=$P2;; *) if [ $((i % 2)) -eq 0 ]; then CP=$P1; else CP=$P2; fi;;
  esac
  python3 -c "
import json, sys
json.dump({'model':'Qwen3.8-27B','messages':[{'role':'user','content':sys.argv[1]}],
           'max_tokens':512,'temperature':0,
           'chat_template_kwargs':{'enable_thinking':False},'draft': sys.argv[2]=='true'},
          open(sys.argv[3], 'w'))" "$CP" "$D" "${REQ}_$i.json"
  i=$((i+1))
done
echo "req draft flags: $(python3 -c "
import json
print([json.load(open('${REQ}_%d.json' % i)).get('draft') for i in range($N)])")"
t0=$(date +%s%3N)
i=0
while [ $i -lt $N ]; do
  curl -s --max-time 900 http://127.0.0.1:8894/v1/chat/completions --data @${REQ}_$i.json > /tmp/w8a16_${L}_c${N}_$i.json &
  i=$((i+1))
done
wait
t1=$(date +%s%3N)
python3 - "$L" "$N" "$t0" "$t1" "$REQ" <<'EOF'
import json, sys
L, N, t0, t1, REQ = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
tot, per, tf = 0, [], {}
for i in range(N):
    j = json.load(open(f"/tmp/w8a16_{L}_c{N}_{i}.json"))
    tf = j.get("tensorfold", {}) or tf
    u = j.get("usage", {})
    n = u.get("completion_tokens", 0)
    tot += n; per.append(n)
wall = t1 - t0
print(f"{L} c{N} tokens={tot} per-stream={per} wall_s={wall/1000:.2f} agg_tok_s={tot/(wall/1000):.2f}")
for k in ("rounds", "accepted", "decode_s", "drafts", "ms_per_round", "draft_accept_rate"):
    if k in tf:
        print(f"  {k}={tf[k]}")
EOF
echo "$L done"
