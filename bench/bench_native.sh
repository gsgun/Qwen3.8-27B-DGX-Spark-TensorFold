#!/bin/bash
L=$1
P1="Explain how a hash table works, including collision resolution strategies such as chaining and open addressing."
P2="Write a Python function that implements a binary search tree with insert and in-order traversal methods, and include a demo."
RDIR=/tmp/bench_${L}_req
ODIR=/tmp/bench_${L}_out
mkdir -p $RDIR $ODIR
rm -f $RDIR/*.json $ODIR/*.json

run_c() {
  local tag=$1 draft=$2 N=$3 mode=$4
  local i=0
  while [ $i -lt $N ]; do
    case "$mode" in
      p) CP=$P1;; c) CP=$P2;; m) if [ $((i % 2)) -eq 0 ]; then CP=$P1; else CP=$P2; fi;;
    esac
    python3 -c "
import json, sys
json.dump({'model':'Qwen3.8-27B','messages':[{'role':'user','content':sys.argv[1]}],
           'max_tokens':200,'temperature':0,
           'chat_template_kwargs':{'enable_thinking':False},'draft': sys.argv[2]=='true'},
          open(sys.argv[3], 'w'))" "$CP" "$draft" "$RDIR/${tag}_$i.json"
    i=$((i+1))
  done
  local t0=$(date +%s%3N)
  i=0
  while [ $i -lt $N ]; do
    curl -s --max-time 900 http://127.0.0.1:8894/v1/chat/completions --data @$RDIR/${tag}_$i.json > $ODIR/${tag}_$i.json &
    i=$((i+1))
  done
  wait
  local t1=$(date +%s%3N)
  python3 -c "
import json, sys
L, tag, N, t0, t1, ODIR = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
tot, per = 0, []
for i in range(N):
    j = json.load(open(ODIR + '/' + tag + '_' + str(i) + '.json'))
    u = j.get('usage', {})
    n = u.get('completion_tokens', 0)
    tot += n; per.append(n)
wall = (t1 - t0) / 1000.0
print(f'{tag} n={N} tokens={tot} wall={wall:.2f}s agg={tot/wall:.2f}tok/s per={per}')
" "$L" "$tag" "$N" "$t0" "$t1" "$ODIR"
}

run_prefill() {
  local tag=$1 tok=$2
  python3 -c "
import json, sys
tok = int(sys.argv[1])
chars = tok * 4
text = 'The quick brown fox jumps over the lazy dog. ' * (chars // 45 + 1)
text = text[:chars]
json.dump({'model':'Qwen3.8-27B','messages':[{'role':'user','content':text}],
           'max_tokens':1,'temperature':0,
           'chat_template_kwargs':{'enable_thinking':False},'draft':True},
          open(sys.argv[2], 'w'))" "$tok" "$RDIR/${tag}.json"
  local t0=$(date +%s%3N)
  curl -s --max-time 300 http://127.0.0.1:8894/v1/chat/completions --data @$RDIR/${tag}.json > $ODIR/${tag}.json
  local t1=$(date +%s%3N)
  python3 -c "
import json, sys
L, tag, tok, t0, t1, ODIR = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
j = json.load(open(ODIR + '/' + tag + '.json'))
u = j.get('usage', {})
pt = u.get('prompt_tokens', 0)
wall = (t1 - t0) / 1000.0
print(f'prefill {tag}: prompt_tokens={pt} ttft={wall*1000:.0f}ms prefill_tok_s={pt/wall:.1f}')
" "$L" "$tag" "$tok" "$t0" "$t1" "$ODIR"
}

echo "=== $L Benchmark Start $(date) ==="
run_c p1d true 1 p
run_c c1d true 1 c
run_c p4d true 4 p
run_c c4d true 4 c
run_c m4d true 4 m
run_c c4off false 4 c
run_prefill p8k 8192
run_prefill p16k 16384
run_prefill p32k 32768
run_prefill p64k 65536
echo "=== $L Benchmark End $(date) ==="
