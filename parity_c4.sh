#!/bin/bash
# C=4 byte-exact parity: 4 concurrent identical streams per prompt, each must
# hash to the boot-10 reference. p=prose d5589b366369, c=code 39191bf72982.
P1="Explain how a hash table works, including collision resolution strategies such as chaining and open addressing."
P2="Write a Python function that implements a binary search tree with insert and in-order traversal methods, and include a demo."
mk() { python3 -c "
import json,sys
json.dump({'model':'Qwen3.8-27B','messages':[{'role':'user','content':sys.argv[1]}],
 'max_tokens':512,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},'draft':True},
 open(sys.argv[2],'w'))" "$1" "$2"; }
for label in p c; do
  if [ "$label" = p ]; then P="$P1"; else P="$P2"; fi
  mk "$P" /tmp/c4_${label}.json
  for i in 1 2 3 4; do curl -s --max-time 300 http://127.0.0.1:8894/v1/chat/completions --data @/tmp/c4_${label}.json > /tmp/c4_${label}_${i}.json & done
  wait
done
python3 -c "
import json,hashlib
for label,ref in [('p','d5589b366369'),('c','39191bf72982')]:
    shas=[]
    for i in range(1,5):
        c=json.load(open('/tmp/c4_%s_%d.json'%(label,i)))['choices'][0]['message']['content']
        shas.append(hashlib.sha256(c.encode()).hexdigest()[:12])
    ok=all(s==ref for s in shas)
    print('C=4 %s: %s  ref=%s  %s'%(label,' '.join(shas),ref,'PASS' if ok else 'MISMATCH'))
"
