# Gen-1 `3490ac1f` Rescue — forensic record (Oct 26)

## Outcome
Canonical Gen-1 binary `3490ac1f` (Oct 25 ~17:00-19:30 build) is LOST (scratch .so
overwritten by later rebuilds; only md5 known). The Gen-1 SOURCE was recovered
authentic from two independent artifacts:

1. `qmm_kernel_gen1_recovered.cu` (md5 a6d6411640cebfad7affa312d054eca8)
   - The file copied verbatim from the live scratch .cu at 17:31 PDT by the
     recipe-repo sync commit a6d64116 ("Add W8A16 CUDA extension") — i.e. the
     exact 16:50-era Gen-1 source: initial write (state.db msg 72891) +
     scale/bias direct 2-byte loads (72911) + W8 swizzle removal (72917) +
     X swizzle removal (72950) + B-frag k0 fix (72956) + manual A-frag
     (72999). Dequant kernel NOT yet added at this point.
2. `qmm_kernel_gen1_initial.cu` (md5 45826cd0, 13015 bytes) — the untouched
   initial write, replayed from state.db msg 72891 tool_call.
3. `w8a16_cuda_qmm.so.gen1-2f84` (md5 2f841d3de90fe399985d62312c571eee, 11,283,560 bytes) — surviving
   17:53 PDT rebuild of the same source (canonical 3490ac1f was the 17:00-19:30
   build of this source; 2f843753 = same source + nvcc rebuild, pre-dequant).

## Verification
- `prod_drain.cu` (current deployed source, b8440d18) minus the drain fix
  (L134 cp_wait<0> -> cp_wait<STAGES-2>) minus the dequant block == repo
  a6d64116 .cu. Confirmed by diff: only the drain line + dequant differ.
- Replay of state.db execute_code chain (72891->72911->72917->72950) reproduces
  d48f6597; the repo file adds the two final manual A-frag edits (72956/72999)
  whose old-strings no longer matched in sandbox replay (indentation).

## state.db anchors
session local-fe6d52e46c059906b3ec82c332b29ef6c108b335874b5096f1f6fe6febc5cc0b
72891 initial .cu | 72911 scale fix | 72917 W8 swizzle rm | 72950 X swizzle rm
72956 B-frag k0 fix | 72999 manual A-frag | 73289 repo sync (a6d64116)
73743 dequant add | 73753 cpp host wrapper | 73915 namespace restructure
