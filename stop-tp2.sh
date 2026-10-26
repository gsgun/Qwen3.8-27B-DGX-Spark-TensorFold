#!/bin/bash
# stop-tp2.sh (HEAD copy on spark1) — stop the Qwen3.8 TP=2 pair and free GPU memory.
# Also stops the worker rank1 via ssh (the worker's own staged copy stops locally only).
docker rm -f qwen38-tf-rank0 >/dev/null 2>&1 || true
timeout 15 ssh -o ConnectTimeout=6 -o BatchMode=yes gsgun@192.168.8.150 \
  "$HOME/qwen38-tf-cluster/stop-tp2.sh" >/dev/null 2>&1 || true
echo "qwen38 TF2 ranks removed ($(date +%T))"
