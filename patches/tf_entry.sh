#!/bin/bash
# Startup wrapper: make engine.py startup_copies env-tunable (TF2_STARTUP_COPIES),
# then exec the real tensorfold serve command. Keeps the image pristine; boot79
# containers are unaffected (they already booted with the original engine.py).
ENGINE=/usr/local/lib/python3.12/dist-packages/tensorfold/families/qwen3_5/cuda/engine.py
if [ -f "$ENGINE" ] && grep -q 'startup_copies=int(tp == 2)' "$ENGINE"; then
  sed -i 's/startup_copies=int(tp == 2)/startup_copies=int(os.environ.get("TF2_STARTUP_COPIES", "1" if tp == 2 else "0"))/' "$ENGINE"
fi
# ensure `os` is imported in engine.py, inserted AFTER the __future__ block (line 3)
if ! grep -qE '^import os' "$ENGINE"; then
  sed -i '3a import os' "$ENGINE"
fi
exec "$@"
