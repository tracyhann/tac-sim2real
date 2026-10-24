#!/bin/bash
# Pull the e2 / e3 / f2 / f3 LoRA eval outputs (generated canvases + decoded touch) from Vision into
# runs/clips/r2s2v_lora/pull_out/, then rebuild the wrong-vs-fixed boards (r2s2v/lora_fix_grid.py).
#   bash r2s2v/lora_fix_pull.sh
set -e; cd "$(dirname "$0")/.."
mkdir -p runs/clips/r2s2v_lora/pull_out
ssh linux3090 'ssh -O check vision >/dev/null 2>&1 && ssh -o BatchMode=yes vision "cd /scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/outputs/R2S2V_mug/lora_tactile/out && tar cf - \$(ls -d wan22fun-a14b-tact[ef][23]c* 2>/dev/null)"' \
  | tar xf - -C runs/clips/r2s2v_lora/pull_out
.venv/bin/python r2s2v/lora_fix_grid.py 2>&1 | grep -v objc
