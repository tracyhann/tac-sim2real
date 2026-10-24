#!/bin/bash
# stop the tagged e1 LoRA run (user 2026-10-22: train mixed, no real/sim prompt tag); checkpoints are kept
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r
for pid in $(pgrep -u $USER -f "lora_tactile/after_train.sh"); do [ "$pid" != "$$" ] && kill -TERM $pid && echo "stopped after_train $pid"; done
for p in e1_high e1_low; do
  pr=$(pgrep -u $USER -f "pool_run.py --name cfix_r2s2r_lora_$p"); [ -z "$pr" ] && { echo "$p: no pool_run"; continue; }
  for sr in $(pgrep -P $pr); do kill -TERM $sr && echo "$p: stopped srun $sr (pool_run $pr)"; done
done
sleep 15
pgrep -u $USER -af "cfix_r2s2r_lora" | cut -c1-120 || echo "no lora processes left"
ls $R/outputs/R2S2V_mug/lora_tactile/e1_high/ $R/outputs/R2S2V_mug/lora_tactile/e1_low/ | grep -v comfyui | grep safetensors
