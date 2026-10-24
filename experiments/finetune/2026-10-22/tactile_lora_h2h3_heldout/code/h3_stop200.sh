#!/bin/bash
# h3_high was launched with 600 steps; h3 now ends at the equal-passes point (step 200, run_lora.sh STEPS). Once checkpoint-200 is
# saved (3 min old), mark h3_high done and stop its pool step (its pool_retry loop then exits on the marker).
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; LT=$R/outputs/R2S2V_mug/lora_tactile; f=$LT/h3_high/checkpoint-200.safetensors
until [ -f $f ] && [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 180 ]; do sleep 60; done
touch $R/outputs/R2S2V_mug/done_exp_lora_h3_high; echo "$(date +%T) h3_high checkpoint-200 saved: marked done"
bash $R/experiments/R2S2V_mug/lora_tactile/stop_lora.sh h3_high
