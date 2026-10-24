#!/bin/bash
# stop running LoRA training steps (their pool_retry loops relaunch them after 120 s with the current run_lora.sh)
#   bash stop_lora.sh e2_high e2_low ...
for p in "$@"; do
  pr=$(pgrep -u $USER -f "pool_run.py --name cfix_r2s2r_lora_$p"); [ -z "$pr" ] && { echo "$p: no pool_run"; continue; }
  for sr in $(pgrep -P $pr); do kill -TERM $sr && echo "$p: stopped srun $sr (pool_run $pr)"; done
done
