#!/bin/bash
# start one pool_retry loop per part (card cap enforced by pool_retry.sh): bash launch_ge.sh prep U_high ...
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug; mkdir -p $R/logs/lora_ge
for p in "$@"; do
  M=$R/outputs/R2S2V_mug/done_exp_lora_ge_$p
  [ -f $M ] && { echo "$p already done"; continue; }
  pgrep -u $USER -f "[p]ool_retry.sh cfix_r2s2r_ge_$p " > /dev/null && { echo "$p loop already running"; continue; }
  setsid nohup bash $X/pool_retry.sh cfix_r2s2r_ge_$p $M 44G bash $X/lora_ge/run_ge.sh $p > $R/logs/lora_ge/loop_$p.log 2>&1 < /dev/null &
  echo "started $p loop $!"
done
