#!/bin/bash
# Queue LoRA parts on pool cards (4-card cap via pool_retry.sh), detached.   bash launch_lora.sh e1_high e1_low
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; cd $R/experiments/R2S2V_mug
for p in "$@"; do
  setsid nohup bash pool_retry.sh cfix_r2s2r_lora_$p $R/outputs/R2S2V_mug/done_exp_lora_$p 44G bash $R/experiments/R2S2V_mug/lora_tactile/run_lora.sh $p > $R/logs/r2s2v/retry_lora_$p.log 2>&1 < /dev/null &
  sleep 3
done
echo queued "$@"
