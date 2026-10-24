#!/bin/bash
# Queue parts (run_est.sh) on pool cards through the shared R2S2V_mug/pool_retry.sh (user's card cap, counts every cfix_r2s2r step).
# Gives up after 8 h without its marker (POOL_GIVEUP_S).   bash launch_est.sh est_U8_r10 est_U8_r8b est_f2_r10 ...
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; E=$R/experiments/finetune/2026-10-23/estimate_release_videos; cd $R/experiments/R2S2V_mug
for p in "$@"; do
  POOL_GIVEUP_S=28800 setsid nohup bash pool_retry.sh cfix_r2s2r_exp_$p $R/outputs/R2S2V_mug/done_exp_$p 44G bash $E/code/run_est.sh $p > $R/logs/r2s2v/retry_exp_$p.log 2>&1 < /dev/null &
  sleep 3
done
echo queued "$@"
