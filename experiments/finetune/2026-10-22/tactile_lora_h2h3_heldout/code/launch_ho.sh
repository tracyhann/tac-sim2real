#!/bin/bash
# Queue held-out parts of run_ho.sh on pool cards (4-card cap via pool_retry.sh), detached.   bash launch_ho.sh hog_old
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; cd $R/experiments/R2S2V_mug
for p in "$@"; do
  setsid nohup bash pool_retry.sh cfix_r2s2r_ho_$p $R/outputs/R2S2V_mug/done_exp_ho_$p 44G bash $R/experiments/R2S2V_mug/run_ho.sh $p > $R/logs/r2s2v/retry_ho_$p.log 2>&1 < /dev/null &
  sleep 3
done
echo queued "$@"
