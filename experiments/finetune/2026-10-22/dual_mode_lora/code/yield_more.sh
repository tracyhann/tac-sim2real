#!/bin/bash
# pause one more ge part ($1) for f3_low and re-queue it when yield_f3.sh finishes (f3_low advancing)
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug/lora_ge; P=$1
bash $X/stop_ge.sh $P
setsid nohup bash -c "until grep -q ' done$' $R/logs/lora_ge/yield_f3.log; do sleep 60; done; bash $X/launch_ge.sh $P; echo \"\$(date +%T) re-queued $P\"" >> $R/logs/lora_ge/yield_f3.log 2>&1 < /dev/null &
echo "paused $P; re-queue waiter $!"
