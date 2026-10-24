#!/bin/bash
# restart the dispatcher with an optional priority list: bash restart_dispatch.sh "U_high ... "
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug/lora_ge
for pid in $(pgrep -u $USER -f "lora_ge/dispatch_ge[.]sh"); do kill $pid; pkill -u $USER -P $pid sleep; echo "stopped dispatcher $pid"; done
sleep 2; GE_PRIO="$1" setsid nohup bash $X/dispatch_ge.sh >> $R/logs/lora_ge/dispatch.log 2>&1 < /dev/null &
echo "dispatcher started $! prio: ${1:-default}"
