#!/bin/bash
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug/lora_ge
bash $X/stop_chain.sh
pgrep -u $USER -f "lora_ge/dispatch_ge[.]sh" > /dev/null && { echo "dispatcher already running"; exit 0; }
setsid nohup bash $X/dispatch_ge.sh >> $R/logs/lora_ge/dispatch.log 2>&1 < /dev/null &
echo "dispatcher started $!"
