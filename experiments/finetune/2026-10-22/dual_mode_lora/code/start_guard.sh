#!/bin/bash
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug/lora_ge
pgrep -u $USER -f "lora_ge/cap_guard_ge[.]sh" > /dev/null && { echo "guard already running"; exit 0; }
setsid nohup bash $X/cap_guard_ge.sh >> $R/logs/lora_ge/cap_guard.log 2>&1 < /dev/null &
echo "guard started $!"
