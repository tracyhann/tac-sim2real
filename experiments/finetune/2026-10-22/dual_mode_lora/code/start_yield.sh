#!/bin/bash
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r
setsid nohup bash $R/experiments/R2S2V_mug/lora_ge/yield_f3.sh >> $R/logs/lora_ge/yield_f3.log 2>&1 < /dev/null &
echo "yield started $!"
