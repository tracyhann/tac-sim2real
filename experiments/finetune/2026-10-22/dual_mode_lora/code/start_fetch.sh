#!/bin/bash
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; mkdir -p $R/logs/lora_ge
cd $R/experiments/R2S2V_mug/lora_ge
HF_HOME=$R/cache/huggingface setsid nohup nice -n 10 $R/envs/armA/bin/python fetch_eps.py --workers 6 > $R/logs/lora_ge/fetch.log 2>&1 < /dev/null &
echo started $!
