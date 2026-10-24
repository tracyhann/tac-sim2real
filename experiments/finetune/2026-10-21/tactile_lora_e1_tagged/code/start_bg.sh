#!/bin/bash
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r
setsid nohup bash $R/experiments/R2S2V_mug/lora_tactile/after_train.sh ${1:-e2} > $R/logs/r2s2v/after_lora_train_${1:-e2}.log 2>&1 < /dev/null &
echo started
