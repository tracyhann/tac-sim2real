#!/bin/bash
# h2_low waits until 4397541c ViPE (b3) is done, so the uncontended twin is not stuck behind training at the 4-card cap
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r
until [ -f $R/outputs/R2S2V_mug/done_vipe_b3 ]; do sleep 60; done
echo "$(date +%T) ViPE b3 done"; bash $R/experiments/R2S2V_mug/lora_tactile/launch_lora.sh h2_low
