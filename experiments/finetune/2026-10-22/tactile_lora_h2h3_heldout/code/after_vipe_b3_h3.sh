#!/bin/bash
# h3_high / h3_low queue 3 min after ViPE b3 is done (h2_low queues first), then take cards as they free (4-card cap)
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r
until [ -f $R/outputs/R2S2V_mug/done_vipe_b3 ]; do sleep 60; done; sleep 180
echo "$(date +%T) queue h3"; bash $R/experiments/R2S2V_mug/lora_tactile/launch_lora.sh h3_high h3_low
