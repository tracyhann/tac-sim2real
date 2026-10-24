#!/bin/bash
# Queue <run>_eval only once both experts of the run finished training (no card is held while waiting).   bash after_train.sh e2
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; RUN=${1:-e2}
until [ -f $R/outputs/R2S2V_mug/done_exp_lora_${RUN}_high ] && [ -f $R/outputs/R2S2V_mug/done_exp_lora_${RUN}_low ]; do sleep 120; done
echo "$(date +%T) both experts trained"; bash $R/experiments/R2S2V_mug/lora_tactile/launch_lora.sh ${RUN}_eval
