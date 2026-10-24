#!/bin/bash
# Queue the held-out eval of an f-run once both experts saved checkpoint <ck> (like after_ck.sh, part <run>_hoev<ck>).
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; LT=$R/outputs/R2S2V_mug/lora_tactile; RUN=$1; CK=$2
ok() { local f=$LT/${RUN}_$1/checkpoint-$CK.safetensors; [ -f $f ] && [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 180 ]; }
until ok high && ok low; do sleep 120; done
echo "$(date +%T) $RUN checkpoint-$CK saved by both experts"; bash $R/experiments/R2S2V_mug/lora_tactile/launch_lora.sh ${RUN}_hoev$CK
