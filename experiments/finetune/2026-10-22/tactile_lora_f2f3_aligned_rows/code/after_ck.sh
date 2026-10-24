#!/bin/bash
# Queue an eval part of the held-out runs once both experts saved checkpoint <ck> (no card is held while waiting; the file
# must be 3 min old so a half-written safetensors is never read).   bash after_ck.sh h3 200   -> launches part h3_ev200
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; LT=$R/outputs/R2S2V_mug/lora_tactile; RUN=$1; CK=$2
ok() { local f=$LT/${RUN}_$1/checkpoint-$CK.safetensors; [ -f $f ] && [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 180 ]; }
until ok high && ok low; do sleep 120; done
echo "$(date +%T) $RUN checkpoint-$CK saved by both experts"; bash $R/experiments/R2S2V_mug/lora_tactile/launch_lora.sh ${RUN}_ev$CK
