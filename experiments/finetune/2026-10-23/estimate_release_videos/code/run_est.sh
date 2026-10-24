#!/bin/bash
# One pool step: predict_release.py for one model x group. Part names est_<model>_<group>: U8 | f2  x  r10 | r8b | r6; est_R6 = regr_release.py (R6 regressor, 3 seeds, all 25).
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; E=$R/experiments/finetune/2026-10-23/estimate_release_videos
export HF_HOME=$R/cache/huggingface TOKENIZERS_PARALLELISM=false; unset HF_TOKEN
cd $E/code; echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T) part $1"
p=${1#est_}; model=${p%_*}; group=${p#*_}
if [ "$1" = est_R6 ]; then $R/envs/videoxfun/bin/python regr_release.py && touch $R/outputs/R2S2V_mug/done_exp_$1 && echo "part $1 done $(date +%T)"; exit; fi
$R/envs/videoxfun/bin/python predict_release.py --model $model --group $group && touch $R/outputs/R2S2V_mug/done_exp_$1 && echo "part $1 done $(date +%T)"
