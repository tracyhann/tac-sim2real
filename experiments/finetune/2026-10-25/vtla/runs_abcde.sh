#!/bin/bash
# The five comparison runs (README section 10), one per call, on one GPU:
#   R2S2R=<root> bash runs_abcde.sh A|B|C|D|E        (after setup.sh, the dataset with vtla/ labels, and code/make_split.py)
#   A  VTS, touch                 pred:1000, align:1000, joint:6000
#   B  VTF, touch                 same
#   C  VTS, no touch              align:1000, joint:6000 (no tactile target, so no pred stage)
#   D  VTF, no touch              same as C
#   E  VTF, touch, no pred stage  align:1000, joint:6000 (the paper's free-latent variant: z shaped by the action loss only)
# Held-out rows = <dataset>/vtla/heldout.txt; batch 32; validation every 500 steps; <out>/best = best joint-stage checkpoint.
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=${R2S2R:?set R2S2R}; X=${1:?run name A-E}; PY=$R/envs/vtla/bin/python
DS=${VTLA_DATASET:-$R/outputs/R2S2V_mug/hf25/dataset}; O=${VTLA_OUT:-$R/outputs/vtla/runs}/$X; BATCH=${BATCH:-32}
export N0VTLA_DATA_HOME=$R/cache/n0vtla HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $X in
  A) ARGS="--arm vts --stages pred:1000,align:1000,joint:6000" ;;
  B) ARGS="--arm vtf --stages pred:1000,align:1000,joint:6000" ;;
  C) ARGS="--arm vts --no_touch --stages align:1000,joint:6000" ;;
  D) ARGS="--arm vtf --no_touch --stages align:1000,joint:6000" ;;
  E) ARGS="--arm vtf --stages align:1000,joint:6000" ;;
  *) echo "unknown run $X"; exit 2 ;;
esac
[ -f $DS/vtla/heldout.txt ] || $PY $HERE/code/make_split.py --data $DS
BASE=$R/models/n0-vtla-base; L=${SLURM_TMPDIR:-${TMPDIR:-}}
if [ -n "$L" ] && [ -d "$L" ]; then mkdir -p $L/n0base && cp --update=none $BASE/model.safetensors $L/n0base/ && BASE=$L/n0base; fi
cd $HERE/code
$PY train_vtla.py $ARGS --data $DS --base $BASE --out $O --batch $BATCH --workers 8 --cache $R/cache/vtla_frames --val_ids $(cat $DS/vtla/heldout.txt) --save last
for ck in final best; do $PY eval_vtla.py --ckpt $O/$ck --data $DS --stride 8 --cache $R/cache/vtla_frames --out $O/$ck/eval.json; done
echo "run $X done $(date +%T)"
