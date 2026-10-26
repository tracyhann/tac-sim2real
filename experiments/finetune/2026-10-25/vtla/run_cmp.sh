#!/bin/bash
# One arm of the touch-data comparison (README section 11) on one GPU: train (no touch input, joint stage only = the pi0.5 policy on
# hand actions; resumable every 500 steps), then predict the test rows' finger targets for the physics test.
#   R2S2R=<root> bash run_cmp.sh T-obs|T-cmd|V-cmd [steps=4000]
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=${R2S2R:?set R2S2R}; X=${1:?run}; N=${2:-4000}; PY=$R/envs/vtla/bin/python; H=$R/outputs/R2S2V_mug/hf25
case $X in
  T-obs) DS=$H/dataset_cmp_t; ARM=obs ;;
  T-cmd) DS=$H/dataset_cmp_t; ARM=cmd ;;
  V-cmd) DS=$H/dataset_cmp_v; ARM=cmd ;;
  *) echo "unknown run $X"; exit 2 ;;
esac
O=${VTLA_OUT:-$R/outputs/vtla/cmp}/$X; T=$H/dataset_cmp_t/test_ids.txt
export N0VTLA_DATA_HOME=$R/cache/n0vtla HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=$R/models/n0-vtla-base; L=${SLURM_TMPDIR:-${TMPDIR:-}}
if [ -n "$L" ] && [ -d "$L" ]; then mkdir -p $L/n0base && cp --update=none $BASE/model.safetensors $L/n0base/ && BASE=$L/n0base; fi
cd $HERE/code; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; echo "$X: data $DS arm $ARM steps $N out $O $(date +%T)"
VAL=""; [ $DS = $H/dataset_cmp_t ] && VAL="--val_ids $(cat $T)"                          # the test rows never train
$PY train_vtla.py --arm $ARM --no_touch --data $DS --base $BASE --out $O --stages joint:$N --batch 32 --workers 8 --cache $R/cache/vtla_frames \
    --val_every 0 --ckpt_every 500 --save last --log_every 50 $VAL
$PY predict_cmp.py --ckpt $O/final --data $H/dataset_cmp_t --ids $(cat $T) --out $O/pred --cache $R/cache/vtla_frames
echo "run $X done $(date +%T)"
