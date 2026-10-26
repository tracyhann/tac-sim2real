#!/bin/bash
# The held-out GT test of the touch-data comparison (README section 11; user 2026-10-26: "测试集用我们的 gt recording heldout"):
# one trained arm's finger targets for the two recorded episodes (dataset_gt from code/build_gt.py: the REAL head-camera clip, the
# recording's measured hand state), 10 noise seeds each -> <run>/pred_gt/s<k>/gt_<ep>.npz. The physics test then re-runs each
# recording with these targets (scene_mug datagen/policy_gt.sh).
#   R2S2R=<root> bash predict_gt.sh T-obs|T-cmd|V-cmd [K]     K: re-plan every K frames (predict_cmp --every) -> <run>/pred_gt_r<K>
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=${R2S2R:?set R2S2R}; X=${1:?run}; PY=$R/envs/vtla/bin/python; H=$R/outputs/R2S2V_mug/hf25
E=${2:-}; P=pred_gt${E:+_r$E}; O=${VTLA_OUT:-$R/outputs/vtla/cmp}/$X; [ -e $O/final ] || { echo "no $O/final (train the run first)"; exit 2; }
export N0VTLA_DATA_HOME=$R/cache/n0vtla HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HERE/code; echo "$X: gt predictions $(date +%T)"
$PY predict_cmp.py --ckpt $O/final --data $H/dataset_gt --ids gt_4397541c gt_55494490 --seeds 0 1 2 3 4 5 6 7 8 9 --out $O/$P --cache $R/cache/vtla_frames ${E:+--every $E}
echo "run $X gt predictions done $(date +%T)"
