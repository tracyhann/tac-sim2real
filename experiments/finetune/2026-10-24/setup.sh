#!/bin/bash
# One-shot setup for the HF25 experiment on a fresh Linux machine with an NVIDIA GPU (see README.md).
#   R2S2R=/data/hf25 bash setup.sh [--no-weights] [--raw-release]
# Creates under $R2S2R:
#   third_party/VideoX-Fun            git clone, pinned commit, + the two trainer patches + our trainer copy (train_control_lora_hf25.py)
#   envs/videoxfun                    Python 3.12 venv with env/requirements.txt (torch 2.8 cu128, diffusers 0.35.1, peft 0.21.2, ...)
#   models/Wan2.2-Fun-A14B-Control    the backbone weights from Hugging Face (65 GB; --no-weights skips)
#   outputs/R2S2V_mug/hf25/dataset    the prepared training set from Hugging Face (65 MB)
#   data/mug-sim2video-pose-tactile   (--raw-release) the raw public release the training set was built from (42 MB; not needed to train)
# Idempotent: every step skips what already exists. Needs git, python3.12 (or uv), ~70 GB of disk, internet.
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); C=$HERE/_common/code
R=${R2S2R:-}; [ -n "$R" ] || { echo "usage: R2S2R=<root dir> bash setup.sh [--no-weights] [--raw-release]"; exit 2; }
WEIGHTS=1; RAW=0; for a in "$@"; do case $a in --no-weights) WEIGHTS=0 ;; --raw-release) RAW=1 ;; *) echo "unknown option $a"; exit 2 ;; esac; done
VXF_COMMIT=13bb3c625e8b1e457d7124d92207b49e00e4a6fe
WEIGHTS_REPO=alibaba-pai/Wan2.2-Fun-A14B-Control
DATASET_REPO=tracyhan816/mug-video-touch-lora-hf25
RAW_REPO=tracyhan816/mug-sim2video-pose-tactile; RAW_REV=6cf129c769161eec04b5566a903163913b75ca33
mkdir -p $R/third_party $R/envs $R/models $R/outputs/R2S2V_mug/hf25 $R/data $R/cache/huggingface; export HF_HOME=${HF_HOME:-$R/cache/huggingface}

echo "== 1/6 VideoX-Fun @ $VXF_COMMIT"
V=$R/third_party/VideoX-Fun
[ -d $V/.git ] || git clone -q https://github.com/aigc-apps/VideoX-Fun.git $V
git -C $V fetch -q origin $VXF_COMMIT 2>/dev/null || true; git -C $V checkout -q $VXF_COMMIT

echo "== 2/6 Python env $R/envs/videoxfun"
PY=$R/envs/videoxfun/bin/python
if [ ! -x $PY ]; then
  if command -v uv > /dev/null; then uv venv --python 3.12 $R/envs/videoxfun; else python3.12 -m venv $R/envs/videoxfun; fi
fi
$PY -m pip --version > /dev/null 2>&1 || $PY -m ensurepip --upgrade > /dev/null 2>&1 || uv pip install --python $PY pip   # uv venvs ship without pip
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -r $V/requirements.txt
$PY -m pip install -q -r $HERE/env/requirements.txt
$PY -c "import torch, diffusers, peft, accelerate, transformers, av, decord, cv2, imageio, tensorboard; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| diffusers', diffusers.__version__, 'peft', peft.__version__)"
HF=$R/envs/videoxfun/bin/hf

echo "== 3/6 backbone weights -> $R/models/Wan2.2-Fun-A14B-Control"
if [ $WEIGHTS = 1 ]; then
  [ -f $R/models/Wan2.2-Fun-A14B-Control/low_noise_model/diffusion_pytorch_model.safetensors ] && [ -f $R/models/Wan2.2-Fun-A14B-Control/high_noise_model/diffusion_pytorch_model.safetensors ] \
    || $HF download $WEIGHTS_REPO --local-dir $R/models/Wan2.2-Fun-A14B-Control
  for f in low_noise_model/diffusion_pytorch_model.safetensors high_noise_model/diffusion_pytorch_model.safetensors Wan2.1_VAE.pth models_t5_umt5-xxl-enc-bf16.pth google; do
    [ -e $R/models/Wan2.2-Fun-A14B-Control/$f ] || { echo "missing $f in the weights"; exit 1; }; done
else
  echo "skipped (--no-weights)"
fi

echo "== 4/6 prepared training set -> $R/outputs/R2S2V_mug/hf25/dataset"
DS=$R/outputs/R2S2V_mug/hf25/dataset
[ -f $DS/metadata_FHN.json ] || $HF download $DATASET_REPO --repo-type dataset --local-dir $DS
for f in metadata_FHN.json metadata_SVO.json metadata_CDS.json test_cases.json video tactile ctrl start test; do [ -e $DS/$f ] || { echo "dataset incomplete: $f"; exit 1; }; done
echo "$(ls $DS/video/*.mp4 | wc -l) videos, $(ls $DS/ctrl/*.mp4 | wc -l) control videos"
if [ $RAW = 1 ]; then
  [ -f $R/data/mug-sim2video-pose-tactile/metadata.csv ] || $HF download $RAW_REPO --repo-type dataset --revision $RAW_REV --local-dir $R/data/mug-sim2video-pose-tactile
fi

echo "== 5/6 trainer patches + our trainer copy"
$PY $C/videoxfun_patches/patch_trainer.py  $V/scripts/wan2.2_fun/train_control_lora.py
$PY $C/videoxfun_patches/patch_trainer2.py $V/scripts/wan2.2_fun/train_control_lora.py
$PY $C/make_trainer_hf25.py $V
$PY -m py_compile $V/scripts/wan2.2_fun/train_control_lora_hf25.py

echo "== 6/6 checks"
CUDA_VISIBLE_DEVICES= $PY $C/hf25_patch.py | tail -1
cd $V && $PY -c "import videox_fun, sys; sys.path.insert(0, 'scripts/wan2.2_fun'); import hf25_patch; print('videox_fun + hf25_patch import ok')"
cat <<EOF

Setup done. Next (one GPU each, any order; re-run a part until it exits 0):
  export R2S2R=$R
  bash $C/train.sh FHN_high   # bash $C/train.sh FHN_low   SVO_high  SVO_low  CDS_high  CDS_low
then, per arm:
  bash $C/train.sh ev_FHN_500    # generation + touch on the test cases, scores -> $R/outputs/R2S2V_mug/hf25/eval/table.md
EOF
