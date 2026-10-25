#!/bin/bash
# One-shot setup for the VTLA experiment (N0-VTLA policy on the glove dataset) on a Linux machine with an NVIDIA GPU.
#   R2S2R=/data/r2s2r bash setup.sh [--no-weights] [--no-data]
# Creates under $R2S2R:
#   third_party/N0-VTLA            git clone of neoteai/N0-VTLA at a pinned commit (the model code we build on; not copied here)
#   envs/vtla                      Python 3.11 venv with N0-VTLA's pinned requirements (torch 2.7.1, transformers 4.53.2 + the
#                                  repo's transformers_replace patch) + N0-VTLA installed in editable mode
#   models/n0-vtla-base            the pretrained N0-VTLA base checkpoint from Hugging Face (8.3 GB; --no-weights skips)
#   cache/n0vtla                   the PaliGemma sentencepiece tokenizer (public GCS bucket, fetched once)
#   outputs/R2S2V_mug/hf25/dataset the prepared glove dataset from Hugging Face (--no-data skips)
# Idempotent: every step skips what already exists. Needs git, internet, ~35 GB of disk, uv or Python 3.11.
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=${R2S2R:-}; [ -n "$R" ] || { echo "usage: R2S2R=<root dir> bash setup.sh [--no-weights] [--no-data]"; exit 2; }
WEIGHTS=1; DATA=1; for a in "$@"; do case $a in --no-weights) WEIGHTS=0 ;; --no-data) DATA=0 ;; *) echo "unknown option $a"; exit 2 ;; esac; done
N0_COMMIT=03a0ce4
BASE_REPO=NeoteAI/n0-vtla-base
DATASET_REPO=tracyhan816/mug-video-touch-lora-hf25
mkdir -p $R/third_party $R/envs $R/models $R/outputs/R2S2V_mug/hf25 $R/cache/huggingface $R/cache/pip $R/cache/tmp $R/cache/n0vtla
export HF_HOME=${HF_HOME:-$R/cache/huggingface} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$R/cache/pip} TMPDIR=${TMPDIR:-$R/cache/tmp} N0VTLA_DATA_HOME=$R/cache/n0vtla

echo "== 1/5 N0-VTLA @ $N0_COMMIT"
N0=$R/third_party/N0-VTLA
[ -d $N0/.git ] || git clone -q https://github.com/neoteai/N0-VTLA.git $N0
git -C $N0 fetch -q origin 2>/dev/null || true; git -C $N0 checkout -q $N0_COMMIT

echo "== 2/5 Python env $R/envs/vtla"
PY=$R/envs/vtla/bin/python
if [ ! -x $PY ]; then
  if ! command -v uv > /dev/null && ! command -v python3.11 > /dev/null; then
    python3 -m pip install -q --user uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH
  fi
  if command -v uv > /dev/null; then uv venv -q --python 3.11 $R/envs/vtla; else python3.11 -m venv $R/envs/vtla; fi
fi
$PY -m pip --version > /dev/null 2>&1 || $PY -m ensurepip --upgrade > /dev/null 2>&1 || uv pip install -q --python $PY pip   # uv venvs ship without pip
SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
if [ ! -f $SP/transformers/models/siglip/check.py ]; then
  $PY -m pip install -q --upgrade pip
  $PY -m pip install -q -r $N0/requirements.txt
  $PY -m pip install -q -e $N0 --no-deps
  cp -r $N0/n0vtla/models_pytorch/transformers_replace/* $SP/transformers/       # the repo's patched Gemma / PaliGemma / SigLIP
fi
$PY -m pip install -q -r $HERE/env/requirements.txt                               # our few extras (no-op when present)
$PY -c "
import torch, transformers, safetensors, av, sentencepiece
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly(), 'transformers_replace patch missing'
import n0vtla.models_pytorch.n0vtla_policy
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| n0vtla OK')"
HF=$R/envs/vtla/bin/huggingface-cli; [ -x $R/envs/vtla/bin/hf ] && HF="$R/envs/vtla/bin/hf"

echo "== 3/5 base checkpoint -> $R/models/n0-vtla-base"
if [ $WEIGHTS = 1 ]; then
  [ -f $R/models/n0-vtla-base/model.safetensors ] || $HF download $BASE_REPO --local-dir $R/models/n0-vtla-base
  ls -la $R/models/n0-vtla-base/model.safetensors
else
  echo "skipped (--no-weights)"
fi

echo "== 4/5 PaliGemma tokenizer -> $N0VTLA_DATA_HOME"
$PY -c "from n0vtla.models.tokenizer import PaligemmaTokenizer; t = PaligemmaTokenizer(200); print('tokenizer OK', len(t.tokenize('pick up the mug')[0]))"

echo "== 5/5 glove dataset -> $R/outputs/R2S2V_mug/hf25/dataset"
DS=$R/outputs/R2S2V_mug/hf25/dataset
if [ $DATA = 1 ]; then
  [ -f $DS/metadata_vtla.json ] || $HF download $DATASET_REPO --repo-type dataset --local-dir $DS
  ls $DS/video | wc -l | xargs echo "videos:"
  [ -d $DS/vtla ] && ls $DS/vtla | wc -l | xargs echo "vtla label files:" || echo "NOTE: $DS/vtla/ (per-frame commanded joints + finger forces for the VTF arm) is not in this copy; see README section 3"
else
  echo "skipped (--no-data)"
fi
cd $HERE/code && $PY test_vtla.py && echo "setup done"
