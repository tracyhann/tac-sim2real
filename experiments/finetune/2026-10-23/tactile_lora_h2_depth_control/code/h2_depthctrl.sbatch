#!/bin/bash
# User 10-23: H2 (step 600) generating video + touch from the REAL clip's Depth Anything V2 depth as control (no sim),
# real frame 0 + real frame-0 touch as start, held-out clips 55494490 / 4397541c, seeds 0 1. Tag ...tacth2c600da-real.
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug; PY=$R/envs/videoxfun/bin/python
export HF_HOME=$R/cache/huggingface TOKENIZERS_PARALLELISM=false; unset HF_TOKEN
cd $X && echo "host $(hostname) $(date +%T)" && $PY lora_tactile/predict_tactile_da.py --run h2 --ds dataset_ho --meta metadata_test_da.json --ckpt 600 \
  --clips tsmug-55494490 tsmug-4397541c --sources gt --seeds 0 1 && echo "H2DA_DONE $(date +%T)"
