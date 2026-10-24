#!/bin/bash
# CDS: train both experts' LoRAs (500 steps each, one pool card each) and queue the step-500 eval, which starts by itself once
# both checkpoint-500 files exist. Shared code: ../../_common/code (train.sh, build_hf25.py, predict_hf25.py, score_hf25.py).
# Run on the Vision login node AFTER build_hf25.py has written outputs/R2S2V_mug/hf25/dataset/metadata_CDS.json.
bash /scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/experiments/finetune/2026-10-24/_common/code/launch.sh CDS_high CDS_low ev_CDS_500
