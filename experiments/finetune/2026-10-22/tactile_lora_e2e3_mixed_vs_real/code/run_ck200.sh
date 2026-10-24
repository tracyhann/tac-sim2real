#!/bin/bash
# Early eval of the video + tactile LoRA at step 200 (user 2026-10-22: "evaluate e2 and e3 at the step-200 checkpoint now,
# on one card"). Same as lora_tactile/run_lora.sh <run>_eval, with ckpt 200 for both runs. e3 also takes its prompt /
# control from metadata_mixed.json: identical text and control, and metadata_real.json lists only the real-start rows,
# so both runs are scored on the same two starts (real frame 0 + real touch; generated frame 0 + sim touch).
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug; A=$X/../A_baseline; B_=$X/../B_sim2wm; PY=$R/envs/videoxfun/bin/python
export HF_HOME=$R/cache/huggingface TOKENIZERS_PARALLELISM=false; unset HF_TOKEN
cd $X; echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T) part $1"
C="tsmug-edf79415 tsmug-53780879"
for run in e2 e3; do $PY lora_tactile/predict_tactile.py --run $run --meta metadata_mixed.json --ckpt 200 --clips $C --sources gt gen_s0 --seeds 0; done
M="wan22fun-a14b-tacte2c200-real wan22fun-a14b-tacte2c200-sim wan22fun-a14b-tacte3c200-real wan22fun-a14b-tacte3c200-sim"
bash $X/run_r2s.sh $A/evaluate.py --clips $C --models $M; bash $X/run_r2s.sh $B_/adherence.py --clips $C --models $M; bash $X/run_r2s.sh labeladh.py --clips $C --models $M
touch $R/outputs/R2S2V_mug/done_exp_$1; echo "part $1 done $(date +%T)"
