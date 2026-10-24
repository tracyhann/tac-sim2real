#!/bin/bash
# Held-out test of the video + tactile LoRA (2026-10-22): the zero-shot A14B generations under the star control (views5/head:
# episode-fused scene depth + shadowless sim hand / mug + coaster / mark props, real frame 0 as start + reference), tag
# wan22fun-a14b-head-c5 (the round-5 head videos; seed 0 of edf79415 / 53780879 already existed):
#   hog_old  seed 1 of the two old clips (their sim rows: seeds 0 / 1)
#   hog_<ep8> per clip, queued as each twin is installed: seeds 0 / 1 of the new training clip 9811b8f8 (its sim rows), seed 0
#            of the held-out clips 55494490 / 4397541c (= baseline (c), zero-shot video, and the held-out "generated frame 0" start)
#   hob_c    baseline (c) scored: evaluate / adherence / labeladh of head-c5 seed 0 on the held-out clips
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug; A=$X/../A_baseline; B_=$X/../B_sim2wm; PY=$R/envs/videoxfun/bin/python
export HF_HOME=$R/cache/huggingface TOKENIZERS_PARALLELISM=false; unset HF_TOKEN
cd $X; echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T) part $1"
vid() { $PY fun_zeroshot.py --model a14b --clips $1 --seeds $2 --control views5/head/fun_ctrl.mp4 --start views5/head/cond.png --ref views5/head/cond.png --tag wan22fun-a14b-head-c5; }
case $1 in
  hog_old) vid tsmug-edf79415 1; vid tsmug-53780879 1 ;;
  hog_55494490|hog_4397541c|hog_f8f7ed78) vid tsmug-${1#hog_} 0 ;;   # test clips (f8f7ed78: no twin, swapped for 4397541c)
  hog_9811b8f8) vid tsmug-${1#hog_} "0 1" ;;                        # training clip: its sim rows
  hob_c) HT="tsmug-55494490 tsmug-4397541c"; M="wan22fun-a14b-head-c5 sim-render"
         bash $X/run_r2s.sh $A/evaluate.py --clips $HT --models $M; bash $X/run_r2s.sh $B_/adherence.py --clips $HT --models $M; bash $X/run_r2s.sh labeladh.py --clips $HT --models $M ;;
esac
touch $R/outputs/R2S2V_mug/done_exp_ho_$1; echo "part $1 done $(date +%T)"
