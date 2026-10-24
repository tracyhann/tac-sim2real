#!/bin/bash
# Queue the tiny-model trainer smoke test (smoke_tiny.sh) on one pool card through the shared pool_retry.sh; ~10 min.
# Marker outputs/R2S2V_mug/done_exp_hf25_smoke; log logs/hf25/loop_smoke.log + logs/r2s2v/pool_cfix_r2s2r_hf25_smoke.log.
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug; C=$R/experiments/finetune/2026-10-24/_common/code; mkdir -p $R/logs/hf25
M=$R/outputs/R2S2V_mug/done_exp_hf25_smoke; rm -f $M
pgrep -u $USER -f "[p]ool_retry.sh cfix_r2s2r_hf25_smoke " > /dev/null && { echo "smoke loop already running"; exit 0; }
POOL_GIVEUP_S=10800 setsid nohup bash $X/pool_retry.sh cfix_r2s2r_hf25_smoke $M 44G bash $C/smoke_tiny.sh > $R/logs/hf25/loop_smoke.log 2>&1 < /dev/null &
echo "started smoke loop $!"
