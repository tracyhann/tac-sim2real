#!/bin/bash
# Start HF25 parts on pool cards through R2S2V_mug/pool_retry.sh (the shared card cap in R2S2V_mug/pool_cap), detached:
#   bash launch.sh FHN_high FHN_low            training parts: one retry loop each, until done_exp_hf25_<part> exists
#   bash launch.sh ev_FHN_500                  eval part: a waiter that starts its retry loop once BOTH experts have checkpoint-500
# Loops: cfix_r2s2r_hf25_<part>; logs logs/hf25/{loop,wait}_<part>.log, pool step log logs/r2s2v/pool_cfix_r2s2r_hf25_<part>.log.
# ORIGINAL-CLUSTER ONLY (our GPU pool). Elsewhere run train.sh <part> directly (2026-10-24/README.md).
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; export R2S2R=$R; X=$R/experiments/R2S2V_mug; C=$R/experiments/finetune/2026-10-24/_common/code
O=$R/outputs/R2S2V_mug/hf25; L=$R/logs/hf25; mkdir -p $L
for p in "$@"; do
  case $p in FHN_high|FHN_low|SVO_high|SVO_low|CDS_high|CDS_low) ;; ev_FHN_[0-9]*|ev_SVO_[0-9]*|ev_CDS_[0-9]*) ;; *) echo "unknown part $p"; continue ;; esac
  M=$R/outputs/R2S2V_mug/done_exp_hf25_$p
  [ -f $M ] && { echo "$p already done"; continue; }
  pgrep -u $USER -f "[p]ool_retry.sh cfix_r2s2r_hf25_$p " > /dev/null && { echo "$p loop already running"; continue; }
  LOOP="bash $X/pool_retry.sh cfix_r2s2r_hf25_$p $M 44G bash $C/train.sh $p"
  case $p in
    ev_*) ARM=$(echo $p | cut -d_ -f2); CK=$(echo $p | cut -d_ -f3)
          pgrep -u $USER -f "[w]ait_ck $p" > /dev/null && { echo "$p waiter already running"; continue; }
          setsid nohup bash -c "exec -a 'wait_ck $p' bash -c 'until [ -f $O/${ARM}_low/checkpoint-$CK.safetensors ] && [ -f $O/${ARM}_high/checkpoint-$CK.safetensors ]; do sleep 300; done; sleep 60; $LOOP'" \
            > $L/wait_$p.log 2>&1 < /dev/null &
          echo "started waiter for $p ($!)" ;;
    *)    setsid nohup $LOOP > $L/loop_$p.log 2>&1 < /dev/null &
          echo "started $p loop $!" ;;
  esac
  sleep 2
done
