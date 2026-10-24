#!/bin/bash
# Waiter (setsid on the login node): after prep, launch the training loops (U and E first, G / R / regressor 10 min later, so
# U / E get the first free cards), then each eval as soon as both experts of a run are done:
#   ev_<RUN>_8 for U / G / E / R (matched passes per sample) and ev_E_4 / ev_G_4 (matched per-job exposure: U at epoch 8 has
#   seen each job ~4 times per sample). Log: logs/lora_ge/chain.log
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug/lora_ge; D=$R/outputs/R2S2V_mug
L() { bash $X/launch_ge.sh "$@"; }
until [ -f $D/done_exp_lora_ge_data ]; do sleep 60; done; echo "$(date +%T) dataset built"; L prep
until [ -f $D/done_exp_lora_ge_prep ]; do sleep 120; done; echo "$(date +%T) prep done"
L U_high U_low E_high E_low regr; sleep 600; L G_high G_low R_high R_low
todo="U G E R"
while [ -n "$todo" ]; do
  left=""
  for r in $todo; do
    if [ -f $D/done_exp_lora_ge_${r}_high ] && [ -f $D/done_exp_lora_ge_${r}_low ]; then
      echo "$(date +%T) $r trained"; L ev_${r}_8; case $r in E|G) L ev_${r}_4 ;; esac
    else left="$left $r"; fi
  done
  todo=$(echo $left); [ -n "$todo" ] && sleep 300
done
echo "$(date +%T) all evals launched"
