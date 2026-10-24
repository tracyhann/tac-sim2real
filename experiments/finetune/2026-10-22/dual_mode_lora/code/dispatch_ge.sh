#!/bin/bash
# Priority dispatcher for the dual-mode parts (replaces chain_ge.sh after the cap went back to 4): pool_retry loops race for
# freed cards in no order, so only the highest-priority unfinished parts that are not running keep WAITING loops (as many as there are free cards, at least one); lower ones
# have their waiting loops stopped (running steps are never touched; cap_guard_ge.sh still enforces the cap). Evals start
# once both experts of their run are done. Every 2 min; exits when every part is done. Log: logs/lora_ge/dispatch.log
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug/lora_ge; D=$R/outputs/R2S2V_mug
PRIO=${GE_PRIO:-"U_high U_low E_high E_low ev_U_8 G_high G_low ev_E_8 R_high R_low ev_R_8 ev_G_8 ev_E_4 ev_G_4"}
ready() { case $1 in ev_*) local r=$(echo $1 | cut -d_ -f2); [ -f $D/done_exp_lora_ge_${r}_high ] && [ -f $D/done_exp_lora_ge_${r}_low ] ;; *) true ;; esac; }
while true; do
  steps=$(bash -l -c "squeue -u $USER -s -h -o '%j'" 2>/dev/null); waiting_kept=0; left=0
  c=$(cat $X/../pool_cap 2>/dev/null); [[ "$c" =~ ^[0-9]+$ ]] && [ "$c" -le 8 ] && cap=$c || cap=4
  nrun=$(echo "$steps" | grep -c '^pvbrun-cfix_r2s2r'); allow=$(( cap - nrun )); [ $allow -lt 1 ] && allow=1   # as many waiting as free cards
  for p in $PRIO; do
    [ -f $D/done_exp_lora_ge_$p ] && continue; left=1
    echo "$steps" | grep -qx "pvbrun-cfix_r2s2r_ge_$p" && continue            # running: leave it alone
    hasloop=$(pgrep -u $USER -f "pool_retry[.]sh cfix_r2s2r_ge_$p " > /dev/null && echo 1 || echo 0)
    if [ $waiting_kept -lt $allow ] && ready $p; then
      waiting_kept=$((waiting_kept + 1)); [ $hasloop = 0 ] && { echo "$(date +%T) queue $p"; bash $X/launch_ge.sh $p; }
    elif [ $hasloop = 1 ]; then
      echo "$(date +%T) hold $p (a higher-priority part waits)"; bash $X/stop_ge.sh $p
    fi
  done
  [ $left = 0 ] && { echo "$(date +%T) all parts done"; exit 0; }
  sleep 120
done
