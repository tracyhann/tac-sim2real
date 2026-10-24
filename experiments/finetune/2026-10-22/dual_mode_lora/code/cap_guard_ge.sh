#!/bin/bash
# Keep the shared card cap after a temporary raise ends (user: 4 cards, or R2S2V_mug/pool_cap while a raise lasts; every
# cfix_r2s2r step counts). pool_retry.sh only gates NEW steps, so when pool_cap is withdrawn the dual-mode steps that started
# under the raise would keep running above 4. Every 2 min: if the running cfix_r2s2r steps exceed the cap, pause the
# lowest-priority dual-mode parts (R, then G, then E, then U; low expert first) with stop_ge.sh (loop + pool_run + scancel)
# and re-queue their loops (launch_ge.sh): they resume from their newest checkpoint (saved every 15 steps) when a slot frees.
# Exits when every training part is done. Log: logs/lora_ge/cap_guard.log
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug; D=$R/outputs/R2S2V_mug
ORDER="R_low R_high G_low G_high E_low E_high U_low U_high"
while true; do
  left=0; for p in $ORDER; do [ -f $D/done_exp_lora_ge_$p ] || left=1; done; [ $left = 0 ] && { echo "$(date +%T) all training parts done"; exit 0; }
  c=$(cat $X/pool_cap 2>/dev/null); [[ "$c" =~ ^[0-9]+$ ]] && [ "$c" -le 8 ] && cap=$c || cap=4
  total=$(ps -u $USER -o args | grep -c '[p]ool_run.py --name cfix_r2s2r')
  if [ "$total" -gt "$cap" ]; then
    for p in $ORDER; do
      [ "$total" -le "$cap" ] && break
      if pgrep -u $USER -f "[p]ool_run.py --name cfix_r2s2r_ge_$p " > /dev/null; then
        echo "$(date +%T) total $total > cap $cap: pausing $p"; bash $X/lora_ge/stop_ge.sh $p; sleep 10; bash $X/lora_ge/launch_ge.sh $p; total=$((total - 1))
      fi
    done
  fi
  sleep 120
done
