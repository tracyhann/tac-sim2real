#!/bin/bash
# Give a card to the starved f3_low (parent session 2026-10-23: the 8-card raise was for f2/f3 + e2 ev400): pause my lowest-
# priority running ge part (R_high) and hold my waiting loops (R_low, regrsp) so they cannot take the freed slot; then wait until
# f3_low's training steps ADVANCE (two readings of its pool log 4 min apart), and only then re-queue my loops. Log: logs/lora_ge/yield_f3.log
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug; L=$R/logs/r2s2v/pool_cfix_r2s2r_lora_f3_low.log
echo "$(date +%T) f3_low loop: $(pgrep -u $USER -af 'pool_retry.sh cfix_r2s2r_lora_f3_low' | grep -v pgrep | head -1 | cut -c1-80)"
bash $X/lora_ge/stop_ge.sh R_high R_low regrsp
step() { grep -ao '| *[0-9]*/[0-9]* \[' $L 2>/dev/null | tail -1 | grep -o '[0-9]*/' | tr -d /; }
s0=""; for i in $(seq 1 60); do                                  # up to 4 h
  sleep 240; s=$(step); run=$(bash -l -c "squeue -u $USER -s -h -o '%j'" 2>/dev/null | grep -c 'pvbrun-cfix_r2s2r_lora_f3_low$')
  echo "$(date +%T) f3_low step running=$run progress=$s"
  if [ "$run" -ge 1 ] && [ -n "$s" ] && [ -n "$s0" ] && [ "$s" -gt "$s0" ]; then echo "$(date +%T) f3_low advancing ($s0 -> $s): re-queue my loops"; break; fi
  [ "$run" -ge 1 ] && s0=$s
  [ -f $R/outputs/R2S2V_mug/done_exp_lora_f3_low ] && { echo "f3_low done"; break; }
done
bash $X/lora_ge/launch_ge.sh R_high R_low regrsp; echo "$(date +%T) done"
