#!/bin/bash
# Hold the f2/f3 training + e2_ev400 retry loops (SIGSTOP; they are only sleeping, no card held) until the held-out
# session's hog_4397541c and h3_low have had a pool card (its request, 2026-10-22 ~14:30 CDT), then SIGCONT them.
# Safety: release after 4 h regardless.   setsid nohup bash hold_f.sh <pid...> > log &   (early release: kill -CONT <pids>)
PIDS="$*"; R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; O=$R/outputs/R2S2V_mug
HOGMARK=$(ps -o args= -p 3417729 | awk '{print $4}')
seen() { pgrep -u $USER -f "[p]ool_run.py --name cfix_r2s2r_$1" >/dev/null || [ -n "$2" -a -f "$2" ]; }
kill -STOP $PIDS; echo "$(date +%T) held $PIDS (hog marker: $HOGMARK)"
end=$(( $(date +%s) + 4 * 3600 )); h=0; l=0
while [ $h = 0 -o $l = 0 ]; do
  seen ho_hog_4397541c "$HOGMARK" && h=1; seen lora_h3_low $O/done_exp_lora_h3_low && l=1
  [ $(date +%s) -gt $end ] && { echo "$(date +%T) timeout"; break; }
  sleep 30
done
kill -CONT $PIDS; echo "$(date +%T) released (hog seen $h, h3_low seen $l)"
