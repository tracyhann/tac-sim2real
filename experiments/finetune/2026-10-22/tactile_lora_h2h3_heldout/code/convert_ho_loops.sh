#!/bin/bash
# move this session's pool_retry loops to the new pool_retry.sh (cap from pool_cap): start a new loop per part, then kill the old bash
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug
for p in $(pgrep -u $USER -f "hog_first.s""h"); do kill $p && echo "killed hog_first $p"; done
declare -A OLD
for p in h2_high h2_low h3_high h3_low; do OLD[$p]=$(pgrep -u $USER -f "pool_retry.sh cfix_r2s2r_lora_$p " | tr "\n" " "); done
OLDHOG=$(pgrep -u $USER -f "pool_retry.sh cfix_r2s2r_ho_hog_4397541c " | tr "\n" " ")
echo "old: ${OLD[@]} hog: $OLDHOG"
bash $X/lora_tactile/launch_lora.sh h2_high h2_low h3_high h3_low; bash $X/launch_ho.sh hog_4397541c
sleep 2
for p in h2_high h2_low h3_high h3_low; do for o in ${OLD[$p]}; do kill -CONT $o 2>/dev/null; kill -TERM $o && echo "killed old loop $p $o"; done; done
for o in $OLDHOG; do kill -TERM $o && echo "killed old loop hog $o"; done
sleep 2; ps -u $USER -o pid,etime,args | grep "[p]ool_retry.sh cfix_r2s2r_\(lora_h\|ho_\)" | cut -c1-110
