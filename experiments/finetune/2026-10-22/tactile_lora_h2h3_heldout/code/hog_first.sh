#!/bin/bash
# let hog_4397541c (one ~10-min video) take the next card before h3_low (my own loop): pause the h3_low pool_retry loop while it
# sleeps (never while it holds the queue lock), resume it once hog_4397541c has a pool card or is done.
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r
P="pool_retry.sh cfix_r2s2r_lora_h3""_low"; H="pool_run.py --name cfix_r2s2r_ho_hog_4397""541c"
L=$(pgrep -u $USER -f "$P" | head -1); [ -z "$L" ] && { echo "no h3_low loop"; exit 0; }
until pgrep -P $L -x sleep > /dev/null; do sleep 2; done; kill -STOP $L; echo "$(date +%T) paused h3_low loop $L"
until pgrep -u $USER -f "$H" > /dev/null || [ -f $R/outputs/R2S2V_mug/done_exp_ho_hog_4397541c ]; do sleep 20; done
kill -CONT $L; echo "$(date +%T) hog_4397541c has a card: resumed h3_low loop $L"
