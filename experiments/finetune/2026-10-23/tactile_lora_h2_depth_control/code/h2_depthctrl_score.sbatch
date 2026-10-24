#!/bin/bash
# 10-23: video scores of the H2 depth-control run (tag ...tacth2c600da-real), same three scorers as the h*_ev parts.
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug; A=$X/../A_baseline; B_=$X/../B_sim2wm
HT="tsmug-55494490 tsmug-4397541c"; M=wan22fun-a14b-tacth2c600da-real
cd $X && for s in $A/evaluate.py $B_/adherence.py labeladh.py; do bash $X/run_r2s.sh $s --clips $HT --models $M; done; echo "H2DA_SCORED $(date +%T)"
