#!/bin/bash
# User 10-23 ("为啥 H2 没有 200 步"): the pool never gave these parts a card (cfix share 4 < 8 in use); run them here in turn.
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug
cd $X && for p in h2_ev200 h2_ev400; do [ -f $R/outputs/R2S2V_mug/done_exp_lora_$p ] && continue; echo "== $p $(date +%T)"; bash $X/lora_tactile/run_lora.sh $p; done; echo "H2EV_DONE $(date +%T)"
