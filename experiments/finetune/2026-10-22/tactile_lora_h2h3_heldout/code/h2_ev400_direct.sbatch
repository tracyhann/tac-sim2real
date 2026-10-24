#!/bin/bash
# 10-23: h2_ev400 split off 827946 (h2_ev_direct, 3 h limit, ~2 h per eval part) onto its own card; 827946 skips it via the
# done_exp_lora_h2_ev400 marker touched at submit time (run_lora.sh touches it again on success; removed below on failure).
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug
cd $X && echo "== h2_ev400 $(date +%T)" && bash $X/lora_tactile/run_lora.sh h2_ev400 && echo "H2EV400_DONE $(date +%T)" || { rm -f $R/outputs/R2S2V_mug/done_exp_lora_h2_ev400; echo "H2EV400_FAILED $(date +%T)"; }
