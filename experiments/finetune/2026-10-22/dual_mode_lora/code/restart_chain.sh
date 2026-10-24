#!/bin/bash
X=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/experiments/R2S2V_mug/lora_ge
bash $X/stop_chain.sh; sleep 2; bash $X/start_chain.sh; sleep 1; pgrep -u $USER -af "lora_ge/chain_ge[.]sh"
