#!/bin/bash
# list this user's LoRA waiters / retry loops (script file, so pgrep never matches the ssh command line)
for pat in "lora_tactile/after_train.sh" "pool_retry.sh cfix_r2s2r_lora_" "pool_retry6.sh" "pool_retry7.sh"; do
  for pid in $(pgrep -u $USER -f "$pat"); do echo "$pid $(ps -o etime= -p $pid) $(ps -o args= -p $pid | cut -c1-120)"; done
done
