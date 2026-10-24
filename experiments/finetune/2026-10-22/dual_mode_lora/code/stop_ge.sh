#!/bin/bash
# stop the pool_retry loops of the given dual-mode parts AND their running pool steps. Killing pool_run.py alone does NOT stop
# the srun step on the compute node (found 2026-10-23: an orphaned E_low kept training into the same folder), so the steps
# named pvbrun-cfix_r2s2r_ge_<part> are scancel'ed too. Does not touch the chain waiter (stop_chain.sh).
for p in "$@"; do
  for pid in $(pgrep -u $USER -f "pool_retry.sh cfix_r2s2r_ge_$p "); do echo "kill loop $p $pid"; kill $pid; done
  for pid in $(pgrep -u $USER -f "pool_run.py --name cfix_r2s2r_ge_$p "); do echo "kill pool_run $p $pid"; kill $pid; done
  for st in $(bash -l -c "squeue -u $USER -s -h -o '%i %j'" 2>/dev/null | awk -v n="pvbrun-cfix_r2s2r_ge_$p" '$2 == n {print $1}'); do
    echo "scancel step $p $st"; bash -l -c "scancel $st" < /dev/null
  done
done
