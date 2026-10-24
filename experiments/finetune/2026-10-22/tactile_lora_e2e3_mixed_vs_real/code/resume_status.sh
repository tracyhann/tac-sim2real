#!/bin/bash
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r
for p in e2_high e3_high; do
  echo "== $p: $(tail -n 1 $R/logs/r2s2v/retry_lora_$p.log)"
  f=$R/logs/r2s2v/pool_cfix_r2s2r_lora_$p.log
  grep -h "resume_lora_weights\|did not load\|Traceback\|Error" $f 2>/dev/null | tail -n 4 | cut -c1-260
  grep -o "Steps: *[0-9]*%[^,]*" $f 2>/dev/null | tail -n 1
done
