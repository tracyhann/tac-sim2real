#!/bin/bash
# list cfix_r2s2r pool_retry loops: name, whether its pool_run exists, loop age
for pid in $(pgrep -u $USER -f "pool_retry[.]sh cfix_r2s2r"); do
  n=$(ps -o args= -p $pid | awk '{print $3}'); e=$(ps -o etime= -p $pid | tr -d ' ')
  r=$(pgrep -u $USER -f "[p]ool_run.py --name $n --gpus" > /dev/null && echo RUNNING || echo waiting); echo "$n $r loop_age=$e pid=$pid"
done | sort -u -k1,1
