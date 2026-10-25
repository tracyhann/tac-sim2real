#!/bin/bash
# GPU smoke test (one card, ~30 min): both arms from the N0-VTLA base on the dataset's rows, the three stages for a few steps
# each, then the open-loop eval on three rows. Proves the code runs end to end with the real weights; the numbers mean nothing.
#   R2S2R=<root> bash smoke.sh            (after setup.sh; STAGES=pred:30,align:30,joint:30 by default)
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); C=$HERE/code
R=${R2S2R:?set R2S2R to the setup.sh root}; PY=$R/envs/vtla/bin/python
DS=${VTLA_DATASET:-$R/outputs/R2S2V_mug/hf25/dataset}; O=${VTLA_OUT:-$R/outputs/vtla/smoke}; STAGES=${STAGES:-pred:30,align:30,joint:30}
export N0VTLA_DATA_HOME=$R/cache/n0vtla HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=$R/models/n0-vtla-base
L=${SLURM_TMPDIR:-${TMPDIR:-}}                                     # copy the 8 GB checkpoint to node-local disk when there is one
if [ -n "$L" ] && [ -d "$L" ]; then mkdir -p $L/n0base && cp -n $BASE/model.safetensors $L/n0base/ && BASE=$L/n0base; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo "base $BASE, data $DS, out $O, stages $STAGES"
cd $C
$PY test_vtla.py
for arm in vts vtf; do
  echo "== $arm $(date +%T)"
  $PY train_vtla.py --arm $arm --data $DS --base $BASE --out $O/$arm --stages $STAGES --batch 8 --workers 4 --cache $R/cache/vtla_frames --log_every 5 --warmup 5 --save last
  $PY eval_vtla.py --ckpt $O/$arm/final --data $DS --ids r6_head_recorded r6_head_grip30 r10_front_claw --stride 16 --cache $R/cache/vtla_frames
done
$PY - $O <<'EOF'
import json, sys
from pathlib import Path
O = Path(sys.argv[1])
for arm in ("vts", "vtf"):
    run = json.loads((O / arm / "final" / "run.json").read_text()); ev = json.loads((O / arm / "final" / "eval.json").read_text())
    rows = [json.loads(l) for l in (O / arm / "log.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["stage"], []).append(r["loss"])
    print(arm, {s: (round(sum(v[:5]) / 5, 4), round(sum(v[-5:]) / 5, 4)) for s, v in by.items()}, "| weights missing", run["weights_report"]["missing"],
          "| eval model", {k: round(v, 3) for k, v in ev["model"].items()}, "hold", {k: round(v, 3) for k, v in ev["hold"].items()},
          "| z_shift touch / image", round(ev["z_shift_touch"], 3), round(ev["z_shift_image"], 3))
EOF
echo "smoke done $(date +%T)"
