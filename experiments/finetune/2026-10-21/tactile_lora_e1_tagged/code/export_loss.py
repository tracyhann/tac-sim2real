"""All scalar series in the LoRA runs' TensorBoard logs -> one JSON {run: {tag: [[step, value, wall], ...]}}."""
import glob
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LT = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/outputs/R2S2V_mug/lora_tactile"); out = {}
for run in ("e1_high", "e1_low", "e2_high", "e2_low"):
    series = {}
    for f in sorted(glob.glob(str(LT / run / "logs" / "**" / "events.out.tfevents*"), recursive=True)):
        ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            series.setdefault(tag, []).extend([[e.step, e.value, e.wall_time] for e in ea.Scalars(tag)])
    out[run] = {k: sorted(v) for k, v in series.items()}
    print(run, {k: len(v) for k, v in out[run].items()})
Path(sys.argv[1]).write_text(json.dumps(out))
