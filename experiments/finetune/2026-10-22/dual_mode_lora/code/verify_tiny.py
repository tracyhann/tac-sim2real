"""Check the tiny-model runs of train_control_lora_ge.py (run_ge.sh part prep): run 1 = steps 1-3 (checkpoint at 2),
run 2 = resumed to step 6 (--resume_lora_weights). Every dumped batch: the mask is the job's mask (ge_mask.check_mask at
the training size), the masked input is pixel * (1 - mask) (so the E panel / G later frames are zero), the control is
the job's control file (decoded + resized like the collate); the per-step log: finite loss, the sample order of both runs
equals the seeded epoch order (randperm(n, seed + epoch)) with run 2 continuing exactly after the resumed step.
  envs/videoxfun/bin/python verify_tiny.py RUN_DIR META"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ge_mask import check_mask, panel_x0  # noqa: E402

run, meta = Path(sys.argv[1]), Path(sys.argv[2]); rows = json.loads(meta.read_text()); DS = meta.parent; n = len(rows)
a = Namespace(ge_panel_x0=832, ge_canvas_w=1088); res = dict(batches=[])


def dec(p, h, w):
    c = av.open(str(p)); x = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)]); c.close()
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False, antialias=True) * 2 - 1


for p in sorted((run / "ge_dump").glob("batch_*.pt")):
    b = torch.load(p, weights_only=False, map_location="cpu"); m = b["mask"][0]; ge = b["ge"][0]; pv = b["pixel_values"][0]; mp = b["mask_pixel_values"][0]
    f, _, h, w = pv.shape; x0 = panel_x0(w, a); check_mask(m, ge, x0)
    assert torch.allclose(mp, pv * (1 - m.to(pv.dtype))), "masked input != pixel * (1 - mask)"
    zero_unknown = bool((mp[m.bool().expand_as(mp)] == 0).all())
    cv = dec(DS / ge["control"], h, w); d = float((b["control_pixel_values"][0].float() - cv).abs().mean())
    row = next(r for r in rows if r["id"] == ge["id"]); other = row["control_alt_file_path"] if ge["ctrl"] == "main" else row["control_file_path"]
    d_other = float((b["control_pixel_values"][0].float() - dec(DS / other, h, w)).abs().mean()) if other else None
    res["batches"].append(dict(file=p.name, **{k: ge[k] for k in ("id", "mode", "touch0", "ctrl")}, unknown_zero=zero_unknown, ctrl_diff=d, ctrl_diff_other=d_other))
    print(res["batches"][-1]); assert zero_unknown and d < 0.03 and (d_other is None or d_other > d + 0.05)
log = [json.loads(x) for x in (run / "ge_log.jsonl").read_text().splitlines()]
res["log"] = [(x["step"], x["ge"][0]["id"], x["ge"][0]["mode"], round(x["loss"], 4)) for x in log]
assert all(np.isfinite(x["loss"]) for x in log)
order = {e: [rows[i]["id"] for i in torch.randperm(n, generator=torch.Generator().manual_seed(42 + e)).tolist()] for e in range(3)}
by_step = {}
for x in log:
    by_step.setdefault(x["step"], []).append(x["ge"][0]["id"])
for s, ids in sorted(by_step.items()):
    e, k = (s - 1) // n, (s - 1) % n; exp = order[e][k]
    print("step", s, "epoch", e, "ids", ids, "expected", exp); assert all(i == exp for i in ids), (s, ids, exp)
steps = sorted(by_step); assert steps == list(range(1, steps[-1] + 1)), steps
res["resume_ok"] = True; res["steps"] = steps; (run / "verify.json").write_text(json.dumps(res, indent=1)); print("TINY OK", steps)
