"""Held-out table of the video + tactile LoRA (2026-10-22): runs h2 (mixed real + sim rows) and h3 (real rows only),
trained on edf79415 / 53780879, scored on 55494490 / 4397541c (never trained on; f8f7ed78 / 9811b8f8 had no twin). Per row, mean
over the two held-out clips (and seeds), plus each clip:
  tactile vs the real glove: total-force r / MAE (N), mean per-finger r, contact F1 at 6 mm blocks (k = 4) and per pad
  video: mug / hand IoU vs the simulation's silhouettes (labeladh), LPIPS / DINO vs the real video and the mug-trajectory
  error (evaluate), depth AbsRel vs the simulation (adherence)
Baselines on the same clips: (a) the twin's own sim tactile as is, (b) the real frame-0 touch repeated, (c) zero-shot
A14B video under the same star control (head-c5 seed 0; no tactile output), and the real video (labeladh ceiling).
  envs/armA/bin/python lora_tactile/ho_table.py   -> outputs/R2S2V_mug/lora_tactile/heldout_table.json"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); LT = R / "outputs/R2S2V_mug/lora_tactile"; DS = LT / "dataset_ho"
sys.path.insert(0, str(R / "experiments/R2S2V_mug/lora_tactile")); sys.path.insert(0, str(R / "experiments/R2S2V_mug"))
from multiscale_f1 import f1  # noqa: E402
from score_tactile import scores  # noqa: E402

HT = ["tsmug-55494490", "tsmug-4397541c"]; O = R / "outputs"


def tac(P, G):
    s = scores(P, G); return dict(total_r=s["total_r"], total_mae=s["total_mae"], finger_r=s["finger_r"], f1_6mm=f1(P, G, 4), f1_pad=f1(P, G, "pad"), f1_1p5mm=s["contact_f1"])


def vid(model, clip, seed):
    g = lambda p: json.load(open(p)) if Path(p).exists() else {}
    L = g(O / f"R2S2V_mug/labeladh/{clip}/{model}__seed{seed}.json") if model != "real" else g(O / f"R2S2V_mug/labeladh/{clip}/real.json")
    E = g(O / f"A_baseline/eval/{clip}/{model}__seed{seed}.json"); A = g(O / f"B_sim2wm/adherence/{clip}/{model}__seed{seed}.json")
    return dict(mug_iou=L.get("mug_iou"), hand_iou=L.get("hand_iou"), lpips=E.get("lpips"), dino=E.get("dino_patch"), traj=E.get("obj_rel_err_pct"), absrel=A.get("agent_absrel"))


def mean(v):
    v = [x for x in v if isinstance(x, (int, float)) and not math.isnan(x)]; return float(np.mean(v)) if v else float("nan")


def main():
    T = {c: np.load(DS / "tactile" / f"{c}.npz") for c in HT}; rows = []

    def add(label, kind, per):                                  # per: {clip: [dict, ...]} -> mean over clips of the clip means
        keys = sorted({k for ds in per.values() for d in ds for k in d})
        pc = {c: {k: mean([d.get(k) for d in ds]) for k in keys} for c, ds in per.items()}
        rows.append(dict(label=label, kind=kind, n=sum(len(ds) for ds in per.values()), mean={k: mean([pc[c][k] for c in pc]) for k in keys}, per_clip=pc))

    add("(a) twin's own sim tactile", "baseline", {c: [tac(T[c]["sim"], T[c]["gt"])] for c in HT})
    add("(b) real frame-0 touch repeated", "baseline", {c: [tac(np.repeat(T[c]["gt"][:1], len(T[c]["gt"]), 0), T[c]["gt"])] for c in HT})
    add("(c) zero-shot A14B, star control", "baseline", {c: [vid("wan22fun-a14b-head-c5", c, 0)] for c in HT})
    add("real video (labeladh ceiling)", "reference", {c: [vid("real", c, 0)] for c in HT})
    for run, cks in (("h2", (200, 300, 400, 600)), ("h3", (100, 200, 400, 600)), ("f2", (600,)), ("f3", (200,))):   # f2 / f3 on the held-out clips: 10-23
        for ck in cks:
            for st in ("real", "sim"):
                tag = f"wan22fun-a14b-tact{run}c{ck}-{st}"; per = {}
                for c in HT:
                    for p in sorted(glob.glob(str(LT / "out" / tag / f"{c}_seed*_tactile.npy"))):
                        seed = int(Path(p).stem.split("_seed")[1].split("_")[0]); P = np.load(p)
                        per.setdefault(c, []).append({**tac(P, T[c]["gt"]), **vid(tag, c, seed)})
                if per:
                    passes = ck // (2 if run in ("h3", "f3") else 6)    # samples per run: h2 / f2 6, h3 / f3 2
                    add(f"{run} step {ck} ({passes} passes) · {'real' if st == 'real' else 'generated'} frame 0", run, per)
    # depth control (user 10-23): h2 step 600 with the REAL clip's Depth Anything V2 depth as control (no sim), real frame 0 start
    tag = "wan22fun-a14b-tacth2c600da-real"; per = {}
    for c in HT:
        for p in sorted(glob.glob(str(LT / "out" / tag / f"{c}_seed*_tactile.npy"))):
            seed = int(Path(p).stem.split("_seed")[1].split("_")[0]); per.setdefault(c, []).append({**tac(np.load(p), T[c]["gt"]), **vid(tag, c, seed)})
    if per:
        add("h2 step 600 (100 passes) · real frame 0 · Depth Anything control, no sim", "h2da", per)
    (LT / "heldout_table.json").write_text(json.dumps(rows, indent=1))
    cols = ["total_r", "total_mae", "finger_r", "f1_6mm", "f1_pad", "mug_iou", "hand_iou", "lpips", "dino", "traj"]
    print("| row | n | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
    for r in rows:
        print(f"| {r['label']} | {r['n']} | " + " | ".join("–" if math.isnan(r["mean"].get(k, float("nan"))) else f"{r['mean'][k]:.3f}" for k in cols) + " |")


if __name__ == "__main__":
    main()
