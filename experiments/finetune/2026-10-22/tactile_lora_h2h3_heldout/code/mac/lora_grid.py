"""Grid videos of the video + tactile LoRA (control | real, then the two runs from the real and from a generated frame 0),
one 2 x 3 grid per clip, 1088 x 480 canvases (video | touch panel) scaled to 544 x 240 tiles. The step-200 in-sample grids
(runs/clips/r2s2v_lora/ck200/) were made the same way.
  .venv/bin/python r2s2v/lora_grid.py --ds runs/clips/r2s2v_lora/dataset_ho --pull runs/clips/r2s2v_lora/heldout/out \
      --runs h2:600 h3:200 --clips tsmug-55494490 tsmug-4397541c --out runs/clips/r2s2v_lora/heldout/grids --label held-out"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
from cliprep.wmexport import write_mp4  # noqa: E402

TW, TH = 544, 240
NAME = {"h2": "mixed (real + sim rows)", "h3": "real rows only", "e2": "mixed (real + sim rows)", "e3": "real rows only"}
NS = {"h2": 6, "h3": 2, "e2": 6, "e3": 2}                          # training samples per run (passes = steps / samples)


def read(p):
    c = cv2.VideoCapture(str(p)); fr = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr


def tile(f, label):
    t = cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (TW, 22), (0, 0, 0), -1); cv2.putText(t, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--pull", required=True); ap.add_argument("--runs", nargs=2, required=True)
    ap.add_argument("--clips", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--label", default="held-out"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); ds, pull, out = (HERE / x for x in (a.ds, a.pull, a.out)); out.mkdir(parents=True, exist_ok=True)
    runs = [(r.split(":")[0], int(r.split(":")[1])) for r in a.runs]
    for clip in a.clips:
        ref = ds / "test" / f"{clip}_gt.mp4" if (ds / "test" / f"{clip}_gt.mp4").exists() else ds / "train" / f"{clip}_gt.mp4"
        srcs = [(ds / "control" / f"{clip}.mp4", f"{a.label} {clip[6:]} - control (sim depth + hand/mug, touch black)"), (ref, "real recording + real glove touch")]
        for st, sl in (("real", "real frame 0"), ("sim", "generated frame 0")):
            for run, ck in runs:
                srcs.append((pull / f"wan22fun-a14b-tact{run}c{ck}-{st}" / f"{clip}_seed{a.seed}.mp4", f"{run} {NAME[run]}, step {ck} ({ck // NS[run]} passes) - {sl}"))
        vids = [(read(p), lab) for p, lab in srcs]; n = min(len(v) for v, _ in vids)
        assert n > 0, [str(p) for p, _ in srcs]
        frames = []
        for k in range(n):
            t = [tile(v[k], lab) for v, lab in vids]
            g = np.vstack([np.hstack(t[i:i + 2]) for i in (0, 2, 4)])
            cv2.putText(g, f"{k + 1:02d}/{n}", (TW * 2 - 62, TH * 3 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA); frames.append(g)
        tag = "_".join(f"{r}c{c}" for r, c in runs); write_mp4(out / f"{tag}_{clip}.mp4", frames, 16)
        sheet = np.vstack([frames[k] for k in (0, n // 2, n - 1)])
        cv2.imwrite(str(out / f"sheet_{tag}_{clip}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(out / f"{tag}_{clip}.mp4", n, "frames")


if __name__ == "__main__":
    main()
