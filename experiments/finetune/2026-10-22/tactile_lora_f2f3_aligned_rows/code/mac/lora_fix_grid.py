"""Wrong-data vs fixed-data comparison board for the video + touch LoRA (user 2026-10-22: "evaluate the checkpoints
trained on the wrong data, lay the videos out, and run the fixed method so we can compare"). e2 / e3 were trained on
real rows whose video ran 1.9x slow (LESSONS D68); f2 / f3 are the same runs on dataset_fix, the only change being the
time-aligned real rows. One video per clip and start (real frame 0 / generated frame 0), 16 fps x 5 s:
  row 0   control (what every model sees) | real video + real touch at the same times (the fixed target) | the old,
          slowed real row (what e2 / e3 learned from)
  rows    step 200 / 300-400 / 600: e2 (wrong) | f2 (fixed) | e3 (wrong) | f3 (fixed); e2 / f2 use step 400 in the
          middle row (e2 was stopped at 400), e3 / f3 step 300. Each tile is the generated canvas (video | touch panel),
          labelled with total-force r / MAE vs the real touch; missing tiles say why.
Reads runs/clips/r2s2v_lora/{dataset,dataset_fix} and the pulled outputs runs/clips/r2s2v_lora/pull_out/<tag>/ (tag =
wan22fun-a14b-tact<run>c<step>-<real|sim>; pull them with r2s2v/lora_fix_pull.sh). Writes runs/clips/r2s2v_lora/fixcmp/.
  .venv/bin/python r2s2v/lora_fix_grid.py"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
from cliprep.wmexport import write_mp4  # noqa: E402

L = HERE / "runs/clips/r2s2v_lora"; PULL = L / "pull_out"; OUT = L / "fixcmp"
TW, TH, LH, N, FPS = 480, 212, 22, 81, 16
CLIPS = ("tsmug-edf79415", "tsmug-53780879")
ROWS = (("step 200", {"e2": 200, "f2": 200, "e3": 200, "f3": 200}),
        ("step 300 / 400", {"e2": 400, "f2": 400, "e3": 300, "f3": 300}),
        ("step 600", {"e2": None, "f2": 600, "e3": 600, "f3": 600}))
COLS = (("e2", "e2 mixed, WRONG data"), ("f2", "f2 mixed, FIXED data"), ("e3", "e3 real-only, WRONG data"), ("f3", "f3 real-only, FIXED data"))


def read(p):
    c = cv2.VideoCapture(str(p)); f = []
    while True:
        ok, x = c.read()
        if not ok:
            break
        f.append(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    return f


def total_scores(P, G):
    p, g = P.sum(1), G.sum(1); n = min(len(p), len(g)); p, g = p[:n], g[:n]
    r = float(np.corrcoef(p, g)[0, 1]) if p.std() > 1e-3 else float("nan")
    return r, float(np.abs(p - g).mean())


def tile(frames, k, label, color=(255, 255, 255)):
    t = np.zeros((TH + LH, TW, 3), np.uint8)
    if frames is not None:
        t[LH:] = cv2.resize(frames[min(k, len(frames) - 1)], (TW, TH), interpolation=cv2.INTER_AREA)
    else:
        t[LH:] = 40
    cv2.putText(t, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return t


def board(clip, start):
    src = "real" if start == "gt" else "sim"
    G = np.load(L / "dataset_fix" / "tactile" / f"{clip}.npz")["gt"]
    head = [(read(L / "dataset_fix" / "control" / f"{clip}.mp4"), "control (input to every model)", (255, 255, 255)),
            (read(L / "dataset_fix" / "train" / f"{clip}_gt.mp4"), "real + real touch, time-aligned (fixed target)", (120, 255, 120)),
            (read(L / "dataset" / "train" / f"{clip}_gt.mp4"), "old real row: video 1.9x slow (wrong target)", (255, 120, 120)),
            (None, f"{clip}, start: {'real frame 0' if start == 'gt' else 'generated frame 0'}", (255, 255, 255))]
    grid, have = [], 0
    for rname, steps in ROWS:
        row = []
        for run, cname in COLS:
            ck = steps[run]
            if ck is None:
                row.append((None, f"{rname}: {cname} - stopped at 400", (160, 160, 160))); continue
            v = PULL / f"wan22fun-a14b-tact{run}c{ck}-{src}" / f"{clip}_seed0.mp4"; npy = v.with_name(f"{clip}_seed0_tactile.npy")
            if v.exists():
                lab = f"{run} step {ck}"
                if npy.exists():
                    r, mae = total_scores(np.load(npy), G); lab += f"  touch r {r:.2f} / {mae:.1f} N"
                row.append((read(v), lab, (255, 120, 120) if run[0] == "e" else (120, 255, 120))); have += 1
            else:
                row.append((None, f"{run} step {ck}: queued / training", (160, 160, 160)))
        grid.append(row)
    frames = []
    for k in range(N):
        rows = [np.hstack([tile(f, k, lab, col) for f, lab, col in head])]
        rows += [np.hstack([tile(f, k, lab, col) for f, lab, col in row]) for row in grid]
        g = np.vstack(rows); cv2.putText(g, f"{k / FPS:.1f} s", (g.shape[1] - 60, g.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(g)
    OUT.mkdir(parents=True, exist_ok=True); p = OUT / f"fixcmp_{clip}_{src}start.mp4"; write_mp4(p, frames, FPS)
    cv2.imwrite(str(OUT / f"fixcmp_{clip}_{src}start_t5s.jpg"), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(p.name, "tiles with video:", have, "/ 11")


if __name__ == "__main__":
    for clip in CLIPS:
        for start in ("gt", "gen_s0"):
            board(clip, start)
