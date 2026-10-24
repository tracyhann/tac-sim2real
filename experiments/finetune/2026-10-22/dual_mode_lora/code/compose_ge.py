"""Side-by-side videos of the held-out clips (dual-mode LoRA eval), 16 fps, labelled:
  E  <row>_E_<ctrl>.mp4   the given video | real touch | U | E-only | R (real-only) | regressor (real + sim): the touch panels
                          decoded and re-drawn (tactile_panel.encode), so every panel uses the same encoding
  G  <row>_G_<t|n>.mp4    top: sim control | real video + real touch; bottom: U | G-only, each generated video + its panel
  envs/armA/bin/python compose_ge.py [--epoch 8]   -> lora_ge/videos/"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/experiments/R2S2V_mug") / "lora_tactile"))
from ge_cases import cases  # noqa: E402
from ge_data import DS, LG, read_video, write_video  # noqa: E402
from panel_v3 import encode  # noqa: E402  (relative touch)

X0, H, N = 832, 480, 81


def label(img, text, sub=""):
    bar = np.full((34, img.shape[1], 3), 24, np.uint8); cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (8 + 11 * len(text), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 200, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def panels(P):
    return np.stack([np.repeat(encode(p)[..., None], 3, -1) for p in P])


def blank(w):
    return np.zeros((N, H, w, 3), np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epoch", type=int, default=8); a = ap.parse_args(); p = a.epoch
    rows = {r["id"]: r for r in json.loads((DS / "rows.json").read_text())}; out = LG / "videos"; out.mkdir(exist_ok=True)
    sc = {(x["case"], x["method"]): x for x in json.loads((LG / "eval/scores.json").read_text())["rows"]} if (LG / "eval/scores.json").exists() else {}
    tag = lambda case, m: (f"F1@6mm {sc[(case, m)]['f1_6mm']:.2f}  timeIoU {sc[(case, m)]['time_iou']:.2f}" if (case, m) in sc else "")
    for c in cases(None):
        r = rows[c["row"]]; T = np.load(DS / r["tactile"])["touch_rel"]; vid, _ = read_video(DS / r["file_path"], size=None)
        if c["job"] == "E":
            seq = [vid[:, :, :X0], panels(T)]
            for d in (f"Up{p}", f"Ep{p}", f"Rp{p}", "regr_E_best"):
                f = LG / "out" / d / f"{c['case']}_s0_tactile.npy"; seq.append(panels(np.load(f)) if f.exists() else blank(256))
            heads = [f"given video ({r['actor']}, {c['ctrl']} control)", "real touch", f"U ep{p}", f"E-only ep{p}", f"R ep{p}", "regressor (R6)"]
            subs = ["", "", tag(c["case"], f"Up{p}"), tag(c["case"], f"Ep{p}"), tag(c["case"], f"Rp{p}"), tag(c["case"], "regressor R6, real+sim")]
            fr = [np.concatenate([label(s[k], h, "") for s, h in zip(seq, heads)], 1) for k in range(N)]
            foot = np.full((30, fr[0].shape[1], 3), 24, np.uint8); x = X0 + 256
            for s in subs[2:]:
                cv2.putText(foot, s.replace("  ", " "), (x + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 200, 255), 1, cv2.LINE_AA); x += 256
            fr = [np.concatenate([f, foot], 0) for f in fr]
        else:
            ctl, _ = read_video(DS / c["control"], size=None); seqs = []
            for name, d in ((f"U ep{p}", f"Up{p}"), (f"G-only ep{p}", f"Gp{p}")):
                f = LG / "out" / d / f"{c['case']}_s0.mp4"
                if f.exists():
                    g, _ = read_video(f, size=None); P = np.load(LG / "out" / d / f"{c['case']}_s0_tactile.npy"); seqs.append((name, np.concatenate([g[:, :, :X0], panels(P)], 2), tag(c["case"], d)))
                else:
                    seqs.append((name, blank(X0 + 256), "not run"))
            top = [np.concatenate([label(ctl[k, :, :X0], "sim control", "  scene depth + shadowless sim hand / mug"),
                                   label(np.concatenate([vid[k, :, :X0], panels(T[k:k + 1])[0]], 1), "real video + touch", "  (not given; frame 0 only" + (" + its touch)" if c["touch0"] else ")"))], 1) for k in range(N)]
            bot = [np.concatenate([label(s[1][k], s[0], "  " + s[2]) for s in seqs], 1) for k in range(N)]
            top = [np.concatenate([t, np.zeros((t.shape[0], b.shape[1] - t.shape[1], 3), np.uint8)], 1) if t.shape[1] < b.shape[1] else t for t, b in zip(top, bot)]
            fr = [np.concatenate([t, b], 0) for t, b in zip(top, bot)]
        fr = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in fr]
        write_video(out / f"{c['case']}.mp4", fr, crf=20); print(c["case"], fr[0].shape, flush=True)


if __name__ == "__main__":
    main()
