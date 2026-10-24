"""Contact state and timing (finger_metrics.py) for every method on the release videos: the same predictions, reference and
baselines as compare_full.py. Main threshold 0.04 S; 0.02 / 0.08 S as a sensitivity check (all new videos only).
  /usr/bin/python3 compare_contact.py --pull <dir with U8/ E8/ E4/ R8/ f2/ R6_s*/> --bundle <scene_mug runs/clips/r2s2v_handpose> --out <dir>"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from finger_metrics import FIN_THR, KEYS, finger_metrics, touch_scale  # noqa: E402

GROUPS = [("all new (r10 + r8b, 19)", lambda c: c["group"] in ("r10", "r8b")), ("new: r10, other scenes (7)", lambda c: c["group"] == "r10"),
          ("new: r8b, box scene / top photo (12)", lambda c: c["group"] == "r8b"), ("r6 head = U8 / E / R6 training rows (6)", lambda c: c["group"] == "r6")]
METHODS = [("U8 (both jobs, mixed, 632 steps)", "U8", 1), ("E8 (estimate, mixed, 632 steps)", "E8", 1), ("E4 (estimate, mixed, 316 steps)", "E4", 1),
           ("R8 (estimate, real only, 320 steps)", "R8", 1), ("f2 (video + touch LoRA)", "f2", 0)]
BASE0, BASEREC = "frame-0 sim touch repeated", "recorded replay's touch"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pull", required=True); ap.add_argument("--bundle", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); pull, bun, out = Path(a.pull), Path(a.bundle), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((HERE / "cases.json").read_text())
    G = {c["id"]: np.load(bun / c["bundle"] / "tactile.npz")["normal_N"].astype(np.float64) for c in cases}
    S = {i: touch_scale(g) for i, g in G.items()}; Grel = {i: g / S[i] for i, g in G.items()}; rec = Grel[next(c["id"] for c in cases if c["id"].endswith("_recorded"))]
    seeds = sorted(p.name for p in pull.glob("R6_s*") if p.is_dir()); P = {}
    for name, d, rel in METHODS + [(f"R6 {s}", s, 1) for s in seeds]:
        for c in cases:
            f = pull / d / f"{c['id']}_tactile.npy"
            if f.exists():
                x = np.load(f).astype(np.float64); P[name, c["id"]] = x if rel else x / S[c["id"]]
    for c in cases:
        P[BASE0, c["id"]] = np.repeat(Grel[c["id"]][:1], len(Grel[c["id"]]), 0)
        if not c["id"].endswith("_recorded"):
            P[BASEREC, c["id"]] = rec
    byid = {c["id"]: c for c in cases}; rows = []
    for (m, i), x in P.items():
        for thr in (0.02, FIN_THR, 0.08):
            rows.append(dict(id=i, group=byid[i]["group"], method=m, thr=thr, **finger_metrics(x, Grel[i], thr)))
    (out / "scores_contact.json").write_text(json.dumps(rows, indent=1))
    order = [m for m, _, _ in METHODS] + ["R6"] + [BASE0, BASEREC]; r6 = [f"R6 {s}" for s in seeds]
    fmt = lambda v, k: "-" if not np.isfinite(v) else (f"{v:.1f}" if k.endswith("_err") else f"{v:.2f}")
    gm = lambda xs, k: float(np.nanmean([x[k] for x in xs])) if any(np.isfinite(x[k]) for x in xs) else float("nan")

    def table(sel, thr):
        L = ["| method | n | " + " | ".join(KEYS) + " |", "|---|---|" + "---|" * len(KEYS)]
        for m in order:
            ms = r6 if m == "R6" else [m]; per = [[r for r in rows if r["method"] == q and r["thr"] == thr and sel(byid[r["id"]])] for q in ms]; per = [p for p in per if p]
            if per:
                A = np.array([[gm(p, k) for k in KEYS] for p in per]); mu = np.nanmean(A, 0)
                cell = lambda j, k: fmt(mu[j], k) + (f" ± {fmt(np.nanstd(A[:, j]), k)}" if len(per) > 1 and np.isfinite(mu[j]) else "")
                L.append(f"| {'R6 regressor (3 seeds)' if m == 'R6' else m} | {len(per[0])} | " + " | ".join(cell(j, k) for j, k in enumerate(KEYS)) + " |")
        return L
    md = [f"Contact state and timing (finger_metrics.py). Parts thumb / index / middle / ring / little / palm; contact = part total > {FIN_THR} S "
          "(relative, ~1 N at the median S); errors in frames (16 fps). Up = finger_acc, finger_f1; down = the rest.", ""]
    for g, sel in GROUPS:
        md += [f"**{g}**", ""] + table(sel, FIN_THR) + [""]
    for thr in (0.02, 0.08):
        md += [f"**Sensitivity: all new videos, threshold {thr} S**", ""] + table(GROUPS[0][1], thr) + [""]
    (out / "scores_contact.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
