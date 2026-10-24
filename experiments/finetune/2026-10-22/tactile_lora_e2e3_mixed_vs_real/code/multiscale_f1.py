"""Contact F1 at coarser spatial scales (user 2026-10-22: "1.5 mm is too strict; resample GT and prediction"). Inside each
of the glove's 15 pads, taxels are pooled into k x k blocks (k = 1: the native 1.5 mm cells; 2: 3 mm; 3: 4.5 mm; 4: 6 mm),
and a block is in contact when any of its taxels exceeds 0.15 N (forces clipped at F_MAX, as score_tactile). "pad" = whole
pads (15 units). Per frame F1 = 2TP / (2TP + FP + FN) over the units, averaged over frames where either side has contact.
Scores every decoded LoRA prediction in lora_tactile/out/ against the real and the simulated tactile, plus two baselines:
the sim replay's tactile vs real, and repeating the real frame-0 touch for all frames.
  envs/armA/bin/python multiscale_f1.py   -> outputs/R2S2V_mug/lora_tactile/multiscale_f1.json"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); LT = R / "outputs/R2S2V_mug/lora_tactile"
sys.path.insert(0, str(R / "experiments/R2S2V_mug/lora_tactile"))
from tactile_panel import F_MAX, SLICES  # noqa: E402

THR = 0.15; SCALES = [1, 2, 3, 4, "pad"]


def units(F, k):
    """(frames, units) bool contact for scale k"""
    out = []
    for pid, (a, b, r, c) in SLICES.items():
        blk = (np.minimum(np.clip(F[:, a:b], 0, None), F_MAX) > THR).reshape(len(F), r, c)
        if k == "pad":
            out.append(blk.reshape(len(F), -1).any(1)[:, None]); continue
        rr, cc = -(-r // k), -(-c // k); pad = np.zeros((len(F), rr * k, cc * k), bool); pad[:, :r, :c] = blk
        out.append(pad.reshape(len(F), rr, k, cc, k).any((2, 4)).reshape(len(F), -1))
    return np.concatenate(out, 1)


def f1(P, T, k):
    n = min(len(P), len(T)); p, t = units(P[:n], k), units(T[:n], k); fr = p.any(1) | t.any(1)
    if not fr.any():
        return float("nan")
    tp, fp, fn = (p & t)[fr].sum(1), (p & ~t)[fr].sum(1), (~p & t)[fr].sum(1)
    return float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1)))


def row(P, T):
    return {str(k): round(f1(P, T, k), 3) for k in SCALES}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="dataset"); ap.add_argument("--clips", nargs="+", default=["tsmug-edf79415", "tsmug-53780879"])
    ap.add_argument("--out", default="multiscale_f1.json")   # held-out: --ds dataset_ho --clips tsmug-55494490 tsmug-f8f7ed78 --out multiscale_f1_ho.json
    a = ap.parse_args(); res = {}
    for clip in a.clips:
        z = np.load(LT / a.ds / "tactile" / f"{clip}.npz"); G, S = z["gt"], z["sim"]
        res[f"{clip} | baseline: sim replay vs real"] = row(S, G)
        res[f"{clip} | baseline: real frame-0 touch repeated vs real"] = row(np.repeat(G[:1], len(G), 0), G)
        for p in sorted(glob.glob(str(LT / "out" / "*" / f"{clip}_seed*_tactile.npy"))):
            tag = Path(p).parent.name; P = np.load(p)
            res[f"{clip} | {tag} vs real"] = row(P, G); res[f"{clip} | {tag} vs sim"] = row(P, S)
    (LT / a.out).write_text(json.dumps(res, indent=1))
    print(f"{'':78s}" + "".join(f"{('k=' + str(k)) if k != 'pad' else 'pad':>8s}" for k in SCALES))
    for k, v in res.items():
        print(f"{k:78s}" + "".join(f"{v[str(s)]:8.3f}" for s in SCALES))


if __name__ == "__main__":
    main()
