"""Fixed-squeeze baseline for the touch-data comparison (supervising session, 2026-10-26): does "force must be in the labels"
beat a fixed squeeze? T-obs's predicted finger targets (observed-pose policy) plus a CONSTANT closing offset per joint, equal to
the training set's median commanded-minus-observed angle (q_cmd - q, labels of dataset_cmp_t, test rows excluded):
  T-obs-off    median over every training frame (as asked)
  T-obs-offL   median over the training frames where the glove is loaded (total > 1 N): the offset while gripping
No training; the tables go next to the policies' in predict_cmp's format: <cmp>/<run>/<pred>/s<k>/gt_<ep>.npz.
  R2S2R=<root> python offset_baseline.py --pred pred_gt_r4 pred_gtyf_r4"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

R = Path(os.environ.get("R2S2R", ".")).resolve(); H = R / "outputs/R2S2V_mug/hf25"; CMP = R / "outputs/vtla/cmp"


def offsets():
    ds = H / "dataset_cmp_t"; test = set((ds / "test_ids.txt").read_text().split()); rows = [r for r in json.loads((ds / "rows.json").read_text()) if r["id"] not in test]
    d_all, d_load, names = [], [], None
    for r in rows:
        L = np.load(ds / "vtla" / f"{r['id']}.npz"); nm = [str(x) for x in L["joint_names"]]; names = names or nm; assert nm == names, r["id"]
        d = L["q_cmd"] - L["q"]; d_all.append(d); d_load.append(d[L["f_glove"].sum(1) > 1.0])
    A, B = np.concatenate(d_all), np.concatenate(d_load)
    return names, np.median(A, 0), np.median(B, 0), dict(rows=len(rows), frames=len(A), loaded_frames=len(B))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", nargs="+", required=True); a = ap.parse_args()
    names, off_all, off_load, info = offsets()
    rep = dict(info, joint_names=names, median_all_deg=np.degrees(off_all).round(2).tolist(), median_loaded_deg=np.degrees(off_load).round(2).tolist())
    for run, off in (("T-obs-off", off_all), ("T-obs-offL", off_load)):
        for pred in a.pred:
            src = CMP / "T-obs" / pred
            for f in sorted(src.rglob("*.npz")):                            # held-out: s<k>/gt_<ep>.npz; simulated test rows: <id>.npz
                P = dict(np.load(f)); jn = [str(x) for x in P["joint_names"]]; idx = [names.index(n) for n in jn]
                P["q"] = (P["q"] + off[idx][None, :]).astype(np.float32); P["arm"] = np.array(run); P["ckpt"] = np.array(f"T-obs + constant offset ({run})")
                o = CMP / run / pred / f.parent.relative_to(src); o.mkdir(parents=True, exist_ok=True); np.savez_compressed(o / f.name, **P)
        (CMP / run).mkdir(parents=True, exist_ok=True); (CMP / run / "offset.json").write_text(json.dumps(dict(rep, used="median_all_deg" if run == "T-obs-off" else "median_loaded_deg"), indent=1))
    print(json.dumps({k: rep[k] for k in ("rows", "frames", "loaded_frames")}), "\nmedian all (deg):", rep["median_all_deg"], "\nmedian loaded (deg):", rep["median_loaded_deg"])


if __name__ == "__main__":
    main()
