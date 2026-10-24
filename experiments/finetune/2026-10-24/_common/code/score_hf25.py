"""Scores for the HF25 arms (FHN / SVO / CDS) on dataset/test_cases.json, with the held-out eval's full metric set
(dual_mode_lora/code/eval_ge.metrics, relative touch, panel v3 thresholds: F1 @ 6 mm / pad / taxel, pad-time F1, time IoU,
onset / release, total and finger r, load-share TV, relative MAE, newton MAE with the oracle S) + mug / hand IoU of the
held-out videos vs the sim silhouettes (labeladh.py).
Reference rows next to every arm:
  frame-0 touch repeated        the start's touch for every frame (it knows frame 0, like the model)
  twin sim touch (own scale)    real-start cases only: the twin's simulated touch, relative to its own S
  dual-mode G (Gp8)             real-start cases: the dual-mode G-only LoRA, epoch 8, same start (real frame 0 + touch), its
                                twin control = the FHN format (lora_ge/out/Gp8/G_real_<ep8>_t_s0_tactile.npy)
  h2 step 600 [ABSOLUTE]        real-start cases: the earlier held-out F / H LoRA, scored the old absolute way (panel v1;
                                comparable on the r columns only)
  zero-shot A14B / real video   mug / hand IoU only
Onset is uninformative when the target is in contact from frame 0 (the sim touch is). Writes $R2S2R/outputs/R2S2V_mug/hf25/eval/
{scores.json, table.md}; every arm / checkpoint folder in hf25/out/ is scored. Reference rows whose inputs are not present (the
earlier runs' outputs, the mug / hand IoU jsons) are left out, so the script works outside the original cluster too.
  envs/armA/bin/python score_hf25.py"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; DM = HERE.parents[2] / "2026-10-22/dual_mode_lora/code"     # the metric code lives in this repo
sys.path.insert(0, str(DM))
from eval_ge import KEYS, labeladh, metrics, touch_scale  # noqa: E402
from panel_v3 import F_MAX, PANEL_VERSION, THR_REL, TOT_REL  # noqa: E402

R = Path(os.environ.get("R2S2R") or ("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r" if Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r").exists() else "."))
O = R / "outputs/R2S2V_mug/hf25"; DS = O / "dataset"; LG = R / "outputs/R2S2V_mug/lora_ge"; LT = R / "outputs/R2S2V_mug/lora_tactile"   # LG / LT: earlier runs' reference rows, used when present
GROUP = {("heldout", "real"): "held-out twin, real frame 0 -> real touch", ("heldout", "sim"): "held-out twin, generated frame 0 -> sim touch",
         ("insample", "gen"): "training row (in-sample) -> its sim touch"}


def main():
    res = []
    runs = sorted(d for d in (O / "out").glob("*c*") if d.is_dir()) if (O / "out").exists() else []
    for c in json.loads((DS / "test_cases.json").read_text()):
        z = np.load(DS / c["tactile"]); T, T_N, S = z["touch_rel"], z["touch"], float(z["S"]); ep8 = c["clip"].split("-")[1]
        base = dict(case=c["case"], group=GROUP[(c["kind"], c["source"])], clip=c["clip"])
        for d in runs:
            regr = d.name.endswith("_regr"); arm, ck = (d.name[:-5] if regr else d.name).split("c", 1)
            for f in sorted(d.glob(f"{c['case']}_s*_tactile.npy")):
                seed = int(f.name[len(c["case"]) + 2:].split("_")[0])
                adh = labeladh(f"wan22fun-a14b-hf25{arm.lower()}c{ck}-{c['source']}", c["clip"], seed) if c["kind"] == "heldout" and not regr else {}
                name = f"R6 regressor on the {arm} step {ck} video (no head)" if regr else f"{arm} step {ck} (own touch head)"
                res.append(dict(base, method=name, arm=arm, ckpt=int(ck), seed=seed, control=c["control"].get(arm), **metrics(np.load(f), T, S, T_N), **adh))
        res.append(dict(base, method="frame-0 touch repeated", seed=0, **metrics(np.repeat(T[:1], len(T), 0), T, S, T_N)))
        if c["kind"] == "heldout" and c["source"] == "real":
            st = DS / "test" / "simtouch" / f"{c['clip']}.npz"                    # shipped with the dataset; else the cluster copy
            st = st if st.exists() else LG / "simtouch" / f"{c['clip']}.npz"
            if st.exists():
                sim = np.load(st)["sim"]; res.append(dict(base, method="twin sim touch (own scale)", seed=0, **metrics(sim / touch_scale(sim), T, S, T_N)))
            g = LG / "out" / "Gp8" / f"G_real_{ep8}_t_s0_tactile.npy"
            if g.exists():
                res.append(dict(base, method="dual-mode G epoch 8 (twin control = FHN format)", seed=0, **metrics(np.load(g), T, S, T_N), **labeladh("wan22fun-a14b-geGp8gt", c["clip"])))
            h = LT / "out" / "wan22fun-a14b-tacth2c600-real" / f"{c['clip']}_seed0_tactile.npy"
            if h.exists():
                res.append(dict(base, method="h2 step 600 (F/H, unnormalised) [ABSOLUTE N scoring, panel v1]", seed=0,
                                **metrics(np.load(h), T_N, thr=0.15, tot=5.0, cap=3.0), **labeladh("wan22fun-a14b-tacth2c600-real", c["clip"])))
            res.append(dict(base, method="zero-shot A14B (video only)", seed=0, **labeladh("wan22fun-a14b-head-c5", c["clip"])))
            res.append(dict(base, method="real video (ceiling)", seed=0, **labeladh("real", c["clip"])))
    od = O / "eval"; od.mkdir(exist_ok=True)
    (od / "scores.json").write_text(json.dumps(dict(panel=PANEL_VERSION, cap_rel=F_MAX, thr_rel=THR_REL, tot_rel=TOT_REL, rows=res), indent=1))
    lines = ["| cases | method | n | " + " | ".join(KEYS) + " |", "|" + "---|" * (len(KEYS) + 3)]
    keyf = lambda x: (x["group"], x["method"])
    for k in sorted({keyf(x) for x in res}):
        xs = [x for x in res if keyf(x) == k]
        vals = [float(np.nanmean([x[m] for x in xs if m in x])) if any(m in x and not np.isnan(x[m]) for x in xs) else float("nan") for m in KEYS]
        lines.append(f"| {k[0]} | {k[1]} | {len({x['case'] for x in xs})} | " + " | ".join("-" if np.isnan(v) else (f"{v:.0f}" if m.endswith("_ms") else f"{v:.3f}") for v, m in zip(vals, KEYS)) + " |")
    (od / "table.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
