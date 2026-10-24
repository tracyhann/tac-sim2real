"""Scores and tables for the dual-mode LoRA (held-out clips only), on RELATIVE touch (panel v3: F / S, S = the episode's
touch scale; user 2026-10-22). Primary, scale-free (prediction and truth both relative, both clipped at the panel's caps):
  f1_6mm      contact F1, taxels pooled in 4 x 4 blocks per pad (6 mm), contact = taxel > THR_REL (same threshold on both sides)
  f1_pad      the same with whole pads as units (15);  f1_taxel  1.5 mm cells
  pad_time_f1 per pad, F1 of its in-contact frames predicted vs real, mean over pads in contact
  time_iou    IoU of the frames with total > TOT_REL;  onset_ms / release_ms  |error| of the first / last such frame
  total_r, finger_r   Pearson r of the total / per-finger (6 groups) force course
  share_tv    load-share total variation: per frame where both are in contact, 0.5 * sum |finger share pred - real|
  total_mae_rel, finger_mae_rel   MAE of the total / per-finger relative force (units of S)
Secondary: total_mae_N_oracle = |S * total_pred_rel - real total (N)|, the magnitude with the oracle scale S.
Methods: LoRA runs out/<RUN>p<P>/ (U, G, E, R), regressors out/regr_E|regr_R/, baselines "frame-0 touch repeated" (the real
frame-0 relative touch for every frame; it knows frame 0) and "twin sim touch" (the twin's own simulated touch, relative to
its own scale). The earlier held-out LoRA h2 predicts absolute N on panel v1: its row is scored the OLD absolute way
(0.15 N / 5 N thresholds, 3.0 N cap) and labelled so; it is not comparable to the relative rows beyond the r columns.
Video adherence of G (mug / hand IoU vs the sim silhouettes) comes from labeladh.py.
  envs/armA/bin/python eval_ge.py [--epochs 8]   -> lora_ge/eval/{scores.json, table.md}"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from panel_v3 import F_MAX, PANEL_VERSION, SLICES, THR_REL, TOT_REL, finger_totals  # noqa: E402

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); LG = R / "outputs/R2S2V_mug/lora_ge"; DS = LG / "dataset"; FPS = 16
THR, TOT, CAP = THR_REL, TOT_REL, F_MAX


def units(F, k, thr):
    """(frames, units) contact for blocks of k x k taxels inside each pad (k = 4: 6 mm) or whole pads (multiscale_f1.py)"""
    out = []
    for pid, (a, b, r, c) in SLICES.items():
        blk = (F[:, a:b] > thr).reshape(len(F), r, c)
        if k == "pad":
            out.append(blk.reshape(len(F), -1).any(1)[:, None]); continue
        rr, cc = -(-r // k), -(-c // k); pad = np.zeros((len(F), rr * k, cc * k), bool); pad[:, :r, :c] = blk
        out.append(pad.reshape(len(F), rr, k, cc, k).any((2, 4)).reshape(len(F), -1))
    return np.concatenate(out, 1)


def ms_f1(P, T, k, thr):
    p, t = units(P, k, thr), units(T, k, thr); fr = p.any(1) | t.any(1)
    if not fr.any():
        return float("nan")
    tp, fp, fn = (p & t)[fr].sum(1), (p & ~t)[fr].sum(1), (~p & t)[fr].sum(1)
    return float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1)))


def base_scores(P, T):
    """score_tactile.py's finger / total course scores (inputs already clipped at F_MAX)"""
    ft, fp = np.array([finger_totals(x) for x in T]), np.array([finger_totals(x) for x in P])
    rs = [float(np.corrcoef(ft[:, i], fp[:, i])[0, 1]) for i in range(ft.shape[1]) if ft[:, i].std() > 1e-3 and fp[:, i].std() > 1e-3]
    return dict(finger_r=float(np.mean(rs)) if rs else float("nan"), finger_mae=float(np.abs(fp - ft).mean()),
                total_r=float(np.corrcoef(T.sum(1), P.sum(1))[0, 1]) if P.sum(1).std() > 1e-3 and T.sum(1).std() > 1e-3 else float("nan"),
                total_mae=float(np.abs(P.sum(1) - T.sum(1)).mean()))


def frames_on(x, tot):
    on = np.nonzero(x.sum(1) > tot)[0]; return (int(on[0]), int(on[-1])) if len(on) else (None, None)


def metrics(P, T, S=None, T_N=None, thr=None, tot=None, cap=None):
    """P, T relative (F / S) unless thr / tot / cap say otherwise (the h2 absolute row)"""
    thr, tot, cap = thr or THR, tot or TOT, cap or CAP
    P, T = np.minimum(np.clip(P, 0, None), cap), np.minimum(np.clip(T, 0, None), cap); n = min(len(P), len(T)); P, T = P[:n], T[:n]
    b = base_scores(P, T); out = dict(f1_taxel=ms_f1(P, T, 1, thr), f1_6mm=ms_f1(P, T, 4, thr), f1_pad=ms_f1(P, T, "pad", thr), total_r=b["total_r"],
                                      finger_r=b["finger_r"], total_mae_rel=b["total_mae"], finger_mae_rel=b["finger_mae"])
    pt = []
    for pid, (a, z, _, _) in SLICES.items():
        tp, pp = (T[:, a:z] > thr).any(1), (P[:, a:z] > thr).any(1)
        if tp.any():
            pt.append(2 * (tp & pp).sum() / max(2 * (tp & pp).sum() + (tp & ~pp).sum() + (~tp & pp).sum(), 1))
    out["pad_time_f1"] = float(np.mean(pt)) if pt else float("nan")
    ct, cp = T.sum(1) > tot, P.sum(1) > tot; out["time_iou"] = float((ct & cp).sum() / max((ct | cp).sum(), 1))
    (t0, t1), (p0, p1) = frames_on(T, tot), frames_on(P, tot)
    out["onset_ms"] = abs(p0 - t0) * 1000 / FPS if t0 is not None and p0 is not None else float("nan")
    out["release_ms"] = abs(p1 - t1) * 1000 / FPS if t1 is not None and p1 is not None else float("nan")
    out["missed_contact"] = bool(t0 is not None and p0 is None)
    both = ct & cp
    if both.any():
        fp, ft = np.array([finger_totals(x) for x in P[both]]), np.array([finger_totals(x) for x in T[both]])
        out["share_tv"] = float(np.mean(0.5 * np.abs(fp / fp.sum(1, keepdims=True) - ft / ft.sum(1, keepdims=True)).sum(1)))
    else:
        out["share_tv"] = float("nan")
    if S is not None and T_N is not None:
        out["total_mae_N_oracle"] = float(np.abs(S * P.sum(1) - T_N[:n].sum(1)).mean())
    return {k: (float(v) if not isinstance(v, bool) else v) for k, v in out.items()}


def touch_scale(T):
    tot = T.sum(1); on = tot > 1.0; return float(np.median(tot[on])) if on.any() else float("nan")


KEYS = ["mug_iou", "hand_iou", "f1_6mm", "f1_pad", "pad_time_f1", "time_iou", "onset_ms", "release_ms", "total_r", "finger_r", "share_tv", "total_mae_rel",
        "finger_mae_rel", "total_mae_N_oracle", "f1_taxel"]


def labeladh(model, clip, seed=0):
    """mug / hand IoU vs the sim silhouettes (labeladh.py: outputs/R2S2V_mug/labeladh/<clip>/<model>__seed<k>.json)"""
    p = R / "outputs/R2S2V_mug/labeladh" / clip / (f"{model}__seed{seed}.json" if model != "real" else "real.json")
    if not p.exists():
        return {}
    j = json.loads(p.read_text()); return {k: j[k] for k in ("mug_iou", "hand_iou", "mug_iou_late", "hand_iou_late") if k in j}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", nargs="+", type=int, default=[8]); a = ap.parse_args()
    sys.path.insert(0, str(HERE)); from ge_cases import cases
    rows = {r["id"]: r for r in json.loads((DS / "rows.json").read_text())}; res = []
    methods = [(f"{run}p{p}", LG / "out" / f"{run}p{p}") for p in a.epochs for run in ("U", "G", "E", "R")] + [("regressor R6, real+sim", LG / "out" / "regr_E_best"), ("regressor R6, real only", LG / "out" / "regr_R_best")]   # R0 (L1) collapses to zero (loss_table)
    for c in cases(None):
        r = rows[c["row"]]; z = np.load(DS / r["tactile"]); T, T_N, S = z["touch_rel"], z["touch"], float(z["S"])
        grp = ("G " if c["job"] == "G" else "E ") + ("new participant (actor_01)" if r["actor"] == "actor_01" else "held-out twin, real video" if r["kind"] == "real" else "held-out twin, generated video (sim touch)")
        base = dict(case=c["case"], job=c["job"], ctrl=c["ctrl"], row=c["row"], group=grp, touch0=c.get("touch0"))
        for name, d in methods:
            for f in sorted(d.glob(f"{c['case']}_s*_tactile.npy")):
                seed = int(f.name.split("_s")[-1].split("_")[0]); adh = {}
                if c["job"] == "G":
                    run, p = name[0], name[2:]; adh = labeladh(f"wan22fun-a14b-ge{run}p{p}g{'t' if c['touch0'] else 'n'}", r["clip"], seed)
                res.append(dict(base, method=name, seed=seed, **metrics(np.load(f), T, S, T_N), **adh))
        if c["job"] == "G" and c["touch0"]:                         # references: the earlier held-out LoRA h2 (step 600, start = real
            for ck in (600, 300):                                   # frame 0 + its touch, ref image passed), zero-shot video, real video
                f = R / "outputs/R2S2V_mug/lora_tactile/out" / f"wan22fun-a14b-tacth2c{ck}-real" / f"{r['clip']}_seed0_tactile.npy"
                if f.exists():
                    res.append(dict(base, method=f"h2 step {ck} (earlier LoRA) [ABSOLUTE N scoring, panel v1]", seed=0,
                                    **metrics(np.load(f), T_N, thr=0.15, tot=5.0, cap=3.0), **labeladh(f"wan22fun-a14b-tacth2c{ck}-real", r["clip"]))); break
            res.append(dict(base, method="zero-shot A14B (video only)", seed=0, **labeladh("wan22fun-a14b-head-c5", r["clip"])))
            res.append(dict(base, method="real video (ceiling)", seed=0, **labeladh("real", r["clip"])))
        if c["job"] == "E" and c["ctrl"] in ("da", "sim"):           # baselines once per (row, target), listed under each control
            res.append(dict(base, method="frame-0 touch repeated", seed=0, **metrics(np.repeat(T[:1], len(T), 0), T, S, T_N)))
            st = LG / "simtouch" / f"{r['clip']}.npz"
            if r["kind"] == "real" and st.exists():
                sim = np.load(st)["sim"]; res.append(dict(base, method="twin sim touch (own scale)", seed=0, **metrics(sim / touch_scale(sim), T, S, T_N)))
    out = LG / "eval"; out.mkdir(exist_ok=True); (out / "scores.json").write_text(json.dumps(dict(panel=PANEL_VERSION, cap_rel=CAP, thr_rel=THR, tot_rel=TOT, rows=res), indent=1))
    # tables: mean over clips (and seeds) per group x control x method
    lines = ["| job / clips | control | method | n | " + " | ".join(KEYS) + " |", "|" + "---|" * (len(KEYS) + 4)]
    keyf = lambda x: (x["group"], x["ctrl"], x.get("touch0"), x["method"])
    for k in sorted({keyf(x) for x in res}, key=lambda k: (k[0], k[1], str(k[2]), k[3])):
        xs = [x for x in res if keyf(x) == k]
        vals = [float(np.nanmean([x[m] for x in xs if m in x])) if any(m in x and not np.isnan(x[m]) for x in xs) else float("nan") for m in KEYS]
        ctl = k[1] + ("" if k[2] is None else (" + frame-0 touch" if k[2] else ", no frame-0 touch"))
        lines.append(f"| {k[0]} | {ctl} | {k[3]} | {len({x['row'] for x in xs})} | " + " | ".join("-" if np.isnan(v) else (f"{v:.0f}" if m.endswith("_ms") else f"{v:.3f}") for v, m in zip(vals, KEYS)) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
