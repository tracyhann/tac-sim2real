"""[Trimmed copy for the HF25 scorer: the metric functions only; the original lives in the authors' tree.]
Scores and tables for the dual-mode LoRA (held-out clips only), on RELATIVE touch (panel v3: F / S, S = the episode's
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
