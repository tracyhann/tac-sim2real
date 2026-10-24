"""Per-episode touch scale S for the relative touch panel v3 (user 2026-10-22: touch must be relative, not absolute
newtons; "250 g and 300 g mugs look the same in RGB; after normalisation the relative force relations stay about the same").
  real  S = median total hand force (N, the touch() convention: above each taxel's episode baseline, max over +-34 ms)
        over the contact frames (total > 1 N) of the WHOLE episode (every head frame), so every window of one episode
        shares one S
  sim   S = the same over the replay (the replays span the 5 s window: their 81 canvas-time frames are the whole episode)
Also measures the relative distribution (F / S) to choose panel v3's caps: taxel cap and bar cap such that < 0.5 % of
contact readings and < 0.5 % of frames clip (contact reading: F / S > THR_REL on that taxel).
  envs/armA/bin/python ge_scale.py   -> dataset/scale.json (S per row, per actor / object summary, chosen caps)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ge_data import DS, LG, Ep  # noqa: E402
from panel_v2 import finger_totals  # noqa: E402

CONTACT_N = 1.0


def episode_scale(ep_id):
    e = Ep(ep_id); T, _ = e.touch(e.th); tot = T.sum(1); on = tot > CONTACT_N
    return float(np.median(tot[on])), int(on.sum()), len(tot)


def sim_scale(T):
    tot = T.sum(1); on = tot > CONTACT_N; return float(np.median(tot[on])), int(on.sum()), len(tot)


def nice_up(x):
    for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        if x <= 20 * s:
            return float(np.ceil(x / s) * s)
    return float(np.ceil(x))


def main(thr_rel=None):
    rows = json.loads((DS / "rows.json").read_text()); out = {}
    for r in rows:
        T = np.load(DS / r["tactile"])["touch"]
        s, n_on, n = episode_scale(r["episode"]) if r["kind"] == "real" else sim_scale(T)
        out[r["id"]] = dict(S=s, contact_frames=n_on, frames=n, kind=r["kind"], split=r["split"], actor=r["actor"], task=r["task"])
    Ss = np.array([v["S"] for v in out.values() if v["kind"] == "real"])
    thr = thr_rel if thr_rel else float(np.round(0.15 / np.median(Ss), 4))         # 0.15 N at the median real episode scale
    rel_on, fr_max = [], []
    for r in rows:
        T = np.load(DS / r["tactile"])["touch"] / out[r["id"]]["S"]; on = T > thr
        rel_on.append(T[on]); fr_max.append(np.array([finger_totals(x) for x in T]).max(1))
    rel_on, fr_max = np.concatenate(rel_on), np.concatenate(fr_max)
    tax_cap, bar_cap = nice_up(float(np.percentile(rel_on, 99.5))), nice_up(float(np.percentile(fr_max, 99.5)))
    summ = {}
    for key in ("actor", "task"):
        for g in sorted({v[key] for v in out.values()}):
            x = np.array([v["S"] for v in out.values() if v[key] == g])
            summ[f"{key}: {g}"] = dict(n=len(x), median_S=round(float(np.median(x)), 2), min_S=round(float(x.min()), 2), max_S=round(float(x.max()), 2),
                                       cv=round(float(x.std() / x.mean()), 3))
    res = dict(rule="S = median total force over frames with total > 1 N, whole episode (real) / whole replay (sim)", thr_rel=thr,
               taxel_cap_rel=tax_cap, bar_cap_rel=bar_cap, clip_taxel=float((rel_on > tax_cap).mean()), clip_frames=float((fr_max > bar_cap).mean()),
               rel_contact_p99_5=float(np.percentile(rel_on, 99.5)), finger_rel_p99_5=float(np.percentile(fr_max, 99.5)),
               S_real=dict(median=float(np.median(Ss)), min=float(Ss.min()), max=float(Ss.max()), cv=float(Ss.std() / Ss.mean())), groups=summ, rows=out)
    (DS / "scale.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
