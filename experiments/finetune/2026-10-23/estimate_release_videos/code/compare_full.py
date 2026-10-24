"""Full scoring of the touch estimated on the release videos, with the SAME metric set as the dual-mode held-out eval
(dual_mode_lora/code/eval_ge.metrics, relative touch F / S, panel v3 caps and thresholds), plus baselines and a perturbation
score that only these counterfactual videos allow. compare_est.py (first pass) used its own reduced metrics; this replaces it.
Reference = each video's simulated tactile (release bundle tactile.npz normal_N, 880 glove channels), relative with its own
S = median total over frames with total > 1 N (eval_ge.touch_scale).
Methods: U8, E8, R8 (dual-mode runs at epoch 8, job E, relative), R6 regressor seeds 0-2 (relative; regr_release.py), f2 (N, divided by the oracle S; its panel holds
scene, see README), baselines "frame-0 sim touch repeated" (knows frame 0, as in eval_ge) and "recorded replay's touch" (the
unperturbed sim's touch for every video = ignores the perturbation; the 3 recorded videos are left out of that row).
eval_ge keys: f1_6mm f1_pad f1_taxel pad_time_f1 time_iou onset_ms release_ms total_r finger_r share_tv total_mae_rel
finger_mae_rel total_mae_N_oracle (+ missed_contact). Added here:
  delta_r     Pearson r of the CHANGE in relative total force caused by the perturbation, predicted vs simulated
              (video minus the recorded video of the same view; 15 videos: r6 head, r8b head, r8b top photo x 5 perturbations)
  delta_gain  |predicted change| / |simulated change| (L2 over frames; 1 = right size, 0 = no response)
  known_mae   mean |output video - given video| in 0-255 pixel units (the E job must return the given video unchanged)
  /usr/bin/python3 compare_full.py --pull <dir with U8/ f2/ R6_s*/> --bundle <scene_mug runs/clips/r2s2v_handpose> --out <dir>"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; GE = HERE.parents[2] / "2026-10-22/dual_mode_lora/code"; sys.path.insert(0, str(GE))
from eval_ge import metrics, touch_scale  # noqa: E402

KEYS = ["f1_6mm", "f1_pad", "pad_time_f1", "time_iou", "onset_ms", "release_ms", "total_r", "finger_r", "share_tv", "total_mae_rel",
        "finger_mae_rel", "total_mae_N_oracle", "f1_taxel", "delta_r", "delta_gain", "known_mae"]
GROUPS = [("new: r10, other scenes", lambda c: c["group"] == "r10"), ("new: r8b, box scene / top photo", lambda c: c["group"] == "r8b"),
          ("all new (r10 + r8b)", lambda c: c["group"] in ("r10", "r8b")), ("r6 head = U8 / R6 training rows", lambda c: c["group"] == "r6")]
BASE0, BASEREC = "frame-0 sim touch repeated", "recorded replay's touch"


def view(c):                                    # recorded partner of a perturbation video: same group and view
    g, rest = c["id"].split("_", 1); v = rest.split("_")[0]; return f"{g}_{v}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pull", required=True); ap.add_argument("--bundle", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); pull, bun, out = Path(a.pull), Path(a.bundle), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((HERE / "cases.json").read_text()); byid = {c["id"]: c for c in cases}
    G = {c["id"]: np.load(bun / c["bundle"] / "tactile.npz")["normal_N"].astype(np.float64) for c in cases}
    S = {i: touch_scale(g) for i, g in G.items()}; Grel = {i: g / S[i] for i, g in G.items()}
    rec = {view(c): c["id"] for c in cases if c["id"].endswith("_recorded")}; rec_touch = Grel[next(iter(rec.values()))]
    seeds = sorted(p.name for p in pull.glob("R6_s*") if p.is_dir())
    methods = {"U8": ("U8", 1), "E8 (estimate only, real + sim)": ("E8", 1), "R8 (estimate only, real)": ("R8", 1), "E4 (E at R8's 320 steps)": ("E4", 1), "f2 (N / oracle S)": ("f2", 0)} \
        | {f"R6 regressor {s[3:]}": (s, 1) for s in seeds}
    P = {}                                       # (method, id) -> relative prediction
    for name, (d, rel) in methods.items():
        for c in cases:
            f = pull / d / f"{c['id']}_tactile.npy"
            if f.exists():
                x = np.load(f).astype(np.float64); P[name, c["id"]] = x if rel else x / S[c["id"]]
    for c in cases:
        P[BASE0, c["id"]] = np.repeat(Grel[c["id"]][:1], len(Grel[c["id"]]), 0)
        if not c["id"].endswith("_recorded"):
            P[BASEREC, c["id"]] = rec_touch
    rows = []
    for (m, i), x in P.items():
        c = byid[i]; r = dict(id=i, group=c["group"], method=m, S_N=round(S[i], 1), **metrics(x, Grel[i], S[i], G[i]))
        partner = rec.get(view(c))
        if partner and partner != i and (m, partner) in P and m in methods:     # models only: a baseline's change is not a prediction
            dp = x.sum(1) - P[m, partner].sum(1); dg = Grel[i].sum(1) - Grel[partner].sum(1)
            r["delta_r"] = float(np.corrcoef(dp, dg)[0, 1]) if dp.std() > 1e-6 and dg.std() > 1e-6 else float("nan")
            r["delta_gain"] = float(np.linalg.norm(dp) / max(np.linalg.norm(dg), 1e-9))
        else:
            r["delta_r"] = r["delta_gain"] = float("nan")
        js = pull / methods.get(m, ("-", 0))[0] / f"{i}.json"
        r["known_mae"] = float(json.loads(js.read_text()).get("known_mae") or np.nan) if js.exists() else float("nan")
        rows.append(r)
    (out / "scores_full.json").write_text(json.dumps(dict(keys=KEYS, rows=rows), indent=1, default=float))
    order = list(methods) + [BASE0, BASEREC]; fmt = lambda v, k: "-" if not np.isfinite(v) else (f"{v:.0f}" if k.endswith("_ms") else f"{v:.1f}" if k in ("total_mae_N_oracle", "known_mae") else f"{v:.2f}")
    md = ["Mean over videos (R6: mean ± sd of the 3 seed means; relative touch, panel v3 thresholds; total_mae_N_oracle in N with the true S).", ""]
    gmean = lambda xs, k: float(np.nanmean([x[k] for x in xs])) if any(np.isfinite(x[k]) for x in xs) else float("nan")
    r6 = [m for m in order if m.startswith("R6 regressor")]
    for gname, sel in GROUPS:
        md += [f"**{gname}**", "", "| method | n | " + " | ".join(KEYS) + " |", "|---|---|" + "---|" * len(KEYS)]
        for m in [x for x in order if x not in r6] + (["R6"] if r6 else []):
            ms = r6 if m == "R6" else [m]; per = [[r for r in rows if r["method"] == q and sel(byid[r["id"]])] for q in ms]; per = [xs for xs in per if xs]
            if per:
                A = np.array([[gmean(xs, k) for k in KEYS] for xs in per]); mu = np.nanmean(A, 0) if np.isfinite(A).any() else A[0]
                cell = lambda j, k: fmt(mu[j], k) + (f" ± {fmt(np.nanstd(A[:, j]), k)}" if len(per) > 1 and np.isfinite(mu[j]) else "")
                miss = sum(x["missed_contact"] for x in per[0])
                name = f"R6 regressor (seeds {len(per)})" if m == "R6" else m
                md.append(f"| {name}{f' (missed contact {miss})' if miss else ''} | {len(per[0])} | " + " | ".join(cell(j, k) for j, k in enumerate(KEYS)) + " |")
        md.append("")
    md += ["**Per video**", "", "| video | method | S (N) | " + " | ".join(KEYS) + " |", "|---|---|---|" + "---|" * len(KEYS)]
    md += [f"| {r['id']} | {r['method']} | {r['S_N']} | " + " | ".join(fmt(r[k], k) for k in KEYS) + " |" for r in sorted(rows, key=lambda r: (r["group"], r["id"], order.index(r["method"])))]
    (out / "scores_full.md").write_text("\n".join(md) + "\n"); print("\n".join(md[: md.index("**Per video**")]))
    import matplotlib                                   # total relative force per video: sim vs each model (R6 seed 0) vs recorded replay
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [("U8", "#d1495b"), ("E8 (estimate only, real + sim)", "#edae49"), ("R8 (estimate only, real)", "#2e86ab"), ("R6 regressor s0", "#3a9d44")]
    nc = 5; nr = -(-len(cases) // nc); fig, ax = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.4 * nr), squeeze=False, sharex=True); tt = np.arange(81) / 16
    for q, c in enumerate(cases):
        x = ax[q // nc, q % nc]; x.plot(tt, Grel[c["id"]].sum(1), c="k", lw=1.8, label="simulated")
        if (BASEREC, c["id"]) in P:
            x.plot(tt, P[BASEREC, c["id"]].sum(1), c="#999", lw=1, ls="--", label="recorded replay")
        for m, col in show:
            if (m, c["id"]) in P:
                x.plot(tt, P[m, c["id"]].sum(1), c=col, lw=1.1, label=m.split(" (")[0].replace(" regressor s0", " (s0)"))
        x.set_title(c["id"], fontsize=7.5); x.grid(alpha=0.3); x.tick_params(labelsize=7)
    for q in range(len(cases), nr * nc):
        ax[q // nc, q % nc].axis("off")
    ax[0, 1].legend(fontsize=6.5); fig.supxlabel("video time (s)", fontsize=9); fig.supylabel("total force / S (relative)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_total_full.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
