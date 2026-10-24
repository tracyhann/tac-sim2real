"""Compare the estimated touch (predict_release.py, pulled to the Mac) with each video's simulated tactile (release bundle,
tactile.npz normal_N, 880 glove channels, N). Runs on numpy + matplotlib only (/usr/bin/python3).
U8 predicts RELATIVE touch (F / S); it is put into N with the clip's true scale S (median simulated total force over the contact
frames, total > 1 N) - the "oracle S" convention of the dual-mode eval. f2 predicts N directly.
Per video and model:
  r_total        Pearson r of the total force over the 81 frames (scale-free)
  mae_total_N    mean |predicted - simulated| total force (N)
  contact_iou    IoU of the contact frames (total > 1 N)
  onset_ms / release_ms   |first / last contact frame difference| (ms)
  share_tv       total-variation distance of the per-part force shares (thumb .. little, palm) over the simulated contact frames
  pad_f1         F1 of pad contact (pad force > 0.5 N) over frames x 15 pads
Writes <out>/scores.json, scores.md (per video + per group mean) and fig_total.png (total force: sim vs U8 vs f2, every video).
  /usr/bin/python3 compare_est.py --pull <dir with U8/ f2/> --bundle <scene_mug runs/clips/r2s2v_handpose> --out <dir>"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; FPS = 16; ON = 1.0; PAD_ON = 0.5
PARTS = ["thumb", "index", "middle", "ring", "little", "palm"]
GROUP = {"r10": "new (r10, other scenes)", "r8b": "new (r8b, box scene / top photo)", "r6": "U training rows (r6 head)"}


def scale(T):
    tot = T.sum(1); on = tot > ON; return float(np.median(tot[on])) if on.any() else float("nan")


def metrics(P, G, pad, names):
    tp, tg = P.sum(1), G.sum(1); r = float(np.corrcoef(tp, tg)[0, 1]) if tp.std() > 1e-9 and tg.std() > 1e-9 else float("nan")
    cp, cg = tp > ON, tg > ON; iou = float((cp & cg).sum() / max((cp | cg).sum(), 1))
    first = lambda c: int(np.argmax(c)) if c.any() else None; last = lambda c: int(len(c) - 1 - np.argmax(c[::-1])) if c.any() else None
    ms = lambda a, b: None if a is None or b is None else round(abs(a - b) * 1000 / FPS)
    part = {p: np.isin(pad, [i for i, n in names.items() if n.startswith(p)]) for p in PARTS}
    sh = lambda T: np.array([T[cg][:, part[p]].sum() for p in PARTS]) / max(T[cg].sum(), 1e-9)
    tv = float(0.5 * np.abs(sh(P) - sh(G)).sum()) if cg.any() and P[cg].sum() > 0 else float("nan")
    ids = sorted(names); pp = np.stack([P[:, pad == i].sum(1) for i in ids], 1) > PAD_ON; pg = np.stack([G[:, pad == i].sum(1) for i in ids], 1) > PAD_ON
    tpos = (pp & pg).sum(); f1 = float(2 * tpos / max(pp.sum() + pg.sum(), 1))
    return dict(r_total=round(r, 3), mae_total_N=round(float(np.abs(tp - tg).mean()), 2), contact_iou=round(iou, 3), onset_ms=ms(first(cp), first(cg)),
                release_ms=ms(last(cp), last(cg)), share_tv=round(tv, 3), pad_f1=round(f1, 3))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pull", required=True); ap.add_argument("--bundle", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["U8"])   # f2 is not scored: under the full-video mask it paints scene into the panel, no touch map
    a = ap.parse_args(); pull, bun, out = Path(a.pull), Path(a.bundle), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((HERE / "cases.json").read_text()); rows = []; curves = {}
    for c in cases:
        t = np.load(bun / c["bundle"] / "tactile.npz"); G = t["normal_N"].astype(np.float64); pad = t["taxel_pad"]
        names = dict(zip(t["pad_ids"].tolist(), [str(n) for n in t["pad_names"]])); S = scale(G); curves[c["id"]] = {"sim": G.sum(1)}
        for m in a.models:
            f = pull / m / f"{c['id']}_tactile.npy"
            if not f.exists():
                continue
            P = np.load(f).astype(np.float64); P = P * S if m == "U8" else P; curves[c["id"]][m] = P.sum(1)
            js = json.loads((pull / m / f"{c['id']}.json").read_text())
            rows.append(dict(id=c["id"], group=c["group"], model=m, S_N=round(S, 1), known_mae=js.get("known_mae"), **metrics(P, G, pad, names)))
    (out / "scores.json").write_text(json.dumps(rows, indent=1))
    keys = ["r_total", "mae_total_N", "contact_iou", "onset_ms", "release_ms", "share_tv", "pad_f1"]
    md = ["| group | model | n | " + " | ".join(keys) + " |", "|---|---|---|" + "---|" * len(keys)]
    for g in GROUP:
        for m in a.models:
            rs = [r for r in rows if r["group"] == g and r["model"] == m]
            if rs:
                mean = lambda k: np.nanmean([np.nan if r[k] is None else r[k] for r in rs])
                md.append(f"| {GROUP[g]} | {m} | {len(rs)} | " + " | ".join(f"{mean(k):.2f}" if "ms" not in k else f"{mean(k):.0f}" for k in keys) + " |")
    md += ["", "| video | model | S (N) | known-part MAE | " + " | ".join(keys) + " |", "|---|---|---|---|" + "---|" * len(keys)]
    md += [f"| {r['id']} | {r['model']} | {r['S_N']} | {r['known_mae']} | " + " | ".join(str(r[k]) for k in keys) + " |" for r in sorted(rows, key=lambda r: (r["group"], r["id"], r["model"]))]
    (out / "scores.md").write_text("\n".join(md) + "\n"); print("\n".join(md[:2 + 2 * len(GROUP)]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = [c["id"] for c in cases if len(curves.get(c["id"], {})) > 1]
    if ids:
        nc = 5; nr = -(-len(ids) // nc); fig, ax = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.4 * nr), squeeze=False, sharex=True)
        tt = np.arange(81) / FPS
        for q, i in enumerate(ids):
            x = ax[q // nc, q % nc]; cv = curves[i]
            x.plot(tt, cv["sim"], c="k", lw=1.6, label="simulated")
            for m, col in (("U8", "#d1495b"), ("f2", "#2e86ab")):
                if m in cv:
                    x.plot(tt, cv[m], c=col, lw=1.2, label=m + (" (x S)" if m == "U8" else ""))
            x.set_title(i, fontsize=7.5); x.grid(alpha=0.3); x.tick_params(labelsize=7)
        for q in range(len(ids), nr * nc):
            ax[q // nc, q % nc].axis("off")
        ax[0, 0].legend(fontsize=7); fig.supxlabel("video time (s)", fontsize=9); fig.supylabel("total normal force (N)", fontsize=9)
        fig.tight_layout(); fig.savefig(out / "fig_total.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
