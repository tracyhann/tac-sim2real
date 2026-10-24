"""Loss comparison on the estimate job's regression head (user 2026-10-23 via the parent; project page Methods tab, sections
5-6): frozen Wan VAE latents of the 832 x 480 video -> the same 3D-conv + temporal trunk for every variant -> 880 taxels per
frame, panel v3 RELATIVE touch. Per-taxel losses act on u = (F / S) / CAP (CAP = panel v3 taxel cap 0.11, so u in [0, 1]).
  R0  L1 on relative force (baseline)
  R1  Huber(beta 1) + 1/2 L1, active taxels (u > thr) x10                                   (TouchSight)
  R2  R0 + lam * [(1 - Pearson of each finger's time curve) + (1 - Pearson of each frame's spatial map)]   (FSGS)
  R3  R0 + lam * [temporal gradient matching (VDA L_TGM: |d_t pred - d_t gt| where |d_t gt| < 0.05) + spatial gradient
      L1 inside each pad (Depth Pro style, rows and columns)]
  R4  lam * BCE(contact) + L1 of the contact-gated output u = p(contact) * force               (HOPE / EgoTac)
  R5  16 log bins (bin 0 = no contact), cross-entropy + lam * expected bin distance; prediction = argmax bin centre
                                                                                          (PressureVision++ / EgoPressure)
  R6  L1 on the relative total a(t) + lam * Sinkhorn OT between predicted and real per-frame load share p(t) over 6 mm units
      (glove-layout distances); output F / S = a * p                                       (DM-Count; F = S a p)
Same data (dataset train rows: 40 real + 39 sim), split, optimiser and steps (150 epochs, batch 4, AdamW 3e-4) for all.
lam is chosen per variant on an inner split by participant (actor_10's 8 training clips held out, the rest trained), by a
scale-free composite (F1@6mm + contact-time IoU + total r - load-share TV); then 3 seeds train on all training rows and
are scored on the held-out cases (eval_ge.metrics, relative, + newton MAE with the oracle S).
Also writes the discriminative baselines of the main table: out/regr_E (R0, real + sim rows) and out/regr_R (R0, real rows
only), seed 0, and out/regr_E_best / regr_R_best (the variant with the best inner-split composite).
  envs/videoxfun/bin/python loss_table.py   -> lora_ge/eval/loss_table.{json,md}, out/lt_<V>_s<k>/"""
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from eval_ge import metrics  # noqa: E402
from ge_cases import cases  # noqa: E402
from panel_v3 import F_MAX as CAP, FINGER, SLICES, THR_REL  # noqa: E402
from regress_ge import DS, LG, latents  # noqa: E402

THR_U = THR_REL / CAP; K = 16; EPOCHS, BS, LR = 150, 4, 3e-4; SEEDS = (0, 1, 2)
# sensitivity (2026-10-23): R0 and R3 collapsed with the sigmoid^2 output (zero gradient at 0); "sp" = the same loss on a
# softplus output (bias init -4, u = softplus(z)), to tell the loss from the parameterisation
EXTRA = {"R0sp": [None], "R3sp": [0.1, 1.0, 10.0]}
GRID = {"R0": [None], "R1": [None], "R2": [0.01, 0.1, 1.0], "R3": [0.1, 1.0, 10.0], "R4": [0.1, 1.0, 10.0], "R5": [0.1, 1.0, 10.0], "R6": [0.01, 0.1, 1.0]}
OUT_DIM = {"R0": 880, "R1": 880, "R2": 880, "R3": 880, "R4": 1760, "R5": 880 * K, "R6": 881, "R0sp": 880, "R3sp": 880}
base = lambda v: v[:-2] if v.endswith("sp") else v
dev = __import__("os").environ.get("GE_DEV", "cuda")


# ------------------------------------------------------------------ geometry helpers
def finger_matrix():
    G = torch.zeros(880, len(FINGER))
    for j, pids in enumerate(FINGER.values()):
        for q in pids:
            a, b, _, _ = SLICES[q]; G[a:b, j] = 1
    return G


def unit_matrix_and_cost():
    """6 mm units (4 x 4 taxel blocks per pad) -> pooling matrix (880, n) and normalised centroid distances (n, n)"""
    g = json.loads(Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/data/TouchScale/metadata/glove_geometry_left.json").read_text())
    P = np.frombuffer(base64.b64decode(g["data_base64"]), "<f4").reshape(-1, g["point_stride"])[:, :3]; cols = []
    for pid, (a, b, r, c) in SLICES.items():
        idx = np.arange(a, b).reshape(r, c)
        for i in range(0, r, 4):
            for j in range(0, c, 4):
                cols.append(idx[i:i + 4, j:j + 4].ravel())
    U = np.zeros((880, len(cols)), np.float32)
    for k, ix in enumerate(cols):
        U[ix, k] = 1
    cen = np.stack([P[ix].mean(0) for ix in cols]); C = np.linalg.norm(cen[:, None] - cen[None], axis=-1); C /= C.max()
    return torch.from_numpy(U), torch.from_numpy(C.astype(np.float32))


GM = finger_matrix().to(dev); UM, CM = (x.to(dev) for x in unit_matrix_and_cost())
BIN_LO, BIN_HI = np.log(THR_U), 0.0
CENTRES = torch.tensor([0.0] + [float(np.exp(BIN_LO + (j - 0.5) / (K - 1) * (BIN_HI - BIN_LO))) for j in range(1, K)], device=dev)


def to_bins(g):
    lg = torch.log(g.clamp(min=1e-8)); j = 1 + torch.floor((lg - BIN_LO) / (BIN_HI - BIN_LO) * (K - 1)).long()
    return torch.where(g < THR_U, torch.zeros_like(j), j.clamp(1, K - 1))


# ------------------------------------------------------------------ model
class Net(nn.Module):
    def __init__(self, out_dim, c=16, p=0.2, sp=False):
        super().__init__()
        ch = [c, 64, 128, 192, 256, 256]; L = []
        for i in range(5):
            L += [nn.Conv3d(ch[i], ch[i + 1], 3, stride=(1, 1, 1) if i == 0 else (1, 2, 2), padding=1), nn.GroupNorm(8, ch[i + 1]), nn.GELU()]
        self.enc = nn.Sequential(*L); self.pool = nn.AdaptiveAvgPool3d((None, 4, 7)); self.drop = nn.Dropout(p); self.od = out_dim
        self.fc = nn.Linear(256 * 28, 512); self.tconv = nn.Conv1d(512, 512, 3, padding=1); self.out = nn.Linear(512, 4 * out_dim)
        if sp:
            nn.init.constant_(self.out.bias, -4.0)

    def forward(self, z):                                          # (b, 16, 21, 60, 104) -> (b, 81, out_dim) raw
        h = self.pool(self.enc(z)); b, c, t = h.shape[:3]; h = h.permute(0, 2, 1, 3, 4).reshape(b, t, -1)
        h = torch.relu(self.fc(self.drop(h))); h = h + torch.relu(self.tconv(h.transpose(1, 2))).transpose(1, 2)
        y = self.out(self.drop(h)).reshape(b, t, 4, self.od)
        return torch.cat([y[:, :1, 0], y[:, 1:].reshape(b, -1, self.od)], 1)


def decode(v, y):
    """raw output -> u (b, 81, 880) in panel-cap units"""
    if v.endswith("sp"):
        return Fn.softplus(y)
    if v in ("R0", "R1", "R2", "R3"):
        return torch.sigmoid(y) ** 2
    if v == "R4":
        return torch.sigmoid(y[..., :880]) * torch.sigmoid(y[..., 880:]) ** 2
    if v == "R5":
        return CENTRES[y.reshape(*y.shape[:2], 880, K).argmax(-1)]
    if v == "R6":
        return Fn.softplus(y[..., :1]) * torch.softmax(y[..., 1:], -1) / CAP


def pearson(a, b, dim, eps=1e-6):
    a, b = a - a.mean(dim, keepdim=True), b - b.mean(dim, keepdim=True)
    return (a * b).sum(dim) / (a.norm(dim=dim) * b.norm(dim=dim) + eps)


def sinkhorn(p, q, C, eps=0.05, iters=60):
    """entropic OT cost between histograms p, q (n_pairs, n) with cost C (n, n), log domain"""
    logK = -C / eps; f = torch.zeros_like(p); g = torch.zeros_like(q); lp, lq = torch.log(p + 1e-9), torch.log(q + 1e-9)
    for _ in range(iters):
        f = eps * (lp - torch.logsumexp((g[:, None, :] + eps * logK[None]) / eps, -1))
        g = eps * (lq - torch.logsumexp((f[:, :, None] + eps * logK[None]) / eps, -2))
    pi = torch.exp((f[:, :, None] + g[:, None, :]) / eps + logK[None]); return (pi * C[None]).sum((1, 2))


def loss_fn(v, y, g, lam):
    u = decode(v, y); v = base(v)
    if v == "R5":
        lg = y.reshape(*y.shape[:2], 880, K); tgt = to_bins(g); ce = Fn.cross_entropy(lg.reshape(-1, K), tgt.reshape(-1))
        pr = torch.softmax(lg, -1); dist = (pr * (torch.arange(K, device=dev)[None, None, None] - tgt[..., None]).abs().float()).sum(-1).mean() / (K - 1)
        return ce + lam * dist
    if v == "R6":
        a, q = Fn.softplus(y[..., 0]), torch.softmax(y[..., 1:], -1); ag = g.sum(-1) * CAP
        l = (a - ag).abs().mean(); on = ag > 0.196
        if on.any():
            qu, gu = q[on] @ UM, (g[on] / g[on].sum(-1, keepdim=True)) @ UM
            l = l + lam * sinkhorn(qu, gu, CM).mean()
        return l
    if v == "R4":
        bce = Fn.binary_cross_entropy_with_logits(y[..., :880], (g > THR_U).float())
        return lam * bce + (u - g).abs().mean()
    l1 = (u - g).abs()
    if v == "R1":
        w = 1 + 9 * (g > THR_U).float(); return (w * (Fn.smooth_l1_loss(u, g, beta=1.0, reduction="none") + 0.5 * l1)).sum() / w.sum()
    l = l1.mean()
    if v == "R2":
        ft, fg = u @ GM, g @ GM; m = fg.std(1) > 1e-4                       # (b, 81, 6) finger curves
        lt = (1 - pearson(ft, fg, 1))[m].mean() if m.any() else 0 * l
        on = g.sum(-1) > 0; ls = (1 - pearson(u, g, -1))[on].mean() if on.any() else 0 * l
        l = l + lam * (lt + ls)
    if v == "R3":
        dp, dg = u[:, 1:] - u[:, :-1], g[:, 1:] - g[:, :-1]; m = (dg.abs() < 0.05).float()
        tgm = ((dp - dg).abs() * m).sum() / m.sum().clamp(min=1); sg = 0
        for pid, (a, b_, r, c) in SLICES.items():
            P_, G_ = u[..., a:b_].reshape(*u.shape[:2], r, c), g[..., a:b_].reshape(*g.shape[:2], r, c)
            sg = sg + ((P_[..., 1:, :] - P_[..., :-1, :]) - (G_[..., 1:, :] - G_[..., :-1, :])).abs().mean() + ((P_[..., 1:] - P_[..., :-1]) - (G_[..., 1:] - G_[..., :-1])).abs().mean()
        l = l + lam * (tgm + sg / len(SLICES))
    return l


# ------------------------------------------------------------------ train / eval
def target(r):
    return torch.from_numpy(np.minimum(np.clip(np.load(DS / r["tactile"])["touch_rel"], 0, None) / CAP, 1.0)).float()


def fit(v, lam, rows, lat, seed):
    torch.manual_seed(seed); np.random.seed(seed); m = Net(OUT_DIM[v], sp=v.endswith("sp")).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.05)
    Z = torch.stack([lat[r["id"]] for r in rows]); Y = torch.stack([target(r) for r in rows]); gen = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(rows), generator=gen)
        for i in range(0, len(rows), BS):
            j = perm[i:i + BS]; z = Z[j].to(dev); z = z + 0.05 * torch.randn_like(z)
            loss = loss_fn(v, m(z), Y[j].to(dev), lam); opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def predict(v, m, z):
    m.eval(); return decode(v, m(z[None].to(dev)))[0].clamp(0, 1).cpu().numpy() * CAP          # relative force F / S


def score(v, m, rows, lat):
    out = []
    for r in rows:
        z = np.load(DS / r["tactile"]); out.append(metrics(predict(v, m, lat[r["id"]]), z["touch_rel"], float(z["S"]), z["touch"]))
    return out


def composite(s):
    f = lambda k, d: float(np.nanmean([x[k] for x in s])) if not all(np.isnan(x[k]) for x in s) else d
    return f("f1_6mm", 0) + f("time_iou", 0) + f("total_r", 0) - f("share_tv", 1)


KEYS = ["total_r", "finger_r", "total_mae_rel", "finger_mae_rel", "f1_6mm", "f1_pad", "share_tv", "onset_ms", "time_iou", "total_mae_N_oracle"]


def main():
    extra = "--extra" in sys.argv; grid = EXTRA if extra else GRID; tag = "loss_table_sp" if extra else "loss_table"
    rows = json.loads((DS / "rows.json").read_text()); lat = latents(rows); byid = {r["id"]: r for r in rows}
    train = [r for r in rows if r["split"] == "train"]; real = [r for r in train if r["kind"] == "real"]
    itr, iva = [r for r in train if r["actor"] != "actor_10"], [r for r in train if r["actor"] == "actor_10"]
    E = [c for c in cases("E")]; res = dict(inner={}, chosen={}, test={}, settings=dict(epochs=EPOCHS, batch=BS, lr=LR, seeds=SEEDS, K=K, cap=CAP, thr_u=THR_U))
    for v, lams in grid.items():                                   # 1. choose lam on the inner split (participant actor_10 held out)
        best = None
        for lam in lams:
            t = time.time(); s = score(v, fit(v, lam or 0, itr, lat, 0), iva, lat); c = composite(s)
            res["inner"][f"{v} lam={lam}"] = dict(composite=c, seconds=round(time.time() - t), **{k: float(np.nanmean([x[k] for x in s])) for k in KEYS})
            print("inner", v, lam, round(c, 3), f"{time.time() - t:.0f}s", flush=True)
            if best is None or c > best[1]:
                best = (lam, c)
        res["chosen"][v] = dict(lam=best[0], inner_composite=best[1])
    for v in grid:                                                 # 2. three seeds on all training rows, scored on the held-out cases
        lam = res["chosen"][v]["lam"] or 0
        for seed in SEEDS:
            m = fit(v, lam, train, lat, seed); od = LG / "out" / f"lt_{v}_s{seed}"; od.mkdir(parents=True, exist_ok=True); per = []
            for c in E:
                r = byid[c["row"]]; P = predict(v, m, lat[r["id"]]); np.save(od / f"{c['case']}_s0_tactile.npy", P)
                z = np.load(DS / r["tactile"]); per.append(dict(case=c["case"], ctrl=c["ctrl"], actor=r["actor"], kind=r["kind"], **metrics(P, z["touch_rel"], float(z["S"]), z["touch"])))
            res["test"][f"{v} s{seed}"] = per; print("test", v, seed, flush=True)
    order = sorted(grid, key=lambda v: -res["chosen"][v]["inner_composite"]); best = order[0]
    for name, v, tr in [] if extra else (("regr_E", "R0", train), ("regr_R", "R0", real), ("regr_E_best", best, train), ("regr_R_best", best, real)):   # main-table baselines
        m = fit(v, res["chosen"][v]["lam"] or 0, tr, lat, 0); od = LG / "out" / name; od.mkdir(parents=True, exist_ok=True)
        for c in E:
            np.save(od / f"{c['case']}_s0_tactile.npy", predict(v, m, lat[c["row"]]))
        print("baseline", name, v, flush=True)
    res["inner_best"] = best; (LG / "eval").mkdir(exist_ok=True); (LG / f"eval/{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    # table: held-out groups (one prediction per row; E cases with the sim control duplicate the twin rows -> keep the da case)
    groups = {"new participant actor_01 (8 real)": lambda x: x["actor"] == "actor_01" and x["ctrl"] == "da",
              "held-out twins, real video (2)": lambda x: x["actor"] == "actor_09" and x["kind"] == "real" and x["ctrl"] == "da",
              "held-out twins, generated video, sim touch (2)": lambda x: x["kind"] == "sim"}
    lines = []
    for gname, sel in groups.items():
        lines += [f"\n**{gname}** (mean over clips, then mean ± sd over 3 seeds)\n", "| loss | lam | " + " | ".join(KEYS) + " |", "|" + "---|" * (len(KEYS) + 2)]
        for v in grid:
            per_seed = []
            for seed in SEEDS:
                xs = [x for x in res["test"][f"{v} s{seed}"] if sel(x)]
                per_seed.append([float(np.nanmean([x[k] for x in xs])) for k in KEYS])
            A = np.array(per_seed); mu, sd = np.nanmean(A, 0), np.nanstd(A, 0)
            lines.append(f"| {v} | {res['chosen'][v]['lam']} | " + " | ".join(f"{a:.0f}±{b:.0f}" if k.endswith("_ms") else f"{a:.3f}±{b:.3f}" for a, b, k in zip(mu, sd, KEYS)) + " |")
    (LG / f"eval/{tag}.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
