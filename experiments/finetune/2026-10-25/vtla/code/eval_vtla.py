"""Open-loop evaluation of a VTLA checkpoint on dataset rows (no robot: predicted action chunks against the simulation's).
Per row, start frames t = 0, stride, ...: sample the chunk (10 Euler steps, fixed noise seed), undo the normalisation and the
deltas, compare over the chunk's real steps with the target of the checkpoint's arm:
  wrist_mm         wrist position error (mm)                 wrist_deg   wrist rotation error (deg, rot6d -> Gram-Schmidt)
  joints_deg       mean absolute error of the 22 joint slots (measured joints for VTS, commanded joints for VTF)
  slot9            |error| of dim 9 (closure in deg for VTS, grip force in N for VTF)
  hold             the same errors for the "repeat the current action" baseline (VTS: current state; VTF: current command)
Latent probes (the paper's): z_top1 = top-1 retrieval of the matching future-touch target z* among the evaluated starts that have
a touch change (chance = 1 / n); z_shift_touch / z_shift_image = mean (1 - cos) between z of a sample and z after swapping its
touch input / its image with another sample's (the paper: touch moves z ~0.9, RGB + prompt <= 0.2).
  python eval_vtla.py --ckpt <out>/final --data <dataset> [--ids a b] [--stride 8] [--out eval.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import vtla_data as D
import vtla_model as M


def rot(r6):
    a, b = r6[..., :3], r6[..., 3:6]
    x = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b - (x * b).sum(-1, keepdims=True) * x; y = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], -1)


def ang_deg(Ra, Rb):
    c = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def errors(pred, gt, valid, arm):
    v = valid
    e = dict(wrist_mm=float(np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1)[v].mean() * 1e3),
             wrist_deg=float(ang_deg(rot(pred[..., 3:9]), rot(gt[..., 3:9]))[v].mean()),
             joints_deg=float(np.degrees(np.abs(pred[..., 10:] - gt[..., 10:]).mean(-1))[v].mean()))
    d9 = np.abs(pred[..., 9] - gt[..., 9])[v]; e["slot9"] = float(np.degrees(d9).mean() if arm == "vts" else d9.mean())
    return e


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--ids", nargs="*")
    ap.add_argument("--stride", type=int, default=8); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--cache", default=None); ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); ck = Path(a.ckpt); run = json.loads((ck / "run.json").read_text()); norm = D.Norm.load(ck / "norm_stats.json")
    arm, H, dev = norm.d["arm"], int(norm.d["horizon"]), torch.device(a.device)
    ids = a.ids or norm.d.get("val_ids") or None; rows = D.load_rows(Path(a.data), set(ids) if ids else None)
    data = D.GloveChunks(Path(a.data), rows, arm, H, norm, prompt=norm.d.get("prompt", "observable"), cache=a.cache, starts=range(0, D.N - 1, a.stride))
    cfg = M.make_config(H, dummy=run.get("dummy", False), dtype="float32" if dev.type == "cpu" else "bfloat16")
    model = M.build_model(cfg, dev, encoder=run.get("args", {}).get("tactile_encoder", "mlp")); rep = M.load_weights(model, str(ck / "model.safetensors")); model.eval()
    g = torch.Generator(device="cpu").manual_seed(a.seed); P, G, V, S9, Z, ZT, ZI, TG, per_row = [], [], [], [], [], [], [], [], {}
    for k0 in range(0, len(data), a.batch):
        items = [data[k] for k in range(k0, min(k0 + a.batch, len(data)))]; b = D.collate(items); obs, _act, tdel = D.to_obs(b, dev)
        noise = torch.randn((len(items), H, 32), generator=g).to(dev)
        with torch.no_grad():
            x = model.sample_actions(dev, obs, noise=noise, num_steps=a.steps).float().cpu().numpy()
            z = model.latent(D.to_obs(b, dev)[0]).float().mean(1)
            sh = np.roll(np.arange(len(items)), 1)
            zt = model.latent(D.to_obs(b, dev, tactile=b["tactile"][sh])[0]).float().mean(1)
            bi = dict(b); bi["image"] = b["image"][sh]; zi = model.latent(D.to_obs(bi, dev)[0]).float().mean(1)
            tg, _ = model.target_features(tdel)
        Z.append(z.cpu()); ZT.append(zt.cpu()); ZI.append(zi.cpu()); TG.append(tg.cpu())
        for j, it in enumerate(items):
            r, t = it["row"], it["t"]; st = data.lab[r]["state"]; gt, valid = D.chunk(data.lab[r][f"act_{arm}"], st, t, H)
            pr = norm.unaction(x[j].astype(np.float64)); pr[:, D.DELTA_MASK] += st[t, D.DELTA_MASK]; gt[:, D.DELTA_MASK] += st[t, D.DELTA_MASK]
            hold = np.repeat(data.lab[r][f"act_{arm}"][t][None].astype(np.float64), H, 0)          # repeat the current action (VTF: current command)
            P.append(pr); G.append(gt); V.append(valid); S9.append(hold)
            per_row.setdefault(rows[r]["id"], []).append(len(P) - 1)
    P, G, V, S9 = np.stack(P), np.stack(G), np.stack(V), np.stack(S9)
    res = dict(ckpt=str(ck), arm=arm, stage=run.get("stage"), n_starts=len(P), rows=[r["id"] for r in rows], weights_missing=rep["missing"],
               model=errors(P, G, V, arm), hold=errors(S9, G, V, arm))
    res["per_row"] = {i: errors(P[ix], G[ix], V[ix], arm) for i, ix in per_row.items()}
    Z, ZT, ZI, TG = torch.cat(Z), torch.cat(ZT), torch.cat(ZI), torch.cat(TG)
    res["z_shift_touch"] = float((1 - F.cosine_similarity(Z, ZT)).mean()); res["z_shift_image"] = float((1 - F.cosine_similarity(Z, ZI)).mean())
    ok = TG.norm(dim=-1) > 1e-3
    if int(ok.sum()) >= 2:
        sim = F.normalize(Z[ok], dim=-1) @ F.normalize(TG[ok], dim=-1).T
        res["z_top1"] = float((sim.argmax(1) == torch.arange(int(ok.sum()))).float().mean()); res["z_top1_chance"] = 1.0 / int(ok.sum())
    print(json.dumps({k: v for k, v in res.items() if k != "per_row"}, indent=1))
    Path(a.out or ck / "eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
