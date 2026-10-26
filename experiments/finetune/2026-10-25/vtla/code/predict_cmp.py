"""Policy predictions for the physics test (README section 11): for every test row, the checkpoint's action chunks from the row's
own observations at frames 0, 16, 32, 48, 64 (head-camera video frame, measured hand state, prompt; no touch for no-touch runs),
10 Euler steps, fixed noise; undo normalisation and deltas -> one absolute action per frame 1..80. --every K re-plans every K frames
instead (a chunk from frames 0, K, 2K, ...; its first K actions are executed): pi0.5 executes part of each chunk, and the release
of the held-out recordings starts at frames 64-70, which a 16-frame chunk from frame 64 alone cannot see. Saved per row:
<out>/<id>.npz  t_sim (80,) simulation time of each frame, q (80, 22) the predicted finger joint targets (observed pose for arm obs,
commanded pose for arm cmd), joint_names, wrist (80, 9), slot9 (80,), arm. cliprep/oracle_exec.py --table then drives the replay's
finger servos with q (the robot executes positions only).
  python predict_cmp.py --ckpt <run>/final --data <dataset_cmp_t> --ids $(cat test_ids.txt) --out <run>/pred"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import vtla_data as D
import vtla_model as M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=10); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--seeds", type=int, nargs="*", help="several noise seeds: <out>/s<k>/ each"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default=None); ap.add_argument("--every", type=int, default=0, help="re-plan every K frames (default: the horizon)")
    a = ap.parse_args(); ck = Path(a.ckpt); run = json.loads((ck / "run.json").read_text()); norm = D.Norm.load(ck / "norm_stats.json")
    arm, H, dev = norm.d["arm"], int(norm.d["horizon"]), torch.device(a.device); no_touch = bool(run.get("args", {}).get("no_touch", False))
    step = a.every or H; assert 1 <= step <= H, (step, H); starts = list(range(0, D.N - 1, step)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = D.load_rows(Path(a.data), set(a.ids)); assert len(rows) == len(set(a.ids)), (len(rows), len(set(a.ids)))
    data = D.GloveChunks(Path(a.data), rows, arm, H, norm, prompt=norm.d.get("prompt", "observable"), cache=a.cache, starts=starts)
    model = M.build_model(M.make_config(H, dtype="bfloat16"), dev, encoder=run.get("args", {}).get("tactile_encoder", "grid"))
    rep = M.load_weights(model, str(ck / "model.safetensors")); model.eval()
    for seed in (a.seeds or [a.seed]):
        predict_rows(a, model, data, rows, norm, arm, H, dev, no_touch, seed, out / f"s{seed}" if a.seeds else out, step)
    (out / "predict.json").write_text(json.dumps(dict(ckpt=str(ck), arm=arm, no_touch=no_touch, n=len(rows), seeds=a.seeds or [a.seed], every=step, weights_missing=rep["missing"]), indent=1))


def predict_rows(a, model, data, rows, norm, arm, H, dev, no_touch, seed, out, step):
    out.mkdir(parents=True, exist_ok=True); g = torch.Generator().manual_seed(seed); ck = Path(a.ckpt)
    for i, r in enumerate(rows):
        items = [data[k] for k, (ri, t) in enumerate(data.items) if ri == i]; b = D.collate(items); obs, _act, _td = D.to_obs(b, dev, drop_touch=no_touch)
        noise = torch.randn((len(items), H, 32), generator=g).to(dev)
        with torch.no_grad():
            x = model.sample_actions(dev, obs, noise=noise, num_steps=a.steps).float().cpu().numpy()
        st = data.lab[i]["state"]; act = np.zeros((D.N - 1, 32))
        for j, it in enumerate(items):
            t = it["t"]; pr = norm.unaction(x[j].astype(np.float64)); pr[:, D.DELTA_MASK] += st[t, D.DELTA_MASK]
            n = min(step, D.N - 1 - t); act[t:t + n] = pr[:n]
        hp = np.load(r["hand_pose"]); names = [str(z) for z in hp["finger_q_names"]]
        np.savez_compressed(out / f"{r['id']}.npz", t_sim=hp["t_sim"][1:].astype(np.float64), q=act[:, 10:].astype(np.float32), joint_names=np.array(names),
                            wrist=act[:, :9].astype(np.float32), slot9=act[:, 9].astype(np.float32), arm=np.array(arm), ckpt=np.array(str(ck)))
        gt = data.lab[i][f"act_{arm}"][1:, 10:]
        print(f"seed {seed} {r['id']:22s} {arm}: |pred - own label| joints {np.degrees(np.abs(act[:, 10:] - gt)).mean():.2f} deg", flush=True)


if __name__ == "__main__":
    main()
