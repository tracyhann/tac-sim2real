"""Train one VTLA arm on the glove dataset: the paper's three stages in one run, each saved.
  pred   tactile stack only (encoder, predictor, recon head) on aux_loss (InfoNCE z vs z* + per-pad L1); policy frozen
  align  z_proj, z gate, action expert on the action loss; action queries cannot see the VL prefix (only z)
  joint  everything on the action loss (+ --aux_weight x aux_loss), tactile stack and z_proj at 0.1 x lr
Arms (only the action target differs; inputs, data, steps and seeds are the same):
  vts  action = measured state chunk  [wrist xyz, rot6d, closure, 22 measured joints]
  vtf  action = force-informed chunk  [wrist xyz, rot6d, total commanded grip force, 22 commanded joints]
Writes <out>/<stage>/{model.safetensors, norm_stats.json, run.json} and <out>/log.jsonl.
  python train_vtla.py --arm vtf --data <dataset> --base <n0-vtla-base dir> --out <dir> [--stages pred:500,align:500,joint:3000]
  python train_vtla.py --arm vts --dummy --data <dataset> --out /tmp/t --stages pred:2,align:2,joint:2 --device cpu   (no weights)"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

import vtla_data as D
import vtla_model as M


def lr_at(step, total, peak, warm, floor_frac=0.1):
    if step < warm:
        return peak * (step + 1) / (warm + 1)
    p = min(1.0, (step - warm) / max(1, total - warm)); return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def save(model, out: Path, norm: D.Norm, run: dict):
    out.mkdir(parents=True, exist_ok=True); tmp = out / "model.safetensors.tmp"
    safetensors.torch.save_model(model, str(tmp)); os.replace(tmp, out / "model.safetensors")
    norm.save(out / "norm_stats.json"); (out / "run.json").write_text(json.dumps(run, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=D.ARMS, required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None, help="dir with the N0-VTLA base model.safetensors (omit with --dummy)")
    ap.add_argument("--init", default=None, help="start from an earlier stage dir of this code (model.safetensors) instead of --base")
    ap.add_argument("--stages", default="pred:500,align:500,joint:3000"); ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr_pred", type=float, default=1e-4); ap.add_argument("--lr_align", type=float, default=5e-5); ap.add_argument("--lr_joint", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--aux_weight", type=float, default=0.0); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--prompt", choices=("observable", "full"), default="observable"); ap.add_argument("--val_ids", nargs="*", default=[])
    ap.add_argument("--val_groups", nargs="*", default=[]); ap.add_argument("--cache", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dummy", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--no_grad_ckpt", action="store_true"); ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--save", choices=("all", "last"), default="all", help="save every stage (~7 GB each) or only the last")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = torch.device(a.device)
    ds = Path(a.data); rows = D.load_rows(ds); rows = rows[:a.max_rows] if a.max_rows else rows
    tr, va = D.split_rows(rows, set(a.val_ids), set(a.val_groups)); assert tr, "no training rows (are vtla/<id>.npz present?)"
    norm = D.compute_norm(ds, tr, a.arm, a.horizon); norm.d.update(prompt=a.prompt, val_ids=[r["id"] for r in va])
    data = D.GloveChunks(ds, tr, a.arm, a.horizon, norm, prompt=a.prompt, cache=a.cache)
    dl = torch.utils.data.DataLoader(data, batch_size=a.batch, shuffle=True, drop_last=len(data) >= a.batch, num_workers=a.workers, collate_fn=D.collate,
                                     persistent_workers=a.workers > 0)
    logging.info("arm %s: %d train rows (%d samples), %d val rows; tactile scale %.3f N; prompt[0] %r", a.arm, len(tr), len(data), len(va), norm.tac, data.prompts[0])
    cfg = M.make_config(a.horizon, dummy=a.dummy, dtype="float32" if dev.type == "cpu" else "bfloat16")
    model = M.build_model(cfg, dev); report = None
    if a.init:
        report = M.load_weights(model, str(Path(a.init) / "model.safetensors"))
    elif a.base:
        report = M.load_weights(model, str(Path(a.base) / "model.safetensors"))
    if not a.no_grad_ckpt and not a.dummy:
        model.gradient_checkpointing_enable()
    model.train(); logf = (out / "log.jsonl").open("a"); step_total = 0
    stages = [(s.split(":")[0], int(s.split(":")[1])) for s in a.stages.split(",") if s]
    for stage, n_steps in stages:
        groups = M.set_stage(model, stage); peak = dict(pred=a.lr_pred, align=a.lr_align, joint=a.lr_joint)[stage]; model.zero_grad(set_to_none=True)
        opt = torch.optim.AdamW(groups, lr=peak, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)   # the repo's AdamW defaults
        logging.info("== stage %s: %d steps, %.1f M trainable params, peak lr %.1e", stage, n_steps, M.count(groups) / 1e6, peak)
        it = iter(dl); t0 = time.time(); hist = []
        for s in range(n_steps):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            obs, actions, tac_delta = D.to_obs(b, dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(s, n_steps, peak, min(a.warmup, n_steps // 5)) * g.get("lr_scale", 1.0)
            parts = {}
            if stage == "pred":
                loss, parts = model.aux_loss(obs, tac_delta)
            else:
                loss = model(obs, actions).mean(); parts["act"] = float(loss.detach())
                if a.aux_weight > 0:
                    aux, ap_ = model.aux_loss(obs, tac_delta); loss = loss + a.aux_weight * aux; parts.update({f"aux_{k}": v for k, v in ap_.items()})
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], a.clip)
            if torch.isfinite(gn):
                opt.step()
            rec = dict(stage=stage, step=s, total=step_total, loss=float(loss.detach()), grad_norm=float(gn), lr=opt.param_groups[0]["lr"], **parts)
            if hasattr(model, "z_gate"):
                rec["z_gate"] = float(model.z_gate.detach().float().reshape(-1)[0])
            hist.append(rec["loss"]); logf.write(json.dumps(rec) + "\n"); step_total += 1
            if s % a.log_every == 0 or s == n_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0
                logging.info("%s %4d/%d loss %.4f (mean10 %.4f) gn %.2f lr %.2e %s | %.2f s/step, %.1f GB", stage, s, n_steps, rec["loss"], float(np.mean(hist[-10:])), rec["grad_norm"], rec["lr"],
                             {k: round(v, 4) if isinstance(v, float) else v for k, v in parts.items()}, (time.time() - t0) / (s + 1), mem)
                logf.flush()
        if a.save == "all" or (stage, n_steps) == stages[-1]:
            save(model, out / stage, norm, dict(arm=a.arm, stage=stage, steps=n_steps, args=vars(a), weights_report=report, first10=float(np.mean(hist[:10])), last10=float(np.mean(hist[-10:])),
                                                sec_per_step=(time.time() - t0) / max(n_steps, 1), dummy=a.dummy))
        logging.info("== stage %s done: loss first10 %.4f -> last10 %.4f", stage, np.mean(hist[:10]), np.mean(hist[-10:]))
    M.set_stage(model, "joint"); model._vl_dropout_override = None
    final = out / "final"
    if final.is_symlink() or final.exists():
        final.unlink()
    final.symlink_to(stages[-1][0])


if __name__ == "__main__":
    main()
