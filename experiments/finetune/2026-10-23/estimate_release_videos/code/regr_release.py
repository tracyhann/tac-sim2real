"""Discriminative baseline on the released videos: the R6 regression head of the loss table (dual_mode_loss_table; frozen Wan VAE
latents of the 832 x 480 video -> 3D-conv + temporal trunk -> relative total x Sinkhorn load share), trained exactly as there
(79 lora_ge training rows = the rows U saw, lam chosen on the inner split, 150 epochs) with seeds 0 / 1 / 2, then run on the
25 release videos. Same input as U8's job E: the generated video only (the regressor sees no control). Output relative F / S.
Writes outputs/R2S2V_mug/release_estimate/R6_s<k>/<id>_tactile.npy (+ release_estimate/regr_lat/<id>.pt, the video latents).
  envs/videoxfun/bin/python regr_release.py"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); HERE = Path(__file__).resolve().parent
GE = R / "experiments/finetune/2026-10-22/dual_mode_lora/code"; sys.path.insert(0, str(GE))
from loss_table import DS, LG, SEEDS, fit, latents, predict  # noqa: E402

OUT = R / "outputs/R2S2V_mug/release_estimate"; LAT = OUT / "regr_lat"


def release_latents(cases):
    LAT.mkdir(parents=True, exist_ok=True); todo = [c for c in cases if not (LAT / f"{c['id']}.pt").exists()]
    if todo:
        import av
        import cv2
        from omegaconf import OmegaConf
        sys.path.insert(0, str(R / "third_party/VideoX-Fun")); from videox_fun.models import AutoencoderKLWan
        cfg = OmegaConf.load(R / "third_party/VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml")
        vae = AutoencoderKLWan.from_pretrained(str(R / "models/Wan2.2-Fun-A14B-Control/Wan2.1_VAE.pth"), additional_kwargs=OmegaConf.to_container(cfg["vae_kwargs"])).to(torch.bfloat16).cuda().eval()
        for c in todo:                                   # as predict_release.read (832 x 480, INTER_AREA) and regress_ge.latents (mode)
            v = av.open(str(R / c["gen"])); x = np.stack([cv2.resize(f.to_ndarray(format="rgb24"), (832, 480), interpolation=cv2.INTER_AREA) for f in v.decode(video=0)]); v.close()
            assert len(x) == 81, (c["id"], len(x))
            t = torch.from_numpy(x).permute(3, 0, 1, 2)[None].float().div(127.5).sub(1).to(torch.bfloat16).cuda()
            with torch.no_grad():
                d = vae.encode(t)[0]; z = d.mode() if hasattr(d, "mode") else d.mean
            torch.save(z[0].float().cpu().half(), LAT / f"{c['id']}.pt"); print("latent", c["id"], tuple(z.shape), flush=True)
        del vae; torch.cuda.empty_cache()
    return {c["id"]: torch.load(LAT / f"{c['id']}.pt").float() for c in cases}


def main():
    cases = json.loads((HERE / "cases.json").read_text()); zl = release_latents(cases)
    rows = json.loads((DS / "rows.json").read_text()); lat = latents(rows); train = [r for r in rows if r["split"] == "train"]
    lam = json.loads((LG / "eval/loss_table.json").read_text())["chosen"]["R6"]["lam"]; print("R6 lam", lam, "train rows", len(train), flush=True)
    for seed in SEEDS:
        od = OUT / f"R6_s{seed}"; od.mkdir(parents=True, exist_ok=True)
        if all((od / f"{c['id']}_tactile.npy").exists() for c in cases):
            continue
        m = fit("R6", lam, train, lat, seed)
        for c in cases:
            np.save(od / f"{c['id']}_tactile.npy", predict("R6", m, zl[c["id"]]))
        print("seed", seed, "done", flush=True)


if __name__ == "__main__":
    main()
