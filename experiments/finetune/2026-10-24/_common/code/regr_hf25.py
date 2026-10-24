"""Reference row "world-model video + a separate touch estimator" (train.sh part ev_<ARM>_<CK>, after predict_hf25.py):
the R6 head of the loss table (dual_mode_lora/code/loss_table.py: frozen Wan VAE latents -> 3D conv + temporal trunk -> relative
total x Sinkhorn load share; lam 0.01 as chosen there), trained on the 25 HF rows' video latents and touch (seed 0, 150 epochs,
~2 min on one card), then applied to the arm's generated test videos. Against the arm's own head this shows what the backbone's
touch head adds over estimating touch from the finished video.
Writes outputs/R2S2V_mug/hf25/out/<ARM>c<CK>_regr/<case>_s<seed>_tactile.npy (relative F / S) + regr/{R6_s0.pt, lat/*.pt}.
  envs/videoxfun/bin/python regr_hf25.py --arm FHN --ckpt 500 [--seed 0]"""
import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import torch

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); GE = R / "experiments/finetune/2026-10-22/dual_mode_lora/code"
sys.path.insert(0, str(GE)); sys.path.insert(0, str(R / "third_party/VideoX-Fun"))
import loss_table as LT  # noqa: E402

O = R / "outputs/R2S2V_mug/hf25"; DS = O / "dataset"; LAT = O / "regr" / "lat"; LT.DS = DS          # loss_table.target reads DS / row["tactile"] (touch_rel)
LAM = 0.01


def vae_model():
    from omegaconf import OmegaConf
    from videox_fun.models import AutoencoderKLWan
    cfg = OmegaConf.load(R / "third_party/VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml")
    return AutoencoderKLWan.from_pretrained(str(R / "models/Wan2.2-Fun-A14B-Control/Wan2.1_VAE.pth"), additional_kwargs=OmegaConf.to_container(cfg["vae_kwargs"])).to(torch.bfloat16).cuda().eval()


def latent(vae, key, path):
    """(16, 21, 60, 104) latent of an 832 x 480 x 81 video, cached under regr/lat/<key>.pt (as regress_ge.latents: mode)"""
    p = LAT / f"{key}.pt"
    if not p.exists():
        v = av.open(str(path)); x = np.stack([cv2.resize(f.to_ndarray(format="rgb24"), (832, 480), interpolation=cv2.INTER_AREA) for f in v.decode(video=0)]); v.close()
        assert len(x) == 81, (path, len(x))
        t = torch.from_numpy(x).permute(3, 0, 1, 2)[None].float().div(127.5).sub(1).to(torch.bfloat16).cuda()
        with torch.no_grad():
            d = vae.encode(t)[0]; z = d.mode() if hasattr(d, "mode") else d.mean
        torch.save(z[0].float().cpu().half(), p)
    return torch.load(p).float()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--ckpt", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); LAT.mkdir(parents=True, exist_ok=True); tag = f"{a.arm}c{a.ckpt}"; od = O / "out" / f"{tag}_regr"; od.mkdir(parents=True, exist_ok=True)
    rows = json.loads((DS / "metadata_FHN.json").read_text()); cases = json.loads((DS / "test_cases.json").read_text())
    gens = [(c, p) for c in cases for p in sorted((O / "out" / tag).glob(f"{c['case']}_s*.mp4"))]
    todo = [(c, p) for c, p in gens if not (od / f"{p.stem}_tactile.npy").exists()]
    print(tag, len(gens), "generated videos,", len(todo), "to estimate", flush=True)
    if not todo:
        return
    vae = vae_model()
    lat = {r["id"]: latent(vae, r["id"], DS / r["file_path"]) for r in rows}
    zs = {p.stem: latent(vae, f"{tag}_{p.stem}", p) for c, p in todo}
    del vae; torch.cuda.empty_cache()
    mp = O / "regr" / f"R6_s{a.seed}.pt"
    if mp.exists():
        m = LT.Net(LT.OUT_DIM["R6"]).to(LT.dev); m.load_state_dict(torch.load(mp, map_location=LT.dev))
    else:
        m = LT.fit("R6", LAM, rows, lat, a.seed); torch.save(m.state_dict(), mp); print("fitted R6 on", len(rows), "HF rows", flush=True)
    for c, p in todo:
        np.save(od / f"{p.stem}_tactile.npy", LT.predict("R6", m, zs[p.stem])); print("estimated", p.stem, flush=True)
    (od / "info.json").write_text(json.dumps(dict(method="R6 regressor (loss_table) trained on the 25 HF rows, seed %d, lam %g" % (a.seed, LAM), videos=[p.stem for _, p in gens]), indent=1))


if __name__ == "__main__":
    main()
