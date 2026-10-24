"""Does the tactile panel survive Wan's VAE (8x spatial, 4x temporal)? Encode + decode each training canvas with the
Wan2.2-Fun-A14B VAE, read the panel back (tactile_panel.decode) and compare with the target taxels.
  python vae_check.py <dataset_dir>   -> <dataset_dir>/vae_check.json"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from tactile_panel import F_MAX, decode, finger_totals  # noqa: E402

R = "/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"; V = f"{R}/third_party/VideoX-Fun"; sys.path.insert(0, V)


def main(ds):
    from omegaconf import OmegaConf
    from videox_fun.models import AutoencoderKLWan
    from videox_fun.utils import get_video_to_video_latent
    ds = Path(ds); os.chdir(V); cfg = OmegaConf.load("config/wan2.2/wan_civitai_i2v.yaml"); path = f"{R}/models/Wan2.2-Fun-A14B-Control"
    vae = AutoencoderKLWan.from_pretrained(os.path.join(path, cfg["vae_kwargs"].get("vae_subpath", "vae")), additional_kwargs=OmegaConf.to_container(cfg["vae_kwargs"])).to(torch.bfloat16).cuda()
    out = {}
    for m in json.loads((ds / "metadata.json").read_text()):
        v, _, _, _ = get_video_to_video_latent(str(ds / m["file_path"]), video_length=81, sample_size=[480, 1088], fps=16, ref_image=None)   # (1, 3, F, H, W) in [0, 1]
        with torch.no_grad():
            x = (v * 2 - 1).to("cuda", torch.bfloat16); lat = vae.encode(x)[0].mode(); y = vae.decode(lat).sample if hasattr(vae.decode(lat), "sample") else vae.decode(lat)[0]
        y = ((y.float().clamp(-1, 1) + 1) / 2 * 255).round().byte()[0].permute(1, 2, 3, 0).cpu().numpy()          # (F, H, W, 3)
        T = np.minimum(np.load(ds / m["tactile"])[m["tactile_key"]], F_MAX); P = np.stack([decode(f[:, 832:])[0] for f in y[:len(T)]])
        on = T > 0.15; ft, fp = np.array([finger_totals(t) for t in T]), np.array([finger_totals(p) for p in P])
        out[f"{m['clip']}_{m['source']}"] = dict(taxel_mae=float(np.abs(P - T).mean()), contact_mae=float(np.abs(P - T)[on].mean()) if on.any() else None,
                                                 finger_corr=float(np.corrcoef(ft.ravel(), fp.ravel())[0, 1]), total_corr=float(np.corrcoef(T.sum(1), P.sum(1))[0, 1]))
        print(m["clip"], m["source"], {k: round(x, 4) if x is not None else None for k, x in out[f"{m['clip']}_{m['source']}"].items()}, flush=True)
    (ds / "vae_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
