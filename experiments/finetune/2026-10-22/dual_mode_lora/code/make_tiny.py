"""A tiny Wan2.2-Fun-A14B-Control stand-in for the trainer test (2 blocks, dim 256, random weights; the real VAE, T5 and
tokenizer are symlinked): the dual-mode trainer runs end to end on it in minutes on one card.
  envs/videoxfun/bin/python make_tiny.py OUTDIR"""
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); M = R / "models/Wan2.2-Fun-A14B-Control"
sys.path.insert(0, str(R / "third_party/VideoX-Fun"))
from videox_fun.models import Wan2_2Transformer3DModel  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for f in ("google", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
    if not (out / f).exists():
        os.symlink(M / f, out / f)
for sub in ("high_noise_model", "low_noise_model"):
    cfg = json.loads((M / sub / "config.json").read_text())
    cfg.update(dim=256, hidden_size=256, ffn_dim=512, num_heads=2, num_layers=2)
    torch.manual_seed(0); m = Wan2_2Transformer3DModel.from_config({k: v for k, v in cfg.items() if not k.startswith("_")})
    (out / sub).mkdir(exist_ok=True); (out / sub / "config.json").write_text(json.dumps(cfg, indent=1))
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(out / sub / "diffusion_pytorch_model.safetensors"))
    print(sub, sum(p.numel() for p in m.parameters()) / 1e6, "M params")
