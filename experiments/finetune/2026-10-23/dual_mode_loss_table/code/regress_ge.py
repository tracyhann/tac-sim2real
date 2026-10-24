"""Discriminative baseline for the estimate job (E): frozen Wan2.1 VAE latents of the 832 x 480 video (the same encoder
the LoRA sees its known video through; 16 x 21 x 60 x 104) -> a small 3D-conv + temporal head -> the 880 taxels of every
frame. Target = the panel's own encoding of RELATIVE touch (v3), sqrt(min(F/S, cap) / cap) per taxel (capped where the LoRA's panel is).
Two variants, on the same rows as the LoRA runs: "regr_E" (real + sim train rows, as run E) and "regr_R" (real rows only,
as run R). The number of epochs is picked on a split by participant inside the training rows (actor_10 held out), then
the model is retrained on all training rows for that many epochs; seed 0.
Writes lora_ge/out/regr_<v>/E_<row>_<ctrl>_s0_tactile.npy for every held-out E case (the regressor sees no control: the
same prediction is written under each control name, so the table lines up) + regr/<v>.json (curves, chosen epochs).
  envs/videoxfun/bin/python regress_ge.py"""
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/experiments/R2S2V_mug")))
R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); LG = R / "outputs/R2S2V_mug/lora_ge"; DS = LG / "dataset"; RG = LG / "regr"
X0 = 832
from panel_v3 import F_MAX  # noqa: E402  (relative cap, units of S: the target is what the LoRA's panel v3 shows)
from ge_cases import cases  # noqa: E402


def latents(rows):
    (RG / "lat").mkdir(parents=True, exist_ok=True); todo = [r for r in rows if not (RG / "lat" / f"{r['id']}.pt").exists()]
    if todo:
        from omegaconf import OmegaConf
        sys.path.insert(0, str(R / "third_party/VideoX-Fun")); from videox_fun.models import AutoencoderKLWan
        cfg = OmegaConf.load(R / "third_party/VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml")
        vae = AutoencoderKLWan.from_pretrained(str(R / "models/Wan2.2-Fun-A14B-Control/Wan2.1_VAE.pth"), additional_kwargs=OmegaConf.to_container(cfg["vae_kwargs"])).to(torch.bfloat16).cuda().eval()
        for r in todo:
            c = av.open(str(DS / r["file_path"])); x = np.stack([f.to_ndarray(format="rgb24")[:, :X0] for f in c.decode(video=0)]); c.close()
            t = torch.from_numpy(x).permute(3, 0, 1, 2)[None].float().div(127.5).sub(1).to(torch.bfloat16).cuda()
            with torch.no_grad():
                d = vae.encode(t)[0]; z = d.mode() if hasattr(d, "mode") else d.mean      # the trainer samples; the mean is the deterministic version
            torch.save(z[0].float().cpu().half(), RG / "lat" / f"{r['id']}.pt"); print("latent", r["id"], tuple(z.shape), flush=True)
        del vae; torch.cuda.empty_cache()
    return {r["id"]: torch.load(RG / "lat" / f"{r['id']}.pt").float() for r in rows}


def target(r):
    T = np.load(DS / r["tactile"])["touch_rel"]; return torch.from_numpy(np.sqrt(np.minimum(np.clip(T, 0, None), F_MAX) / F_MAX)).float()   # (81, 880)


class Head(nn.Module):
    def __init__(self, c=16, p=0.2):
        super().__init__()
        ch = [c, 64, 128, 192, 256, 256]; L = []
        for i in range(5):
            L += [nn.Conv3d(ch[i], ch[i + 1], 3, stride=(1, 1, 1) if i == 0 else (1, 2, 2), padding=1), nn.GroupNorm(8, ch[i + 1]), nn.GELU()]
        self.enc = nn.Sequential(*L); self.pool = nn.AdaptiveAvgPool3d((None, 4, 7)); self.drop = nn.Dropout(p)
        self.fc = nn.Linear(256 * 28, 512); self.tconv = nn.Conv1d(512, 512, 3, padding=1); self.out = nn.Linear(512, 4 * 880)

    def forward(self, z):                                          # (b, 16, 21, 60, 104) -> (b, 81, 880) in [0, 1]
        h = self.pool(self.enc(z)); b, c, t = h.shape[:3]; h = h.permute(0, 2, 1, 3, 4).reshape(b, t, -1)
        h = torch.relu(self.fc(self.drop(h))); h = h + torch.relu(self.tconv(h.transpose(1, 2))).transpose(1, 2)
        y = torch.sigmoid(self.out(self.drop(h))).reshape(b, t, 4, 880)
        return torch.cat([y[:, :1, 0], y[:, 1:].reshape(b, -1, 880)], 1)   # latent frame 0 = frame 0; j = frames 4j-3 .. 4j


def fit(tr, lat, epochs, val=None, seed=0, bs=4, lr=3e-4):
    torch.manual_seed(seed); m = Head().cuda(); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05); curve = []
    Z = torch.stack([lat[r["id"]] for r in tr]); Y = torch.stack([target(r) for r in tr]); g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        m.train(); perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(tr), bs):
            j = perm[i:i + bs]; z = Z[j].cuda(); z = z + 0.05 * torch.randn_like(z)            # light latent noise
            loss = ((m(z) - Y[j].cuda()) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if val is not None and (ep % 5 == 0 or ep == epochs):
            curve.append((ep, evaluate(m, val, lat)))
    return m, curve


@torch.no_grad()
def predict(m, z):
    m.eval(); y = m(z[None].cuda())[0].clamp(0, 1).cpu().numpy(); return (y ** 2) * F_MAX


def evaluate(m, rows, lat):
    from eval_ge import metrics
    s = [metrics(predict(m, lat[r["id"]]), np.load(DS / r["tactile"])["touch_rel"]) for r in rows]
    return {k: float(np.nanmean([x[k] for x in s])) for k in ("f1_6mm", "f1_pad", "total_r", "time_iou")}


def main():
    rows = json.loads((DS / "rows.json").read_text()); lat = latents(rows); byid = {r["id"]: r for r in rows}
    train = [r for r in rows if r["split"] == "train"]; rep = {}
    for v, tr in (("regr_E", train), ("regr_R", [r for r in train if r["kind"] == "real"])):
        inner_tr = [r for r in tr if r["actor"] != "actor_10"]; inner_va = [r for r in tr if r["actor"] == "actor_10"]
        t = time.time(); _, curve = fit(inner_tr, lat, 150, val=inner_va)
        score = lambda s: s["f1_6mm"] + s["time_iou"] + (s["total_r"] if np.isfinite(s["total_r"]) else 0)   # location + timing + course
        best = max(curve, key=lambda c: score(c[1]))[0]
        m, _ = fit(tr, lat, best); od = LG / "out" / v; od.mkdir(parents=True, exist_ok=True)
        for c in cases("E"):
            np.save(od / f"{c['case']}_s0_tactile.npy", predict(m, lat[c["row"]]))
        rep[v] = dict(n_train=len(tr), inner_train=len(inner_tr), inner_val=len(inner_va), curve=curve, epochs=best, seconds=round(time.time() - t))
        print(v, "epochs", best, "val", dict(curve)[best], flush=True)
    RG.mkdir(parents=True, exist_ok=True); (RG / "regr.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
