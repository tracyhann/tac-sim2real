"""Depth Anything V2 Large depth controls for the dual-mode LoRA dataset (lora_ge/dataset/ctrl_da/<row>.mp4): for every
row's video half (832 x 480; real head video or a zero-shot generated video), per-frame relative inverse depth -> gray,
near = bright (the encoding of the sim controls, scene_mug r2s2v/fun_controls.inv_gray), + the black touch panel.
Each frame's disparity is mapped through its 2-98 % range, the range smoothed over +-4 frames (Depth Anything's per-frame
scale drifts a little; a fixed clip-wide range would let the head's approach swing the whole frame's brightness).
Resumable (skips existing files). Also writes the same for any extra video given with --videos (eval inputs).
  envs/armA/bin/python ge_depth.py [--rows dataset/rows.json]"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ge_data import DS, H, N, R, W, read_video, with_panel, write_video  # noqa: E402

_DA = {}


def da_model():
    if not _DA:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        p = R / "models/Depth-Anything-V2-Large-hf"; _DA["p"] = AutoImageProcessor.from_pretrained(p)
        _DA["m"] = AutoModelForDepthEstimation.from_pretrained(p, torch_dtype=torch.float16).to("cuda").eval()
    return _DA["p"], _DA["m"]


@torch.no_grad()
def disparity(frames, bs=16):
    proc, m = da_model(); out = []
    for i in range(0, len(frames), bs):
        x = proc(images=list(frames[i:i + bs]), return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
        d = m(pixel_values=x).predicted_depth.float()[:, None]; out.append(Fn.interpolate(d, size=(H, W), mode="bicubic", align_corners=False)[:, 0].cpu().numpy())
    return np.concatenate(out)


def gray(D):
    lo, hi = np.percentile(D.reshape(len(D), -1), 2, axis=1), np.percentile(D.reshape(len(D), -1), 98, axis=1)
    k = np.ones(9) / 9; pad = lambda a: np.convolve(np.pad(a, 4, mode="edge"), k, "valid")
    lo, hi = pad(lo), pad(hi)
    return [np.repeat((np.clip((D[t] - lo[t]) / max(hi[t] - lo[t], 1e-6), 0, 1) * 255).round().astype(np.uint8)[..., None], 3, -1) for t in range(len(D))]


def make(src, dst):
    if dst.exists():
        return "cached"
    fr, n = read_video(src, size=None)                            # a canvas (video | panel) or a plain 832 x 480 video
    fr = fr[:, :, :W] if fr.shape[2] == W + 256 else np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in fr]) if fr.shape[1:3] != (H, W) else fr
    assert len(fr) == N and fr.shape[1:3] == (H, W), (src, fr.shape)
    write_video(dst, with_panel(gray(disparity(fr)))); return "ok"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", default=str(DS / "rows.json")); ap.add_argument("--videos", nargs="*", default=[])
    a = ap.parse_args(); rows = json.loads(Path(a.rows).read_text()); todo = {}
    for r in rows:
        for k in ("control_file_path", "control_alt_file_path"):
            if r.get(k) and r[k].startswith("ctrl_da/"):
                todo[DS / r[k]] = DS / r["file_path"]
    for v in a.videos:                                            # extra: src=dst pairs
        s, d = v.split("="); todo[Path(d)] = Path(s)
    for i, (dst, src) in enumerate(sorted(todo.items())):
        print(i, len(todo), dst.name, make(src, dst), flush=True)


if __name__ == "__main__":
    main()
