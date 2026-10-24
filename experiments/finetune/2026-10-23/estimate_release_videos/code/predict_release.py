"""Touch estimation on the released generated videos (user 2026-10-23: "submit a job on Vision: let our fine-tuned world model infer
the tactile of these videos, try it"; then "you may also try our earlier video + paired tactile model; prefer the checkpoints we
already have"). Existing checkpoints only, nothing is trained:
  U8   dual-mode LoRA, run U epoch 8 (lora_ge U_low / U_high checkpoint-632), job E exactly as its held-out eval (predict_ge.py):
       the 81 video frames known, the touch panel unknown (grey 0.5), CFG 1, no reference image, 30 steps; panel v3 -> RELATIVE
       touch F / S (S = the clip's median total force over its contact frames; the model never sees S)
  f2   video + touch LoRA f2 step 600 (lora_tactile f2_low / f2_high checkpoint-600; held-out contact F1 @ 6 mm 0.66). It was
       trained to generate video + touch from frame 0; here it gets the same full-video mask as U8 (a mask it never saw in
       training) and the same settings; panel v1 -> touch in N
Both: control = the video's own sim control (854 x 480 -> 832 x 480 + black 256 px panel, as ge_data.sim_control), prompt = the
video's generation prompt + the panel sentence both LoRAs were trained with. The round-6 head videos were U's training rows.
Cases: cases.json (made on the Mac from the release bundle). Writes outputs/R2S2V_mug/release_estimate/<model>/<id>.mp4 (canvas),
<id>_tactile.npy ((81, 880): relative for U8, N for f2) and <id>.json.
  E8 / R8 (added 2026-10-24, user: "do you have E, R and f2 results?"): the dual-mode estimate-only runs at epoch 8, E (real + sim
       rows, checkpoint-632) and R (real rows only, checkpoint-320), same job-E protocol as U8; relative touch
  envs/videoxfun/bin/python predict_release.py --model U8|f2|E8|R8|E4 --group r10|r8b|r6"""
import argparse
import json
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np
import torch

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); X = R / "experiments/R2S2V_mug"; HERE = Path(__file__).resolve().parent
GE = R / "experiments/finetune/2026-10-22/dual_mode_lora/code"
sys.path.insert(0, str(X)); sys.path.insert(0, str(GE)); sys.path.insert(1, str(X / "lora_tactile"))
from fun_zeroshot import WAN_NEGATIVE, load_pipe, write_video  # noqa: E402
import panel_v3  # noqa: E402  (dual-mode LoRA: relative touch)
import tactile_panel  # noqa: E402  (f2: panel v1, N)

H, W, X0, N = 480, 1088, 832, 81
PANEL = " On the right, a grey touch map of the left glove: each cell is one sensor, brighter is pressed harder; the bars below are the total force per finger."
MODELS = {"U8": (R / "outputs/R2S2V_mug/lora_ge" / "U_{}" / "checkpoint-632.safetensors", panel_v3.decode),
          "f2": (R / "outputs/R2S2V_mug/lora_tactile" / "f2_{}" / "checkpoint-600.safetensors", tactile_panel.decode),
          "E8": (R / "outputs/R2S2V_mug/lora_ge" / "E_{}" / "checkpoint-632.safetensors", panel_v3.decode),      # 10-24: estimate-only runs
          "R8": (R / "outputs/R2S2V_mug/lora_ge" / "R_{}" / "checkpoint-320.safetensors", panel_v3.decode),
          "E4": (R / "outputs/R2S2V_mug/lora_ge" / "E_{}" / "checkpoint-316.safetensors", panel_v3.decode)}    # E at R8's step count (mixed vs real at equal steps)
OUT = R / "outputs/R2S2V_mug/release_estimate"


def read(p, size=(X0, H)):
    c = av.open(str(p)); f = [cv2.resize(x.to_ndarray(format="rgb24"), size, interpolation=cv2.INTER_AREA) for x in c.decode(video=0)]; c.close()
    assert len(f) == N, (p, len(f)); return np.stack(f)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(MODELS)); ap.add_argument("--group", required=True)
    ap.add_argument("--steps", type=int, default=30); ap.add_argument("--guidance", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); ck, decode = MODELS[a.model]; lo, hi = Path(str(ck).format("low")), Path(str(ck).format("high")); assert lo.exists() and hi.exists(), (lo, hi)
    od = OUT / a.model; (od / "ctrl").mkdir(parents=True, exist_ok=True)
    todo = [c for c in json.loads((HERE / "cases.json").read_text()) if c["group"] == a.group and not (od / f"{c['id']}.json").exists()]
    print(a.model, a.group, len(todo), "cases", flush=True)
    if not todo:
        return
    pipe, vae, sch, boundary = load_pipe("a14b")
    from videox_fun.utils import get_video_to_video_latent, merge_lora
    pipe = merge_lora(pipe, str(lo), 1.0, device="cuda", dtype=torch.bfloat16)
    pipe = merge_lora(pipe, str(hi), 1.0, device="cuda", dtype=torch.bfloat16, sub_transformer_name="transformer_2")
    for c in todo:
        cp = od / "ctrl" / f"{c['id']}.mp4"                     # the sim control as a canvas: 832 x 480 + black panel (ge_data.sim_control)
        if not cp.exists():
            write_video(cp, np.concatenate([read(R / c["control"]), np.zeros((N, H, W - X0, 3), np.uint8)], 2), fps=16)
        v = np.full((N, H, W, 3), 0.5, np.float32); v[:, :, :X0] = read(R / c["gen"]).astype(np.float32) / 255.
        m = np.zeros((N, H, W), np.float32); m[:, :, X0:] = 255                   # job E: the video known, the panel unknown
        video = torch.from_numpy(v).permute(3, 0, 1, 2)[None]; mask = torch.from_numpy(m)[None, None]
        cv = get_video_to_video_latent(str(cp), video_length=N, sample_size=[H, W], fps=16, ref_image=None)[0]
        g = torch.Generator(device="cuda").manual_seed(a.seed); t = time.time()
        with torch.no_grad():
            out = pipe(c["prompt"] + PANEL, num_frames=N, negative_prompt=WAN_NEGATIVE, height=H, width=W, generator=g, guidance_scale=a.guidance,
                       num_inference_steps=a.steps, video=video, mask_video=mask, control_video=cv, control_camera_video=None, ref_image=None,
                       boundary=boundary, shift=5).videos
        fr = (out[0].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8); sec = time.time() - t
        write_video(od / f"{c['id']}.mp4", fr, fps=16); P = np.stack([decode(f[:, X0:])[0] for f in fr]); np.save(od / f"{c['id']}_tactile.npy", P)
        kv = float(np.abs(fr[:, :, :X0].astype(np.float32) - v[:, :, :X0] * 255).mean())     # the given video must come back
        (od / f"{c['id']}.json").write_text(json.dumps(dict(case=c, model=a.model, lora=[str(lo), str(hi)], steps=a.steps, guidance=a.guidance, seed=a.seed,
                                                           ref_image=False, seconds=round(sec, 1), known_mae=round(kv, 2),
                                                           touch_units="N" if a.model == "f2" else "relative F / S"), indent=1))
        print(a.model, c["id"], f"{sec:.0f}s known-part MAE {kv:.2f} total max {P.sum(1).max():.2f}", flush=True)
    print("group done", a.model, a.group, flush=True)


if __name__ == "__main__":
    main()
