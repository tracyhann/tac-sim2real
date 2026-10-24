"""Generate video + touch with one HF25 arm (train.sh part ev_<ARM>_<CK>; root = env R2S2R, see 2026-10-24/README.md): both experts' LoRAs at step CK merged into
the W pipeline, the low-noise expert's touch head (tactile_head-CK.pt, hf25_patch.TactileHead) on top, every case of
dataset/test_cases.json that has a control in this arm's format.
Job = the training job: frame 0 known, frames 1-80 generated, control = the arm's control. Settings = the dual-mode G protocol
(dual_mode_infer_diag 2026-10-23): CFG 6, the start frame also as reference image, 50 steps, shift 5, the framework's standard negative prompt; seed 0.
Touch: a forward pre-hook on the low-noise expert's output Head keeps the tokens of the LAST denoising step (t < boundary, so
the low expert); the conditional half of the CFG batch goes through the head with the start's frame-0 touch -> (81, 880) u ->
F / S. No pixels anywhere.
Writes outputs/R2S2V_mug/hf25/out/<ARM>c<CK>/<case>_s<seed>.{mp4, _tactile.npy (RELATIVE touch F / S), json}; held-out
cases also go to A_baseline/gen/w-hf25<arm>c<CK>-<real|sim>/<clip>/seed<s>.mp4 (the video half, for the authors' IoU scorer). Skips cases done.
  envs/videoxfun/bin/python predict_hf25.py --arm FHN --ckpt 500 [--cases G_real_55494490 ...] [--seeds 0]"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from w_pipe import ROOT as R, NEG_PROMPT, load_pipe, lora_device, place_pipe, write_video  # noqa: E402  (root = env R2S2R; W_GPU_MEMORY_MODE = offload)
from hf25_patch import TactileHead, attach_capture, rel_from_u, video_tokens  # noqa: E402

O = R / "outputs/R2S2V_mug/hf25"; DS = O / "dataset"; GEN_DIR = R / "outputs/A_baseline/gen"      # gen/: the video half for the (cluster-only) mug / hand IoU scorer
H, W, N = 480, 832, 81; HW = (H // 16) * (W // 16); F_LAT = (N - 1) // 4 + 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=["FHN", "SVO", "CDS"]); ap.add_argument("--ckpt", type=int, required=True)
    ap.add_argument("--cases", nargs="*", default=None); ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--guidance", type=float, default=6.0); ap.add_argument("--steps", type=int, default=50); ap.add_argument("--no_ref", action="store_true")
    ap.add_argument("--suffix", default="", help="output tag suffix for probes (e.g. _noref); default = the arm's main output")
    a = ap.parse_args(); tag = f"{a.arm}c{a.ckpt}{a.suffix}"; od = O / "out" / tag; od.mkdir(parents=True, exist_ok=True)
    lo, hi, hd = O / f"{a.arm}_low" / f"checkpoint-{a.ckpt}.safetensors", O / f"{a.arm}_high" / f"checkpoint-{a.ckpt}.safetensors", O / f"{a.arm}_low" / f"tactile_head-{a.ckpt}.pt"
    assert lo.exists() and hi.exists() and hd.exists(), (lo, hi, hd)
    cases = [c for c in json.loads((DS / "test_cases.json").read_text()) if a.cases is None or c["case"] in a.cases]
    miss = [c["case"] for c in cases if not c["control"].get(a.arm)]
    if miss:
        print("no control in this arm's format (skipped):", miss, flush=True)
    todo = [(c, s) for c in cases if c["control"].get(a.arm) for s in a.seeds if not (od / f"{c['case']}_s{s}.json").exists()]
    print(tag, len(todo), "cases to run", flush=True)
    if not todo:
        return
    pipe, vae, sch, boundary = load_pipe("w")                                          # also puts VideoX-Fun on sys.path
    from videox_fun.utils import get_image_latent, get_video_to_video_latent, merge_lora
    pipe = merge_lora(pipe, str(lo), 1.0, device=lora_device(), dtype=torch.bfloat16)                 # transformer = low-noise expert
    pipe = merge_lora(pipe, str(hi), 1.0, device=lora_device(), dtype=torch.bfloat16, sub_transformer_name="transformer_2")
    pipe = place_pipe(pipe)                                                                           # no-op unless W_GPU_MEMORY_MODE is set
    head = TactileHead(pipe.transformer.head.dim).cuda().eval(); head.load_state_dict(torch.load(hd, map_location="cpu")); cap = attach_capture(pipe.transformer)
    for c, seed in todo:
        st = cv2.imread(str(DS / c["start"]))[..., ::-1]; assert st.shape == (H, W, 3), (c["start"], st.shape)
        u0 = np.load(DS / c["tactile"])["u"][0].astype(np.float32)
        v = np.repeat(st[None].astype(np.float32) / 255., N, 0); m = np.zeros((N, H, W), np.float32); m[1:] = 255   # frame 0 known, the rest generated
        video = torch.from_numpy(v).permute(3, 0, 1, 2)[None]; mask = torch.from_numpy(m)[None, None]
        cv = get_video_to_video_latent(str(DS / c["control"][a.arm]), video_length=N, sample_size=[H, W], fps=16, ref_image=None)[0]
        ref = None if a.no_ref else get_image_latent(str(DS / c["start"]), sample_size=[H, W])
        g = torch.Generator(device="cuda").manual_seed(seed); t = time.time(); cap.clear()
        with torch.no_grad():
            out = pipe(c["text"], num_frames=N, negative_prompt=NEG_PROMPT, height=H, width=W, generator=g, guidance_scale=a.guidance,
                       num_inference_steps=a.steps, video=video, mask_video=mask, control_video=cv, control_camera_video=None, ref_image=ref,
                       boundary=boundary, shift=5).videos
            x = cap[0][-1:]; e = cap[1][-1:] if cap[1] is not None else None   # last call = last step (low expert); CFG batch = [uncond, cond]
            u = head(video_tokens(x, HW, F_LAT, not a.no_ref), HW, torch.from_numpy(u0)[None].cuda(), e)[0].float().cpu().numpy()
        fr = (out[0].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8); sec = time.time() - t
        name = f"{c['case']}_s{seed}"; write_video(od / f"{name}.mp4", fr, fps=16); np.save(od / f"{name}_tactile.npy", rel_from_u(u))
        kv = float(np.abs(fr[0].astype(np.float32) - st.astype(np.float32)).mean())
        if c["kind"] == "heldout":
            gd = GEN_DIR / f"w-hf25{a.arm.lower()}c{a.ckpt}{a.suffix}-{c['source']}" / c["clip"]; gd.mkdir(parents=True, exist_ok=True)
            write_video(gd / f"seed{seed}.mp4", fr, fps=16)
            (gd / f"seed{seed}.json").write_text(json.dumps(dict(model=gd.parent.name, clip=c["clip"], seed=seed, width=W, height=H, frames=N, fps=16, steps=a.steps,
                                                                  control=f"HF25 LoRA {a.arm} step {a.ckpt}: {c['control'][a.arm]}; start = {c['source']} frame 0 (+ its touch to the head)",
                                                                  prompt=c["text"]), indent=1))
        (od / f"{name}.json").write_text(json.dumps(dict(case=c, arm=a.arm, checkpoint=a.ckpt, seed=seed, seconds=round(sec, 1), frame0_mae=kv, guidance=a.guidance,
                                                       steps=a.steps, ref_image=not a.no_ref, tokens=list(x.shape), touch_total_rel_max=float(rel_from_u(u).sum(1).max())), indent=1))
        print(tag, name, f"{sec:.0f}s frame-0 MAE {kv:.2f} touch max total {rel_from_u(u).sum(1).max():.2f} S", flush=True)


if __name__ == "__main__":
    main()
