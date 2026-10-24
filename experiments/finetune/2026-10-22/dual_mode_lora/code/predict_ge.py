"""Held-out inference with a dual-mode LoRA (run_ge.sh part ev_<RUN>_<P>: run U / G / E / R at epoch P, both experts
merged into Wan2.2-Fun-A14B-Control). The job is picked by the inpaint mask, exactly as in training (ge_mask.py):
  E (runs U, E, R)  known = the clip's 81 video frames, unknown = the touch panel of every frame; control = the clip's
                    Depth Anything depth (every held-out real clip: participant actor_01 + the twins 55494490 / 4397541c)
                    or the twin's sim control (twins only); also the twins' zero-shot generated sim videos under their sim
                    control (target = the sim touch). 30 denoising steps.
  G (runs U, G)     known = frame 0 (real video, with / without its touch panel), control = the twin's sim control
                    (scene depth + shadowless sim hand / mug); unknown = the rest. 50 steps. The video half also goes to
                    A_baseline/gen/wan22fun-a14b-ge<RUN>p<P>g<t|n>/<clip>/seed0.mp4 for labeladh.py (mug / hand IoU vs sim).
No reference image (full_ref) is passed: training zeroes it in first-frame mode, so passing the start image would feed the
LoRA an input it never saw. The touch-panel input is grey (0.5) wherever it is unknown, so no target pixel enters.
Writes lora_ge/out/<RUN>p<P>/<case>.mp4 (canvas), <case>_tactile.npy (decoded (81, 880) RELATIVE touch, F / S) and <case>.json.
  envs/videoxfun/bin/python predict_ge.py --part ev_U_8 [--only E|G] [--seeds 0]"""
import argparse
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/experiments/R2S2V_mug"))); sys.path.insert(0, str(Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/experiments/R2S2V_mug") / "lora_tactile"))   # absolute: this folder may be reached through a symlink
from fun_zeroshot import GEN_DIR, WAN_NEGATIVE, load_pipe, write_video  # noqa: E402
from panel_v3 import decode  # noqa: E402  (panel v3: RELATIVE touch F / S; the model never sees S)
from ge_cases import cases  # noqa: E402

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); LG = R / "outputs/R2S2V_mug/lora_ge"; DS = LG / "dataset"
H, W, X0, N = 480, 1088, 832, 81
JOBS = {"U": "GE", "G": "G", "E": "E", "R": "E"}


def frames(p):
    c = av.open(str(p)); x = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)]); c.close()
    assert x.shape == (N, H, W, 3), (p, x.shape); return x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True); ap.add_argument("--only", default=None); ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps_e", type=int, default=30); ap.add_argument("--steps_g", type=int, default=50)
    # diagnostics (2026-10-23: E outputs did not keep the given video; known-part MAE 35-55)
    # eval protocol (fixed 2026-10-23 after the diagnostic, part diag): E without guidance (CFG 6 regenerated the GIVEN video
    # colour-distorted, known-part MAE 22-55 even on training rows, vs 2-3 at CFG 1, and inflated the panel); G at CFG 6 with the
    # start canvas as reference image (without it G drifted off the control and invented objects). E never gets the reference
    # image: it would leak frame 0's touch.
    ap.add_argument("--guidance_e", type=float, default=1.0); ap.add_argument("--guidance_g", type=float, default=6.0)
    ap.add_argument("--no_ref_g", action="store_true", help="G without the start canvas as reference image")
    ap.add_argument("--guidance", type=float, default=None, help="override both"); ap.add_argument("--negative", default="wan", choices=["wan", "empty"])
    ap.add_argument("--ref_start", action="store_true", help="also pass the reference image in E (diagnostics only: leaks frame-0 touch)")
    ap.add_argument("--no_lora", action="store_true"); ap.add_argument("--cases", nargs="*", default=None); ap.add_argument("--tag", default=None)
    ap.add_argument("--train_rows", nargs="*", default=[], help="also run E on these TRAINING rows (Depth Anything control): a model / pipeline check")
    a = ap.parse_args(); _, run, P = a.part.split("_"); P = int(P)
    meta = DS / ("metadata_train_real.json" if run == "R" else "metadata_train.json"); n = len(json.loads(meta.read_text())); ck = P * n
    lo, hi = LG / f"{run}_low" / f"checkpoint-{ck}.safetensors", LG / f"{run}_high" / f"checkpoint-{ck}.safetensors"
    assert a.no_lora or (lo.exists() and hi.exists()), (lo, hi)
    todo = [c for c in cases(a.only) if c["job"] in JOBS[run]]; rows = {r["id"]: r for r in json.loads((DS / "rows.json").read_text())}
    if a.cases is not None:
        todo = [c for c in todo if c["case"] in a.cases]
    for rid in a.train_rows:
        r = rows[rid]; todo.append(dict(job="E", case=f"E_{rid}_da_TRAIN", row=rid, control=r.get("control_alt_file_path") or r["control_file_path"], ctrl="da", target="real"))
    od = LG / "out" / (a.tag or f"{run}p{P}"); od.mkdir(parents=True, exist_ok=True)
    todo = [(c, s) for c in todo for s in a.seeds if not (od / f"{c['case']}_s{s}.json").exists()]
    print(run, P, "checkpoint", ck, len(todo), "cases to run", flush=True)
    if not todo:
        return
    pipe, vae, sch, boundary = load_pipe("a14b")
    from videox_fun.utils import get_video_to_video_latent, merge_lora
    if not a.no_lora:
        pipe = merge_lora(pipe, str(lo), 1.0, device="cuda", dtype=torch.bfloat16)
        pipe = merge_lora(pipe, str(hi), 1.0, device="cuda", dtype=torch.bfloat16, sub_transformer_name="transformer_2")
    from videox_fun.utils import get_image_latent
    neg = WAN_NEGATIVE if a.negative == "wan" else ""
    for c, seed in todo:
        r = rows[c["row"]]; v = frames(DS / r["file_path"]).astype(np.float32) / 255.; m = np.zeros((N, H, W), np.float32)
        if c["job"] == "E":
            v[:, :, X0:] = 0.5; m[:, :, X0:] = 255
        else:
            v[1:] = v[:1]; m[1:] = 255
            if not c["touch0"]:
                v[0, :, X0:] = 0.5; m[0, :, X0:] = 255
            v[1:, :, X0:] = 0.5
        video = torch.from_numpy(v).permute(3, 0, 1, 2)[None]; mask = torch.from_numpy(m)[None, None]
        cv = get_video_to_video_latent(str(DS / c["control"]), video_length=N, sample_size=[H, W], fps=16, ref_image=None)[0]
        g = torch.Generator(device="cuda").manual_seed(seed); t = time.time()
        with torch.no_grad():
            use_ref = a.ref_start or (c["job"] == "G" and not a.no_ref_g)
            ref = get_image_latent(str(DS / r["start"]), sample_size=[H, W]) if use_ref else None
            gs = a.guidance if a.guidance is not None else (a.guidance_e if c["job"] == "E" else a.guidance_g)
            out = pipe(r["text"], num_frames=N, negative_prompt=neg, height=H, width=W, generator=g, guidance_scale=gs,
                       num_inference_steps=a.steps_e if c["job"] == "E" else a.steps_g, video=video, mask_video=mask, control_video=cv,
                       control_camera_video=None, ref_image=ref, boundary=boundary, shift=5).videos
        fr = (out[0].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8); sec = time.time() - t
        name = f"{c['case']}_s{seed}"; write_video(od / f"{name}.mp4", fr, fps=16)
        P_ = np.stack([decode(f[:, X0:])[0] for f in fr]); np.save(od / f"{name}_tactile.npy", P_)
        if c["job"] == "E":                                        # the known video half must come back (sanity of the job)
            kv = float(np.abs(fr[:, :, :X0].astype(np.float32) - v[:, :, :X0] * 255).mean())
        else:
            kv = float(np.abs(fr[0, :, :X0].astype(np.float32) - v[0, :, :X0] * 255).mean())
            gd = GEN_DIR / f"wan22fun-a14b-ge{a.tag or f'{run}p{P}'}g{'t' if c['touch0'] else 'n'}" / c["clip"]; gd.mkdir(parents=True, exist_ok=True)
            write_video(gd / f"seed{seed}.mp4", np.ascontiguousarray(fr[:, :, :X0]), fps=16)
            (gd / f"seed{seed}.json").write_text(json.dumps(dict(model=gd.parent.name, clip=c["clip"], seed=seed, width=X0, height=H, frames=N, fps=16, steps=a.steps_g,
                                                                  control=f"dual-mode LoRA {run} epoch {P}, job G, start = real frame 0 {'with' if c['touch0'] else 'without'} its touch panel",
                                                                  prompt=r["text"]), indent=1))
        (od / f"{name}.json").write_text(json.dumps(dict(case=c, seed=seed, run=run, epoch=P, checkpoint=ck, seconds=round(sec, 1), known_mae=kv, guidance=gs,
                                                       negative=a.negative, ref_image=use_ref, no_lora=a.no_lora,
                                                       steps=a.steps_e if c["job"] == "E" else a.steps_g), indent=1))
        print(run, P, name, f"{sec:.0f}s known-part MAE {kv:.2f}", flush=True)


if __name__ == "__main__":
    main()
