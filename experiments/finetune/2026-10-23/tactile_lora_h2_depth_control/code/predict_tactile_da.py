"""Generate video + tactile with the overfit LoRA (Wan2.2-Fun-A14B-Control, LoRA on both experts), on the 1088 x 480
canvas: control = the clip's control canvas (scene depth + shadowless sim hand / mug | black panel), start + reference =
the first canvas frame of the chosen source (real frame 0 + real tactile, or generated frame 0 + sim tactile), prompt =
that source's training prompt (e1: tag "Real recording." / "Simulated touch."; e2: one untagged prompt, so the two runs
per clip differ only in the start frame's source). Then:
  - the video half (832 x 480) goes to A_baseline/gen/<tag>/<clip>/seed<k>.mp4 so evaluate.py / labeladh.py score it;
  - the panel is decoded to taxels (81, 880) and scored against the real AND the simulated tactile (score_tactile.py).
  python predict_tactile.py --ckpt 600 --clips tsmug-edf79415 tsmug-53780879 --sources gt gen_s0 --seeds 0"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from fun_zeroshot import GEN_DIR, R, WAN_NEGATIVE, dump_json, load_pipe, write_video  # noqa: E402
from score_tactile import scores  # noqa: E402
from tactile_panel import decode  # noqa: E402

LT = Path(R) / "outputs/R2S2V_mug/lora_tactile"; DS = LT / "dataset"; H, W, N = 480, 1088, 81


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", type=int, required=True); ap.add_argument("--clips", nargs="+"); ap.add_argument("--sources", nargs="+", default=["gt", "gen_s0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0]); ap.add_argument("--weight", type=float, default=1.0); ap.add_argument("--run", default="e1"); ap.add_argument("--meta", default="metadata.json")
    ap.add_argument("--ds", default="dataset")   # dataset_ho: the held-out split (its metadata_test.json lists the test clips' starts)
    a = ap.parse_args(); global DS; DS = LT / a.ds; meta = {(m["clip"], m["source"]): m for m in json.loads((DS / a.meta).read_text())}
    lo, hi = LT / f"{a.run}_low" / f"checkpoint-{a.ckpt}.safetensors", LT / f"{a.run}_high" / f"checkpoint-{a.ckpt}.safetensors"
    pipe, vae, sch, boundary = load_pipe("a14b")
    from videox_fun.utils import get_image_latent, get_image_to_video_latent, get_video_to_video_latent, merge_lora
    pipe = merge_lora(pipe, str(lo), a.weight, device="cuda", dtype=torch.bfloat16)
    pipe = merge_lora(pipe, str(hi), a.weight, device="cuda", dtype=torch.bfloat16, sub_transformer_name="transformer_2")
    for clip in a.clips:
        tac = np.load(DS / f"tactile/{clip}.npz")
        for src in a.sources:
            m = meta[(clip, src)]; tag = f"wan22fun-a14b-tact{a.run}c{a.ckpt}da-{'real' if src == 'gt' else 'sim'}"; start = str(DS / "start" / f"{clip}_{src}.png")
            for seed in a.seeds:
                o = GEN_DIR / tag / clip; o.mkdir(parents=True, exist_ok=True)
                if (o / f"seed{seed}.json").exists():
                    continue
                with torch.no_grad():
                    inpaint, inpaint_mask, _ = get_image_to_video_latent(start, None, video_length=N, sample_size=[H, W]); ref = get_image_latent(start, sample_size=[H, W])
                    cv = get_video_to_video_latent(str(DS / m["control_file_path"]), video_length=N, sample_size=[H, W], fps=16, ref_image=None)[0]
                    g = torch.Generator(device="cuda").manual_seed(seed); t = time.time()
                    vid = pipe(m["text"], num_frames=N, negative_prompt=WAN_NEGATIVE, height=H, width=W, generator=g, guidance_scale=6.0, num_inference_steps=50,
                               video=inpaint, mask_video=inpaint_mask, control_video=cv, control_camera_video=None, ref_image=ref, boundary=boundary, shift=5).videos
                fr = (vid[0].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8); sec = time.time() - t
                (LT / "out" / tag).mkdir(parents=True, exist_ok=True); write_video(LT / "out" / tag / f"{clip}_seed{seed}.mp4", fr, fps=16)
                write_video(o / f"seed{seed}.mp4", np.ascontiguousarray(fr[:, :, :832]), fps=16)
                P = np.stack([decode(f[:, 832:])[0] for f in fr]); np.save(LT / "out" / tag / f"{clip}_seed{seed}_tactile.npy", P)
                sc = dict(vs_real=scores(P, tac["gt"]), vs_sim=scores(P, tac["sim"]))
                dump_json(o / f"seed{seed}.json", dict(model=tag, clip=clip, seed=seed, width=832, height=480, frames=int(len(fr)), fps=16, steps=50, guidance=6.0, shift=5,
                                                        lora=dict(low=str(lo), high=str(hi), weight=a.weight), start=start, prompt=m["text"], seconds=round(sec, 1),
                                                        control="video + tactile LoRA (overfit): control canvas, start = first canvas frame", tactile=sc))
                print(tag, clip, seed, f"{sec:.0f}s", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in sc.items()}, flush=True)


if __name__ == "__main__":
    main()
