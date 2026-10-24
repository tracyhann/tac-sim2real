"""Overfit dataset for the video + tactile LoRA on Wan2.2-Fun-A14B-Control (user 2026-10-22: "use the GT video + GT
tactile and the best generated videos + simulated tactile as pairs, overfit a LoRA as RoboRender does, and make the
world model output the tactile matrix too"). Per clip, three pairs on a 1088 x 480 canvas (video 832 x 480 | tactile
panel 256 x 480, r2s2v/tactile_panel.py), 81 frames at 16 fps:
  gt      the real head video                      + the real glove's taxels (touch(): N above each taxel's baseline)
  gen_s0  the best generated video (seed 0: scene depth + shadowless sim hand / mug, A14B zero-shot) + the simulated taxels
  gen_s1  the same, seed 1                                                                           (replay txc30 -> glove layout)
The control is the same for all three: the generation's control video (real frame-0 scene depth + shadowless sim hand /
mug) with a black panel - the tactile is OUTPUT only. A prompt tag tells the sources apart ("Real recording." /
"Simulated touch."), so one model can be asked for either. The first frame (video frame 0 + its tactile panel) is the
start image at inference.
Writes runs/clips/r2s2v_lora/dataset/{train,control,start,tactile}/ + metadata.json (relative paths).
  .venv/bin/python r2s2v/lora_tactile_data.py
Held-out split (user 2026-10-22: the two-clip LoRA was trained and scored on the same clips): --train / --test name the
clips (tsmug-<ep8>=<cliprep run>), --control picks the control video, --gen the zero-shot generations (seeds 0 / 1 for
the training clips' sim rows; seed 0 of a test clip is only its "generated frame 0" start), --out the dataset dir.
Writes metadata_mixed.json (untagged prompt, real + sim rows of the TRAINING clips), metadata_real.json (their real
rows) and metadata_test.json (held-out clips: start gt / gen_s0, never trained on); test/<clip>_gt.mp4 is the held-out
reference canvas (real video + real touch).
  .venv/bin/python r2s2v/lora_tactile_data.py --out runs/clips/r2s2v_lora/dataset_ho --control views5 \
      --gen runs/clips/r2s2v_lora/heldout/gen/wan22fun-a14b-head-c5 \
      --train tsmug-edf79415=edf79415_11-16_mano tsmug-53780879=53780879_auto_mano --test tsmug-55494490=55494490_auto_mano
  ... later --test tsmug-4397541c=4397541c_auto_mano alone adds a held-out clip without touching the training rows.
(f8f7ed78 / 9811b8f8 were planned as a test and a training clip; their automatic twins fail the hold test.)"""
import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "r2s2v"))
from cliprep.core import Clip, touch  # noqa: E402
from tactile_panel import F_MAX, PH, PW, canvas  # noqa: E402

OUT = HERE / "runs/clips/r2s2v_lora/dataset"; GEN = HERE / "runs/clips/r2s2v_page/pull/outputs/A_baseline/gen/wan22fun-a14b-depthsimrgbns"
CLIPS = {"tsmug-edf79415": "edf79415_11-16_mano", "tsmug-53780879": "53780879_auto_mano"}
N, FPS, W, H = 81, 16, 832, 480
TAG = {"gt": "Real recording.", "gen": "Simulated touch."}
PANEL = " On the right, a grey touch map of the left glove: each cell is one sensor, brighter is pressed harder; the bars below are the total force per finger."


def frames(p):
    c = av.open(str(p)); f = [cv2.resize(x.to_ndarray(format="rgb24"), (W, H), interpolation=cv2.INTER_AREA) for x in c.decode(video=0)]; c.close()
    assert len(f) >= N, (p, len(f)); return f[:N]


def real_frames(c, ts):
    """the real head video at the canvas times: for each t the head frame nearest in time (wmexport's rule), 832 x 480.
    wm/real.mp4 is 30 fps (151 frames over 5 s): its first 81 frames are only t0 .. t0 + 2.7 s. Datasets built before
    2026-10-22 13:30 CDT took those, so their real rows ran 1.9x slower than their control and touch panel (D68)."""
    rgb = np.load(c.work / "frames" / "rgb.npy", mmap_mode="r")                   # stored BGR
    return [cv2.resize(np.ascontiguousarray(np.array(rgb[int(np.argmin(np.abs(c.th - t)))])[..., ::-1]), (W, H), interpolation=cv2.INTER_AREA) for t in ts]


def write(p, fr):
    p.parent.mkdir(parents=True, exist_ok=True); o = av.open(str(p), "w"); st = o.add_stream("h264", rate=FPS)
    st.width, st.height, st.pix_fmt = fr[0].shape[1], fr[0].shape[0], "yuv420p"; st.options = {"crf": "14"}
    for f in fr:
        for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            o.mux(pk)
    for pk in st.encode():
        o.mux(pk)
    o.close()


CONTROL = {"simrgb_ns": "fun_depth_simrgb_ns.mp4",          # frame-0 real scene depth + shadowless sim hand / mug (the 2-clip runs e1-e3)
           "views5": "views5/head/fun_ctrl.mp4"}           # the episode-fused twin's depth + shadowless sim hand / mug + coaster / mark props (r2s2v/views5.py)


def clip_tactile(c, ts):
    """real glove (81, 880) N and the replay's simulated taxels at the video times (linear in time), in glove columns"""
    G = touch(c, ts)["taxel"].astype(np.float32)
    z = np.load(c.out / "sim" / "replay.npz"); tj = json.loads((c.hand_dir / "taxels.json").read_text())["taxels"]; idx = np.array([t["idx"] for t in tj])
    S30 = np.maximum(z["txc30"], 0); S = np.zeros((N, 880), np.float32)
    for k, t in enumerate(ts):
        j = np.searchsorted(z["t30"], t); j0, j1 = max(j - 1, 0), min(j, len(z["t30"]) - 1); w = 0 if j0 == j1 else (t - z["t30"][j0]) / (z["t30"][j1] - z["t30"][j0])
        S[k, idx] = (1 - w) * S30[j0] + w * S30[j1]
    return G, S


def build(out, train, test, control, gen):
    """the held-out dataset: rows for the training clips, starts + references for the test clips (see the docstring)"""
    mixed, test_rows = [], []
    for split, clips in (("train", train), ("test", test)):
        for clip, run in clips.items():
            c = Clip(HERE / f"configs/clips/{run}.yaml"); ts = c.t0 + np.arange(N) / FPS
            G, S = clip_tactile(c, ts); (out / "tactile").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "tactile" / f"{clip}.npz", gt=G, sim=S, t=ts, f_max=F_MAX)
            ctrl = frames(c.out / "wm" / CONTROL[control]); write(out / "control" / f"{clip}.mp4", [np.concatenate([f, np.zeros((PH, PW, 3), np.uint8)], 1) for f in ctrl])
            pj = gen / clip / "seed0.json" if (gen / clip / "seed0.json").exists() else sorted(gen.glob("*/seed0.json"))[0]   # one prompt for every clip (setup_r2s.PROMPT)
            base = json.loads(pj.read_text())["prompt"]; text = f"{base}{PANEL}"
            srcs = [("gt", None, G)] + [(f"gen_s{s}", gen / clip / f"seed{s}.mp4", S) for s in ((0, 1) if split == "train" else (0,))]
            if split == "test" and not (gen / clip / "seed0.mp4").exists():       # its generated start comes later (Vision lora_tactile/ho_starts.py, same rule)
                print("test", clip, "no zero-shot seed 0 yet: gt start only"); srcs = srcs[:1]
            for name, src, tac in srcs:
                vf = real_frames(c, ts) if src is None else frames(src)
                fr = [canvas(f, tac[k]) for k, f in enumerate(vf)]; (out / "start").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / "start" / f"{clip}_{name}.png"), fr[0][..., ::-1])
                row = dict(control_file_path=f"control/{clip}.mp4", text=text, type="video", width=W + PW, height=H, clip=clip, source=name,
                           tactile=f"tactile/{clip}.npz", tactile_key="gt" if name == "gt" else "sim", control=control, split=split)
                if split == "train":
                    vid = f"train/{clip}_{name}.mp4"; write(out / vid, fr); mixed.append(dict(file_path=vid, **row))
                else:                                                    # held out: only the starts (+ the real canvas as the reference answer)
                    vid = f"test/{clip}_{name}.mp4"
                    if name == "gt":
                        write(out / vid, fr)
                    test_rows.append(dict(file_path=vid if name == "gt" else None, **row))
                print(split, clip, name, "tactile total max", round(float(tac.sum(1).max()), 1), "N")
    if not train:                                                # test-only: add held-out clips to a built dataset (its training rows untouched)
        old = json.loads((out / "split.json").read_text()); mixed = json.loads((out / "metadata_mixed.json").read_text()); train = old["train"]
        test_rows = [r for r in json.loads((out / "metadata_test.json").read_text()) if r["clip"] not in test] + test_rows; test = {**old["test"], **test}
    assert not {r["clip"] for r in mixed} & {r["clip"] for r in test_rows}, "a test clip is in the training rows"
    (out / "metadata_mixed.json").write_text(json.dumps(mixed, indent=1)); (out / "metadata_real.json").write_text(json.dumps([r for r in mixed if r["source"] == "gt"], indent=1))
    (out / "metadata_test.json").write_text(json.dumps(test_rows, indent=1))
    (out / "split.json").write_text(json.dumps(dict(train=train, test=test, control=control, gen=str(gen.relative_to(HERE)), n_mixed=len(mixed),
                                                     n_real=sum(r["source"] == "gt" for r in mixed)), indent=1))
    print(out, "mixed", len(mixed), "real", sum(r["source"] == "gt" for r in mixed), "test starts", len(test_rows))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--control", default="simrgb_ns", choices=sorted(CONTROL)); ap.add_argument("--gen")
    ap.add_argument("--train", nargs="*", default=[]); ap.add_argument("--test", nargs="*", default=[])
    a = ap.parse_args()
    if a.train or a.test:                                        # --test alone: add held-out clips to the dataset in --out
        pairs = lambda xs: dict(x.split("=", 1) for x in xs)
        return build(Path(a.out) if Path(a.out).is_absolute() else HERE / a.out, pairs(a.train), pairs(a.test), a.control, HERE / a.gen)
    meta = []
    for clip, run in CLIPS.items():
        c = Clip(HERE / f"configs/clips/{run}.yaml"); ts = c.t0 + np.arange(N) / FPS; prompt = json.loads((c.out / "wm" / "real.json").read_text())["prompt"] if (c.out / "wm" / "real.json").exists() else None
        G = touch(c, ts)["taxel"].astype(np.float32)                                    # real glove, (81, 880) N
        z = np.load(c.out / "sim" / "replay.npz"); tj = json.loads((c.hand_dir / "taxels.json").read_text())["taxels"]; idx = np.array([t["idx"] for t in tj])
        S30 = np.maximum(z["txc30"], 0); S = np.zeros((N, 880), np.float32)
        for k, t in enumerate(ts):                                                       # sim taxels at the video times (linear in time), into glove columns
            j = np.searchsorted(z["t30"], t); j0, j1 = max(j - 1, 0), min(j, len(z["t30"]) - 1); w = 0 if j0 == j1 else (t - z["t30"][j0]) / (z["t30"][j1] - z["t30"][j0])
            S[k, idx] = (1 - w) * S30[j0] + w * S30[j1]
        np.savez_compressed(OUT / "tactile" / f"{clip}.npz", gt=G, sim=S, t=ts, f_max=F_MAX) if (OUT / "tactile").mkdir(parents=True, exist_ok=True) is None else None
        ctrl = frames(c.out / "wm" / "fun_depth_simrgb_ns.mp4"); write(OUT / "control" / f"{clip}.mp4", [np.concatenate([f, np.zeros((PH, PW, 3), np.uint8)], 1) for f in ctrl])
        base = prompt or json.loads((HERE / "runs/clips/r2s2v_page/pull/outputs/A_baseline/gen/wan22fun-a14b-depthsimrgbns" / clip / "seed0.json").read_text())["prompt"]
        for name, src, tac, tag in (("gt", None, G, "gt"), ("gen_s0", GEN / clip / "seed0.mp4", S, "gen"), ("gen_s1", GEN / clip / "seed1.mp4", S, "gen")):
            vf = real_frames(c, ts) if src is None else frames(src)
            fr = [canvas(f, tac[k]) for k, f in enumerate(vf)]; vid = f"train/{clip}_{name}.mp4"; write(OUT / vid, fr)
            (OUT / "start").mkdir(parents=True, exist_ok=True); cv2.imwrite(str(OUT / "start" / f"{clip}_{name}.png"), fr[0][..., ::-1])
            meta.append(dict(file_path=vid, control_file_path=f"control/{clip}.mp4", text=f"{TAG[tag]} {base}{PANEL}", type="video", width=W + PW, height=H,
                             clip=clip, source=name, tactile=f"tactile/{clip}.npz", tactile_key="gt" if tag == "gt" else "sim"))
            print(clip, name, "frames", len(fr), "tactile total max", round(float(tac.sum(1).max()), 1), "N")
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=1)); print(OUT / "metadata.json", len(meta), "samples")


if __name__ == "__main__":
    main()
