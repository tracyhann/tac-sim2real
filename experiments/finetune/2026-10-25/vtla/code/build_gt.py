"""AUTHORS' BUILD SCRIPT: the held-out GT test set (README section 11; user 2026-10-26: "测试集用我们的 gt recording heldout"):
the recorded episodes 4397541c and 55494490 (never in any training row, never a source of a perturbation). Per episode one row:
video = the REAL head-camera clip (outputs/A_baseline/clips/tsmug-<ep>/real.mp4, 5 s from the reconstruction's t0, 30 fps; resampled
to 81 frames at 16 fps = video/gt_<ep>.mp4),
state = the reconstruction's measured hand pose (runs/datagen/gt_<ep>/hand_pose.npz, from datagen/vision_gt_step.sh), prompt = the
plain task instruction; labels vtla/<id>.npz (act_obs / act_cmd of the recording) for the open-loop numbers. The physics test re-runs
the recording (cf_relocate variant rec) with the policy's finger targets.
  R2S2R=<root> envs/scenemug/bin/python build_gt.py
Yaw-fixed twins (D77, 2026-10-26): --suffix _yawfix --ds dataset_gt_yawfix --export gtyf_ --video work. Their windows start
34-67 ms off the canonical ones, so the video is cut from the episode's head video (frame i at the work root's meta t_head[i],
$SCENE_CFG) at the export's times t0 + k/16, checked against the canonical test video, named gtyf_<ep>.mp4 (the frame cache keys on the file name). Row ids stay
gt_<ep>."""
import argparse
import json
import os
import sys
from pathlib import Path

import av

import build_vtla_labels as B

R = Path(os.environ.get("R2S2R", ".")).resolve(); H = R / "outputs/R2S2V_mug/hf25"; S = R / "experiments/scene_mug_dg"; DS = H / "dataset_gt"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "2026-10-24" / "_common" / "code"))
from prompts import ACTION  # noqa: E402


def work_frames(ep, suffix, d, canon=None, raw=None):
    """the reconstruction's head frames (the episode's rgb_head video, frame i at meta t_head[i]) at the export's video times
    t0 + k / 16. canon = the canonical test video of this episode: the same method at the canonical export's times must
    reproduce its frames (offset 0 must be the best match), which checks the time base"""
    import numpy as np
    sys.path.insert(0, str(S)); os.chdir(S)
    from cliprep.core import Clip
    c = Clip(S / "configs" / "clips" / f"{ep}_auto_mano{suffix}.yaml"); th = np.asarray(c.th)
    vf = (Path(raw) / c.ep.id if raw else c.ep.dir) / "rgb_head.mp4"                      # --raw: a data root that has the head videos
    with av.open(str(vf)) as cv_:
        allf = [f.to_ndarray(format="bgr24") for f in cv_.decode(video=0)]
    assert len(allf) >= len(th) - 2, (len(allf), len(th))
    pick = lambda ts: [min(int(np.argmin(np.abs(th - t))), len(allf) - 1) for t in ts]   # noqa: E731
    ts = c.t0 + np.arange(81) / 16; idx = pick(ts)
    hp = np.load(d / "hand_pose.npz"); assert np.abs(hp["t_sim"] - ts).max() < 0.02, (ep, float(np.abs(hp["t_sim"] - ts).max()))
    print(ep, f"head frames {idx[0]}..{idx[-1]} at t0 {c.t0:.3f}, max |t_head - t| {max(abs(th[i] - t) for i, t in zip(idx, ts)) * 1e3:.1f} ms", flush=True)
    if canon is not None:
        tc = np.load(S / "runs/datagen" / f"gt_{ep}" / "hand_pose.npz")["t_sim"]; ic = pick(tc)
        with av.open(str(canon)) as cv_:
            ref = [f.to_ndarray(format="rgb24").astype(np.float32) for f in cv_.decode(video=0)]
        for k in (0, 40, 80):
            dd = {o: float(np.abs(allf[min(max(ic[k] + o, 0), len(allf) - 1)][..., ::-1].astype(np.float32) - ref[k]).mean()) for o in (-2, -1, 0, 1, 2)}
            print(f"  check vs {canon.name} frame {k}: mean |diff| by offset {dd}", flush=True)
    return [np.ascontiguousarray(allf[i][..., ::-1]) for i in idx], list(range(81))


def main():
    global DS
    ap = argparse.ArgumentParser(); ap.add_argument("--suffix", default=""); ap.add_argument("--ds", default="dataset_gt"); ap.add_argument("--export", default="gt_")
    ap.add_argument("--video", choices=("real", "work"), default="real"); ap.add_argument("--raw", default=None); ap.add_argument("--eps", nargs="+", default=["4397541c", "55494490"]); a = ap.parse_args(); DS = H / a.ds
    (DS / "vtla").mkdir(parents=True, exist_ok=True); rows = []; checks = {}
    for ep in a.eps:
        d = S / "runs/datagen" / f"{a.export}{ep}"
        if a.video == "work":
            fr, pick = work_frames(ep, a.suffix, d, canon=H / "dataset_gt" / "video" / f"gt_{ep}.mp4", raw=a.raw)
        else:
            src = R / "outputs/A_baseline/clips" / f"tsmug-{ep}" / "real.mp4"
            with av.open(str(src)) as c:                                     # the real clip is 30 fps (151 frames for 5 s): 16 fps, 81 frames
                fps = float(c.streams.video[0].average_rate); fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
            pick = [min(int(round(k / 16 * fps)), len(fr) - 1) for k in range(81)]
        v = DS / "video" / f"{a.export.rstrip('_')}_{ep}.mp4"; v.parent.mkdir(exist_ok=True)
        out = av.open(str(v), "w"); st = out.add_stream("h264", rate=16); st.height, st.width = fr[0].shape[:2]; st.pix_fmt = "yuv420p"; st.options = {"crf": "14"}
        for k in pick:
            for pkt in st.encode(av.VideoFrame.from_ndarray(fr[k], format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
        out.close(); print(ep, f"video {v.name}: {len(pick)} frames at 16 fps", flush=True)
        r = dict(id=f"gt_{ep}", video=str(v), hand_pose=str(d / "hand_pose.npz"), tactile=str(d / "tactile.npz"), sim=f"runs/counterfactual/{ep}_auto_mano{a.suffix}/relocate/variants/rec/sim",
                 kind="recorded", variant="rec", group="gt", params={}, text=ACTION, text_priv=ACTION, episode=ep, twin=f"{ep}_auto_mano{a.suffix}")
        checks[r["id"]] = B.build_row(DS, r, {}); rows.append(r)
    (DS / "rows.json").write_text(json.dumps(rows, indent=1)); (DS / "metadata_vtla.json").write_text(json.dumps(dict(rows=rows), indent=1))
    (DS / "vtla" / "check.json").write_text(json.dumps(checks, indent=1)); print(json.dumps({k: (v["ok"], v.get("q_cmd_source")) for k, v in checks.items()}))


if __name__ == "__main__":
    main()
