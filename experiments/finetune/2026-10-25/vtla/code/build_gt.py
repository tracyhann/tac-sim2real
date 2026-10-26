"""AUTHORS' BUILD SCRIPT: the held-out GT test set (README section 11; user 2026-10-26: "测试集用我们的 gt recording heldout"):
the recorded episodes 4397541c and 55494490 (never in any training row, never a source of a perturbation). Per episode one row:
video = the REAL head-camera clip (outputs/A_baseline/clips/tsmug-<ep>/real.mp4, 5 s from the reconstruction's t0, 30 fps; resampled
to 81 frames at 16 fps = video/gt_<ep>.mp4),
state = the reconstruction's measured hand pose (runs/datagen/gt_<ep>/hand_pose.npz, from datagen/vision_gt_step.sh), prompt = the
plain task instruction; labels vtla/<id>.npz (act_obs / act_cmd of the recording) for the open-loop numbers. The physics test re-runs
the recording (cf_relocate variant rec) with the policy's finger targets.
  R2S2R=<root> envs/scenemug/bin/python build_gt.py"""
import json
import os
import sys
from pathlib import Path

import av

import build_vtla_labels as B

R = Path(os.environ.get("R2S2R", ".")).resolve(); H = R / "outputs/R2S2V_mug/hf25"; S = R / "experiments/scene_mug_dg"; DS = H / "dataset_gt"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "2026-10-24" / "_common" / "code"))
from prompts import ACTION  # noqa: E402


def main():
    (DS / "vtla").mkdir(parents=True, exist_ok=True); rows = []; checks = {}
    for ep in ("4397541c", "55494490"):
        src = R / "outputs/A_baseline/clips" / f"tsmug-{ep}" / "real.mp4"; d = S / "runs/datagen" / f"gt_{ep}"
        with av.open(str(src)) as c:                                         # the real clip is 30 fps (151 frames for 5 s): 16 fps, 81 frames
            fps = float(c.streams.video[0].average_rate); fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        pick = [min(int(round(k / 16 * fps)), len(fr) - 1) for k in range(81)]; v = DS / "video" / f"gt_{ep}.mp4"; v.parent.mkdir(exist_ok=True)
        out = av.open(str(v), "w"); st = out.add_stream("h264", rate=16); st.height, st.width = fr[0].shape[:2]; st.pix_fmt = "yuv420p"; st.options = {"crf": "14"}
        for k in pick:
            for pkt in st.encode(av.VideoFrame.from_ndarray(fr[k], format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
        out.close(); print(ep, f"real clip {len(fr)} frames at {fps:g} fps -> 81 at 16 fps", flush=True)
        r = dict(id=f"gt_{ep}", video=str(v), hand_pose=str(d / "hand_pose.npz"), tactile=str(d / "tactile.npz"), sim=f"runs/counterfactual/{ep}_auto_mano/relocate/variants/rec/sim",
                 kind="recorded", variant="rec", group="gt", params={}, text=ACTION, text_priv=ACTION, episode=ep)
        checks[r["id"]] = B.build_row(DS, r, {}); rows.append(r)
    (DS / "rows.json").write_text(json.dumps(rows, indent=1)); (DS / "metadata_vtla.json").write_text(json.dumps(dict(rows=rows), indent=1))
    (DS / "vtla" / "check.json").write_text(json.dumps(checks, indent=1)); print(json.dumps({k: (v["ok"], v.get("q_cmd_source")) for k, v in checks.items()}))


if __name__ == "__main__":
    main()
