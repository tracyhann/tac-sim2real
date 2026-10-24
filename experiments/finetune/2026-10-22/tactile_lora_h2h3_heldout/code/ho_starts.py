"""Held-out generated starts (2026-10-22): every test clip in dataset_ho/metadata_test.json with a real-start row but no
generated-start row gets one once its zero-shot head-c5 seed 0 exists: frame 0 of that video (832 x 480, INTER_AREA) beside
the simulated touch at frame 0 (tactile_panel.canvas), as scene_mug r2s2v/lora_tactile_data.py builds it on the Mac. The
eval parts h2_ev* / h3_ev* run it after fun_zeroshot.py, so the start never waits for a separate card.
  envs/videoxfun/bin/python lora_tactile/ho_starts.py"""
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from tactile_panel import canvas  # noqa: E402

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); DS = R / "outputs/R2S2V_mug/lora_tactile/dataset_ho"
GEN = R / "outputs/A_baseline/gen/wan22fun-a14b-head-c5"; W, H = 832, 480


def main():
    rows = json.loads((DS / "metadata_test.json").read_text()); have = {(r["clip"], r["source"]) for r in rows}
    for r in [r for r in rows if r["source"] == "gt" and (r["clip"], "gen_s0") not in have]:
        c, v = r["clip"], GEN / r["clip"] / "seed0.mp4"
        if not v.exists():
            print("ho_starts:", c, "has no zero-shot seed 0 yet"); continue
        cont = av.open(str(v)); f0 = next(cont.decode(video=0)).to_ndarray(format="rgb24"); cont.close()
        cv = canvas(cv2.resize(f0, (W, H), interpolation=cv2.INTER_AREA), np.load(DS / r["tactile"])["sim"][0])
        cv2.imwrite(str(DS / "start" / f"{c}_gen_s0.png"), cv[..., ::-1])
        rows.append({**r, "file_path": None, "source": "gen_s0", "tactile_key": "sim"}); print("ho_starts: added", c, "gen_s0")
    (DS / "metadata_test.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
