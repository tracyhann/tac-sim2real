"""Small web copies of the side-by-side videos for the report page (PyAV; no ffmpeg on Vision or the Mac): every
lora_ge/videos/*.mp4 -> videos/web/<name>.mp4, 1280 px wide, H.264 crf 30.
  envs/armA/bin/python webify.py"""
from pathlib import Path

import av
import cv2
import numpy as np

V = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/outputs/R2S2V_mug/lora_ge/videos"); (V / "web").mkdir(exist_ok=True)
for p in sorted(V.glob("*.mp4")):
    c = av.open(str(p)); fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]; c.close()
    h, w = fr[0].shape[:2]; W = 1280; H = int(round(h * W / w / 2)) * 2
    o = av.open(str(V / "web" / p.name), "w"); st = o.add_stream("h264", rate=16); st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": "30", "preset": "slow", "movflags": "+faststart"}
    for f in fr:
        for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)), format="rgb24")):
            o.mux(pk)
    for pk in st.encode():
        o.mux(pk)
    o.close(); print(p.name, w, "x", h, "->", W, "x", H, (V / "web" / p.name).stat().st_size // 1024, "KB", flush=True)
