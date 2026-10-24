"""Tactile panel (user 2026-10-22: the world model should generate the video AND the glove's tactile matrix). The 880
taxels of the left TouchScale glove are drawn as a 256 x 480 grey panel beside the 832 x 480 video frame, so a video
model can generate both on one 1088 x 480 canvas and the tactile matrix is read back from the pixels.
Layout (8 px per taxel, pads in glove order, distal at the top):
  rows 0-17   index / middle / ring / little: distal 9 x 8, middle 3 x 8, proximal 4 x 8 (little has no proximal)
  rows 19-36  thumb distal 16 x 10 | thumb middle 4 x 8 over thumb base 13 x 8 | palm (hypothenar) 13 x 8
  rows 40-57  six bars, total force per finger (thumb, index, middle, ring, little, palm), 0-BAR_MAX N over 32 blocks
Encoding: background 0; a pad pixel = PAD_LO + (255 - PAD_LO) * sqrt(F / F_MAX) (F clipped to F_MAX), so zero force on a
pad is dim grey and the pad shapes are always visible; decoding reads the centre 4 x 4 px of each block.
  .venv/bin/python r2s2v/tactile_panel.py   -> round-trip test through H.264 + layout sheet"""
import numpy as np

BLOCK, PW, PH = 8, 256, 480
F_MAX, PAD_LO, BAR_MAX = 3.0, 24, 30.0
# glove pad order = cliprep.core.glove_geometry("left") (= the 880 columns of touch()["taxel"]): (id, name, rows, cols)
PADS = [(0, "thumb_distal", 16, 10), (1, "thumb_middle", 4, 8), (2, "thumb_base_thenar", 13, 8), (3, "index_distal", 9, 8),
        (4, "index_middle", 3, 8), (5, "index_proximal", 4, 8), (7, "middle_distal", 9, 8), (8, "middle_middle", 3, 8),
        (9, "middle_proximal", 4, 8), (11, "ring_distal", 9, 8), (12, "ring_middle", 3, 8), (13, "ring_proximal", 4, 8),
        (15, "little_distal", 9, 8), (16, "little_middle", 3, 8), (18, "palm_hypothenar", 13, 8)]
FINGER = {"thumb": (0, 1, 2), "index": (3, 4, 5), "middle": (7, 8, 9), "ring": (11, 12, 13), "little": (15, 16), "palm": (18,)}
_ORIGIN = {3: (0, 0), 4: (10, 0), 5: (14, 0), 7: (0, 8), 8: (10, 8), 9: (14, 8), 11: (0, 16), 12: (10, 16), 13: (14, 16),
           15: (0, 24), 16: (10, 24), 0: (19, 0), 1: (19, 11), 2: (24, 11), 18: (19, 20)}     # (block row, block col) of each pad's top-left
BAR_ROW0, BAR_H, BAR_GAP = 40, 2, 1


def _slices():
    out, off = {}, 0
    for pid, _, r, c in PADS:
        out[pid] = (off, off + r * c, r, c); off += r * c
    assert off == 880
    return out


SLICES = _slices()


def encode(F):
    """(880,) N -> (PH, PW) uint8 panel"""
    F = np.clip(np.nan_to_num(np.asarray(F, np.float32)), 0, None); p = np.zeros((PH, PW), np.uint8)
    for pid, (a, b, r, c) in SLICES.items():
        v = (PAD_LO + (255 - PAD_LO) * np.sqrt(np.minimum(F[a:b], F_MAX) / F_MAX)).reshape(r, c)
        r0, c0 = _ORIGIN[pid]; p[r0 * BLOCK:(r0 + r) * BLOCK, c0 * BLOCK:(c0 + c) * BLOCK] = np.kron(v, np.ones((BLOCK, BLOCK))).round().astype(np.uint8)
    for k, (f, pids) in enumerate(FINGER.items()):
        tot = sum(F[SLICES[q][0]:SLICES[q][1]].sum() for q in pids); n = int(round(32 * min(tot, BAR_MAX) / BAR_MAX))
        y = (BAR_ROW0 + k * (BAR_H + BAR_GAP)) * BLOCK; p[y:y + BAR_H * BLOCK, :n * BLOCK] = 255; p[y:y + BAR_H * BLOCK, n * BLOCK:] = PAD_LO // 2
    return p


def decode(panel):
    """(PH, PW) or (PH, PW, 3) uint8 -> (880,) N from the taxel blocks, and (6,) finger totals from the bars"""
    g = panel.astype(np.float32).mean(-1) if panel.ndim == 3 else panel.astype(np.float32); F = np.zeros(880, np.float32)
    for pid, (a, b, r, c) in SLICES.items():
        r0, c0 = _ORIGIN[pid]; blk = g[r0 * BLOCK:(r0 + r) * BLOCK, c0 * BLOCK:(c0 + c) * BLOCK].reshape(r, BLOCK, c, BLOCK)[:, 2:6, :, 2:6].mean((1, 3))
        F[a:b] = (F_MAX * (np.clip((blk - PAD_LO) / (255 - PAD_LO), 0, 1)) ** 2).ravel()
    bars = []
    for k in range(len(FINGER)):
        y = (BAR_ROW0 + k * (BAR_H + BAR_GAP)) * BLOCK + BLOCK; row = g[y, :].reshape(32, BLOCK)[:, 2:6].mean(1)
        bars.append(BAR_MAX * float((row > 128).sum()) / 32)
    return F, np.array(bars, np.float32)


def finger_totals(F):
    return np.array([sum(F[SLICES[q][0]:SLICES[q][1]].sum() for q in pids) for pids in FINGER.values()], np.float32)


def canvas(frame_rgb, F):
    """832 x 480 RGB frame + panel -> 1088 x 480 RGB"""
    p = encode(F); return np.concatenate([frame_rgb, np.repeat(p[..., None], 3, -1)], 1)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import av
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cliprep.core import Clip, touch
    HERE = Path(__file__).resolve().parents[1]; c = Clip(HERE / "configs/clips/edf79415_11-16_mano.yaml"); ts = c.t0 + np.arange(81) / 16
    G = touch(c, ts)["taxel"]; out = HERE / "runs/clips/r2s2v_page/tactile_roundtrip.mp4"
    o = av.open(str(out), "w"); st = o.add_stream("h264", rate=16); st.width, st.height, st.pix_fmt = PW, PH, "yuv420p"; st.options = {"crf": "18"}
    for k in range(81):
        for pk in st.encode(av.VideoFrame.from_ndarray(np.repeat(encode(G[k])[..., None], 3, -1), format="rgb24")):
            o.mux(pk)
    for pk in st.encode():
        o.mux(pk)
    o.close(); ci = av.open(str(out)); D = np.stack([decode(f.to_ndarray(format="rgb24"))[0] for f in ci.decode(video=0)]); ci.close()
    Gc = np.minimum(G, F_MAX); err = np.abs(D - Gc); on = Gc > 0.1
    print(f"round trip (H.264 crf 18): taxel MAE {err.mean():.4f} N, on contact {err[on].mean():.4f} N, max {err.max():.3f} N; "
          f"finger totals corr {np.corrcoef(finger_totals(D.T).ravel() if False else np.array([finger_totals(x) for x in D]).ravel(), np.array([finger_totals(x) for x in Gc]).ravel())[0, 1]:.4f}")
    k = int(np.argmax(G.sum(1))); sheet = np.concatenate([encode(G[k]), np.full((PH, 8), 128, np.uint8), encode(D[k])], 1)
    cv2.imwrite(str(HERE / "runs/clips/r2s2v_page/tactile_panel_example.png"), sheet); print("example frame", k, "total", round(float(G[k].sum()), 1), "N")
