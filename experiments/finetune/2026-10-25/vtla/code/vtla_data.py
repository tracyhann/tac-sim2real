"""Glove dataset -> N0-VTLA training samples (one sample = one video frame t of one row).

Inputs per sample (the repo's pi0.5 layout):
  base_0_rgb        the row's video frame t (head camera, 832 x 480 -> 224 x 224 with aspect-preserving padding), [-1, 1]
  left/right wrist  zero images with image_mask False (no wrist camera in this data)
  prompt            task instruction + the observable perturbation clause (prompts.py v2). Mass and friction are NOT stated by
                    default (--prompt observable): they are hidden physics the touch has to reveal, and stating them would leak
                    the answer into the language (--prompt full keeps the dataset's full text)
  state             32-D (vtla/<id>.npz state), quantile-normalised to [-1, 1] and written into the prompt as 256-bin tokens
                    (pi0.5 discrete state input)
  glove_left        (880,) touch at t (N above baseline) / tactile_scale_N; glove_left.baseline = 0 (already subtracted)
Targets:
  actions           (H, 32) chunk act[t+1 .. t+H] of the arm (act_vts | act_vtf), the wrist dims 0:9 and the joints 10:32 as deltas
                    to state[t] (the repo's DeltaActions mask make_bool_mask(9, -1, ...) extended to the joint slots), dim 9
                    absolute (closure for VTS, grip force for VTF); steps past the end repeat the last frame; quantile-normalised
  tac_delta         (880,) (touch[t+H] - touch[t]) / tactile_scale_N, the predictor target (paper Stage 1)
norm_stats.json: q01 / q99 of state and of the delta actions, tactile_scale_N (99th percentile of active taxels), the arm, H."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent / "2026-10-24" / "_common" / "code"))   # prompts.py v2 (shared with HF25)
from prompts import _DEFAULT, prompt_from_params  # noqa: E402

N, FPS, IMG = 81, 16, 224
ARMS = ("vts", "vtf", "obs", "cmd")                                  # obs / cmd: the touch-data comparison (README section 11)
DELTA_MASK = np.array([True] * 9 + [False] + [True] * 22)
LAYOUT = {"vts": ["wrist_x", "wrist_y", "wrist_z"] + [f"wrist_rot6d_{i}" for i in range(6)] + ["closure_rad"] + [f"q_{i}" for i in range(22)],
          "vtf": ["wrist_x", "wrist_y", "wrist_z"] + [f"wrist_rot6d_{i}" for i in range(6)] + ["grip_force_N"] + [f"q_cmd_{i}" for i in range(22)]}
LAYOUT["obs"] = LAYOUT["vts"]
LAYOUT["cmd"] = ["wrist_x", "wrist_y", "wrist_z"] + [f"wrist_rot6d_{i}" for i in range(6)] + ["closure_rad"] + [f"q_cmd_{i}" for i in range(22)]
TACTILE_KEY = "glove_left"
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def resize_with_pad(img, size=IMG):
    h, w = img.shape[:2]; s = size / max(h, w); nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA); out = np.zeros((size, size, 3), np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2; out[y:y + nh, x:x + nw] = r
    return out


def load_frames(video: Path, cache: Path | None) -> np.ndarray:
    """(81, 224, 224, 3) uint8, cached as .npy"""
    cp = cache / f"{video.stem}_{IMG}.npy" if cache else None
    if cp is not None and cp.exists():
        return np.load(cp)
    with av.open(str(video)) as c:
        fr = np.stack([resize_with_pad(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)])
    assert len(fr) == N, (video, len(fr))
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True); np.save(cp, fr)
    return fr


def policy_prompt(row, mode="observable"):
    """rows with text_priv (prompts v3: batch 2 on) carry both: text = instruction + observable / commanded quantities,
    text_priv = + mass and friction; older rows carry the full text only, so their observable prompt is rebuilt from params"""
    if row.get("text_priv"):
        return row["text_priv"] if mode == "full" else row["text"]
    if mode == "full":
        return row["text"]
    p = dict(_DEFAULT, **{k: v for k, v in (row.get("params") or {}).items() if k in _DEFAULT or k == "shake_axis"})
    p.update(water_g=0, friction_scale=1.0)                       # hidden physics stays out of the instruction
    return prompt_from_params(p)


def load_rows(ds: Path, ids=None):
    rows = json.loads((ds / "metadata_vtla.json").read_text())["rows"]
    rows = [r for r in rows if (ds / "vtla" / f"{r['id']}.npz").exists() and (ids is None or r["id"] in ids)]
    return rows


def split_rows(rows, val_ids=(), val_groups=()):
    va = [r for r in rows if r["id"] in val_ids or r.get("group") in val_groups]; tr = [r for r in rows if r not in va]
    return tr, va


class Norm:
    def __init__(self, d):
        self.d = d; self.sq01, self.sq99 = np.array(d["state_q01"]), np.array(d["state_q99"]); self.aq01, self.aq99 = np.array(d["action_q01"]), np.array(d["action_q99"])
        self.tac = float(d["tactile_scale_N"])

    # range floor (2026-10-26): finger joints that barely move in most rows have q01..q99 ~ 1e-4 rad, so the rare real movement (the
    # finger batch) normalised to |x| ~ 400 and the first comparison run started at loss 92; 0.02 (~1 deg / 2 cm / 0.02 rot6d) is below
    # every dimension's real spread in the pi0.5 sense and keeps the rest within a few units
    FLOOR = 0.02

    @staticmethod
    def _n(x, lo, hi):
        return 2.0 * (x - lo) / np.maximum(hi - lo, Norm.FLOOR) - 1.0

    @staticmethod
    def _u(x, lo, hi):
        return (x + 1.0) / 2.0 * np.maximum(hi - lo, Norm.FLOOR) + lo

    def state(self, s):
        return self._n(s, self.sq01, self.sq99)

    def action(self, a):
        return self._n(a, self.aq01, self.aq99)

    def unaction(self, a):
        return self._u(a, self.aq01, self.aq99)

    def save(self, path):
        Path(path).write_text(json.dumps(self.d, indent=1))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))


def chunk(act, state, t, H):
    idx = np.minimum(np.arange(t + 1, t + 1 + H), N - 1); a = act[idx].astype(np.float64)
    a[:, DELTA_MASK] -= state[t, DELTA_MASK]
    return a, (np.arange(t + 1, t + 1 + H) <= N - 1)


def compute_norm(ds: Path, rows, arm, H):
    S, A, T = [], [], []
    for r in rows:
        z = np.load(ds / "vtla" / f"{r['id']}.npz"); st, act = z["state"], z[f"act_{arm}"]; S.append(st); T.append(z["tactile"].ravel())
        for t in range(N - 1):
            A.append(chunk(act, st, t, H)[0])
    S, A, T = np.concatenate(S), np.concatenate(A), np.concatenate(T)
    act_tx = T[T > 0.05]
    return Norm(dict(arm=arm, horizon=H, layout=LAYOUT[arm], delta_mask=DELTA_MASK.tolist(), state_q01=np.quantile(S, 0.01, 0).tolist(), state_q99=np.quantile(S, 0.99, 0).tolist(),
                     action_q01=np.quantile(A, 0.01, 0).tolist(), action_q99=np.quantile(A, 0.99, 0).tolist(), tactile_scale_N=float(np.quantile(act_tx, 0.99)) if len(act_tx) else 1.0,
                     n_rows=len(rows)))


class GloveChunks(torch.utils.data.Dataset):
    def __init__(self, ds: Path, rows, arm, H, norm: Norm, prompt="observable", cache=None, starts=None, max_len=200):
        assert arm in ARMS
        self.ds, self.rows, self.arm, self.H, self.norm, self.max_len = Path(ds), rows, arm, H, norm, max_len
        self.frames = [load_frames(self.ds / r["video"], Path(cache) if cache else None) for r in rows]   # r["video"] may be absolute (comparison sets)
        self.lab = [dict(np.load(self.ds / "vtla" / f"{r['id']}.npz")) for r in rows]
        self.prompts = [policy_prompt(r, prompt) for r in rows]
        st = starts if starts is not None else range(N - 1)
        self.items = [(i, t) for i in range(len(rows)) for t in st if t < N - 1]
        self._tok = None

    def __len__(self):
        return len(self.items)

    def tokenizer(self):
        if self._tok is None:
            from n0vtla.models.tokenizer import PaligemmaTokenizer
            self._tok = PaligemmaTokenizer(self.max_len)
        return self._tok

    def __getitem__(self, k):
        i, t = self.items[k]; z = self.lab[i]; st = z["state"]
        a, valid = chunk(z[f"act_{self.arm}"], st, t, self.H); s_n = self.norm.state(st[t].astype(np.float64))
        tok, mask = self.tokenizer().tokenize(self.prompts[i], s_n)
        tac = z["tactile"]; tf = tac[min(t + self.H, N - 1)]
        return dict(image=self.frames[i][t], state=s_n.astype(np.float32), tokens=tok.astype(np.int64), token_mask=mask.astype(bool),
                    tactile=(tac[t] / self.norm.tac).astype(np.float32), tac_delta=((tf - tac[t]) / self.norm.tac).astype(np.float32),
                    actions=self.norm.action(a).astype(np.float32), valid=valid, row=i, t=t)


def collate(items):
    out = {k: np.stack([it[k] for it in items]) for k in items[0]}
    return out


@dataclasses.dataclass
class GloveObs:
    """the fields N0VTLAPolicy / preprocess_observation_pytorch read (the repo's Observation without jaxtyping checks)"""
    images: dict
    image_masks: dict
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None
    expert_images: dict | None = None
    expert_image_masks: dict | None = None

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


def to_obs(b, device, tactile=None, drop_touch=False):
    """batch (numpy) -> (GloveObs, actions, tac_delta) on device. tactile overrides the glove input (probes); drop_touch leaves
    the glove out entirely (no-touch arms: the repo's predictor then uses its learned null token instead of g)."""
    B = len(b["state"])
    img = torch.from_numpy(b["image"]).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    zero = torch.zeros_like(img); on = torch.ones(B, dtype=torch.bool, device=device); off = torch.zeros(B, dtype=torch.bool, device=device)
    tac = torch.from_numpy(b["tactile"] if tactile is None else tactile).to(device)
    obs = GloveObs(images={"base_0_rgb": img, "left_wrist_0_rgb": zero, "right_wrist_0_rgb": zero.clone(), TACTILE_KEY: tac, TACTILE_KEY + ".baseline": torch.zeros_like(tac)},
                   image_masks={"base_0_rgb": on, "left_wrist_0_rgb": off, "right_wrist_0_rgb": off.clone(), TACTILE_KEY: on.clone()},
                   state=torch.from_numpy(b["state"]).to(device), tokenized_prompt=torch.from_numpy(b["tokens"]).to(device),
                   tokenized_prompt_mask=torch.from_numpy(b["token_mask"]).to(device))
    if drop_touch:
        for k in (TACTILE_KEY, TACTILE_KEY + ".baseline"):
            obs.images.pop(k); obs.image_masks.pop(k, None)
    return obs, torch.from_numpy(b["actions"]).to(device), torch.from_numpy(b["tac_delta"]).to(device)
