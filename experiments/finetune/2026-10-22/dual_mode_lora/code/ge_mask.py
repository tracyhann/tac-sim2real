"""Structured inpaint masks for the dual-mode video + touch LoRA (train_control_lora_ge.py --ge_mask; user 2026-10-22).
The canvas is video (x < x0) | touch panel (x >= x0). Per sample, one job:
  G (generate)  known = frame 0's video, and with probability p_touch0 also frame 0's touch panel; unknown = every other
                frame (video + panel), and frame 0's panel otherwise. Control = the row's main control (the sim render
                when the row has one, else the clip's own Depth Anything depth).
  E (estimate)  known = the video in all 81 frames; unknown = the touch panel in all frames. Control = the sim control or
                the Depth Anything depth, 50 / 50 when the row has both, else the one it has.
Mask convention (VideoX-Fun get_random_mask): (f, 1, h, w) uint8, 1 = unknown (generated), 0 = known (given through the
inpaint latents). VideoX-Fun's random masks (rectangles, random pixels, first / last frames ...) are never used: they
hand the model parts of the target (often the touch panel)."""
import os
import random

import numpy as np
import torch

try:
    from videox_fun.data.dataset_image_video import ImageVideoControlDataset
except ImportError:                                               # unit tests without VideoX-Fun on the path
    ImageVideoControlDataset = object


def panel_x0(w, args):
    x0 = int(round(w * args.ge_panel_x0 / args.ge_canvas_w))
    assert 0 < x0 < w, (w, x0)
    return x0


def make_mask(shape, ge, args):
    """shape = pixel_values.size() = (f, c, h, w); ge = dict(mode, touch0, ...) chosen by GEDataset"""
    f, _, h, w = shape; x0 = panel_x0(w, args); m = torch.zeros((f, 1, h, w), dtype=torch.uint8)
    if ge["mode"] == "G":
        m[:] = 1; m[0, :, :, :x0] = 0
        if ge["touch0"]:
            m[0, :, :, x0:] = 0
    elif ge["mode"] == "E":
        m[:, :, :, x0:] = 1
    else:
        raise ValueError(ge)
    return m


class GEDataset(ImageVideoControlDataset):
    """ImageVideoControlDataset that picks the job (G / E) and the control per sample; the instance is converted in place
    (train_dataset.__class__ = GEDataset) so the sampler / bucket code keeps working on the same object."""

    def ge_setup(self, args):
        self.ge_modes = args.ge_modes; self.ge_p_gen = args.ge_p_gen; self.ge_p_touch0 = args.ge_p_touch0; self.ge_p_sim = args.ge_p_sim_ctrl
        self._rng = None; self._ge = None
        bad = [d.get("id", d["file_path"]) for d in self.dataset if not set(d.get("modes", "GE")) & set(self.ge_modes)]
        assert not bad, f"rows with no mode allowed in this run ({self.ge_modes}): {bad}"

    def rng(self):
        if self._rng is None:                                     # per worker process, independent of the seeded global random
            self._rng = random.Random(int.from_bytes(os.urandom(8), "little"))
        return self._rng

    def choose(self, row):
        g = self.rng(); allowed = [m for m in self.ge_modes if m in row.get("modes", "GE")]
        mode = allowed[0] if len(allowed) == 1 else ("G" if g.random() < self.ge_p_gen else "E")
        alt = row.get("control_alt_file_path")
        use_alt = mode == "E" and bool(alt) and g.random() >= self.ge_p_sim
        return dict(mode=mode, touch0=bool(mode == "G" and g.random() < self.ge_p_touch0), ctrl="alt" if use_alt else "main",
                    control=alt if use_alt else row["control_file_path"], id=row.get("id", row["file_path"]))

    def get_batch(self, idx):
        i = idx % len(self.dataset); row = self.dataset[i]; ge = self.choose(row)
        self.dataset[i] = dict(row, control_file_path=ge["control"])
        try:
            out = super().get_batch(idx)
        finally:
            self.dataset[i] = row
        self._ge = ge
        return out

    def __getitem__(self, idx):
        s = super().__getitem__(idx); s["ge"] = dict(self._ge)
        return s


def check_mask(m, ge, x0):
    """the invariants the trainer relies on (used by the unit test)"""
    m = m.numpy()[:, 0] if hasattr(m, "numpy") else m[:, 0]
    if ge["mode"] == "G":
        assert (m[1:] == 1).all() and (m[0, :, :x0] == 0).all() and (m[0, :, x0:] == (0 if ge["touch0"] else 1)).all()
    else:
        assert (m[:, :, :x0] == 0).all() and (m[:, :, x0:] == 1).all()
    return True
