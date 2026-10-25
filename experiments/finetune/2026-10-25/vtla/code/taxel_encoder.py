"""Glove touch -> tactile tokens for N0-VTLA's predictor. Replaces the repo's frozen DINOv2 encoder of optical tactile IMAGES:
our glove gives one normal force per taxel (880 values), so the touch stays a value array (user 2026-10-24: never an image).

Input  (B, 880)  the glove's normal force above each taxel's baseline (N) divided by the dataset's tactile scale
                 (norm_stats.json: tactile_scale_N). The caller (N0VTLAPolicy._build_g) passes tac_t - tac_0; our touch is
                 already baseline-subtracted, so the data loader sends tac_0 = 0.
Output (B, 1 + P, llm_dim)  one whole-hand token + one token per glove pad (P = 15, assets/glove_taxels.json), the analogue
                 of the repo's CLS + 3x3 pooled tokens per tactile view. A pad token = a per-pad MLP over that pad's taxels
                 (zero-padded to the largest pad) + a learned pad embedding; the whole-hand token = an MLP over all taxels
                 and the per-pad totals. Trainable (the repo trains only the projection after its frozen backbone; we have no
                 pretrained tactile backbone for this glove).
GloveTargetFeatures is the FIXED f_enc of the paper's predictor target z* = f_enc(tac_{t+H} - tac_t): a random Gaussian
projection (no bias, frozen) of [taxel change, per-pad change totals], so cosine similarity between targets follows the
geometry of the touch change itself (the paper's f_enc is the frozen DINOv2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def load_taxel_pads(path: str | Path | None = None) -> tuple[np.ndarray, list[str]]:
    """(880,) pad index per taxel (0..P-1) and the pad names, from assets/glove_taxels.json"""
    d = json.loads(Path(path or ASSETS / "glove_taxels.json").read_text())
    return np.asarray(d["taxel_pad_index"], dtype=np.int64), list(d["pad_names"])


def _pad_index(taxel_pad: np.ndarray) -> tuple[torch.Tensor, int]:
    P = int(taxel_pad.max()) + 1; L = max(int((taxel_pad == p).sum()) for p in range(P)); idx = -np.ones((P, L), np.int64)
    for p in range(P):
        m = np.flatnonzero(taxel_pad == p); idx[p, :len(m)] = m
    return torch.from_numpy(idx), P


class TaxelTactileEncoder(nn.Module):
    """(B, N_taxel) -> (B, 1 + P, llm_dim). `pool_grid` is accepted for signature compatibility with the repo's encoder."""

    def __init__(self, llm_dim: int, pool_grid: int = 3, taxel_pad: np.ndarray | None = None, hidden: int = 256):
        super().__init__()
        del pool_grid
        if taxel_pad is None:
            taxel_pad, _ = load_taxel_pads()
        idx, P = _pad_index(np.asarray(taxel_pad))
        self.register_buffer("pad_idx", idx, persistent=False)
        self.n_taxel, self.n_pads, self.out_dim = int(len(taxel_pad)), P, llm_dim
        L = idx.shape[1]
        self.pad_w = nn.Parameter(torch.randn(P, L, hidden) / L ** 0.5)
        self.pad_b = nn.Parameter(torch.zeros(P, hidden))
        self.pad_out = nn.Linear(hidden, llm_dim)
        self.pad_emb = nn.Parameter(torch.randn(P, llm_dim) * 0.02)
        self.hand = nn.Sequential(nn.Linear(self.n_taxel + P, hidden), nn.GELU(), nn.Linear(hidden, llm_dim))
        self.hand_emb = nn.Parameter(torch.randn(1, llm_dim) * 0.02)

    @property
    def n_tokens(self) -> int:
        return 1 + self.n_pads

    def per_pad(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N_taxel) -> (B, P, L) taxel values grouped by pad (0 in the padding slots)"""
        g = x[:, self.pad_idx.clamp(min=0)]
        return g * (self.pad_idx >= 0).to(g.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.pad_w.dtype).reshape(x.shape[0], -1)
        xp = self.per_pad(x)
        h = F.gelu(torch.einsum("bpl,plh->bph", xp, self.pad_w) + self.pad_b)
        pads = self.pad_out(h) + self.pad_emb
        hand = self.hand(torch.cat([x, xp.sum(-1)], -1))[:, None] + self.hand_emb
        return torch.cat([hand, pads], 1)


class GloveTargetFeatures(nn.Module):
    """Frozen f_enc for the predictor target: (B, N_taxel) touch change -> (B, dim) and the per-pad change (B, P)."""

    def __init__(self, dim: int, taxel_pad: np.ndarray | None = None, seed: int = 0):
        super().__init__()
        if taxel_pad is None:
            taxel_pad, _ = load_taxel_pads()
        idx, P = _pad_index(np.asarray(taxel_pad))
        self.register_buffer("pad_idx", idx, persistent=False)
        gen = torch.Generator().manual_seed(seed); n_in = len(taxel_pad) + P
        self.register_buffer("proj", torch.randn(dim, n_in, generator=gen) / n_in ** 0.5, persistent=False)
        self.n_pads = P

    def pad_totals(self, d: torch.Tensor) -> torch.Tensor:
        g = d[:, self.pad_idx.clamp(min=0)] * (self.pad_idx >= 0).to(d.dtype)
        return g.sum(-1)

    @torch.no_grad()
    def forward(self, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = d.float(); pt = self.pad_totals(d); phi = torch.cat([d, pt], -1)
        return phi @ self.proj.T, pt
