"""Glove touch -> tactile tokens for N0-VTLA's predictor. Replaces the repo's frozen DINOv2 encoder of optical tactile IMAGES:
our glove gives one normal force per taxel (880 values), so the touch stays a value array (user 2026-10-24: never an image).

Input  (B, 880)  the glove's normal force above each taxel's baseline (N) divided by the dataset's tactile scale
                 (norm_stats.json: tactile_scale_N). The caller (N0VTLAPolicy._build_g) passes tac_t - tac_0; our touch is
                 already baseline-subtracted, so the data loader sends tac_0 = 0.
Output (B, 1 + P, llm_dim)  one whole-hand token + one token per glove pad (P = 15, assets/glove_taxels.json), the analogue
                 of the repo's CLS + 3x3 pooled tokens per tactile view. Two encoders (ENCODERS):
  grid (default)  TaxelGridEncoder: shared force embedding + Fourier (row, col) position inside the pad + pad embedding,
                  attention pooling per pad + explicit centroid / spread / area summaries (spatial structure, no image)
  mlp             TaxelTactileEncoder (the first version, kept for the ablation): a pad token = a per-pad MLP over that pad's taxels
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


def load_taxel_grid(path: str | Path | None = None) -> np.ndarray:
    """(880, 2) row / column of each taxel inside its pad, scaled to [-1, 1] per pad (every pad is a full rows x cols grid:
    distal 9 x 8, thumb distal 16 x 10, thenar / hypothenar 13 x 8, the rest 3 x 8 / 4 x 8; names tx_p<pad>_r<row>_c<col>)"""
    import re
    d = json.loads(Path(path or ASSETS / "glove_taxels.json").read_text())
    rc = np.array([[int(x) for x in re.match(r"tx_p\d+_r(\d+)_c(\d+)", n).groups()] for n in d["taxel_names"]], np.float64)
    pad = np.asarray(d["taxel_pad_index"]); out = np.zeros_like(rc)
    for p in np.unique(pad):
        m = pad == p; ext = np.maximum(rc[m].max(0), 1.0); out[m] = rc[m] / ext * 2.0 - 1.0
    return out.astype(np.float32)


class TaxelGridEncoder(nn.Module):
    """Spatial glove encoder (the default since 2026-10-25, user: "不用视觉会丢掉一些 spatial 的信息"): ViT-style tokens on the
    glove's own force grids, not an image. Every taxel gets the SAME value embedding of its force plus a Fourier encoding of its
    (row, col) inside its pad and the pad's embedding, so a contact that moves by one row looks like a small change, not like
    different inputs. Each pad pools its taxels with a learned query (attention pooling) and adds explicit "where" summaries:
    total force, contact area, force-weighted centroid (row, col) and spread. The whole-hand token pools the pad tokens.
    (B, N_taxel) -> (B, 1 + P, llm_dim), the same layout as TaxelTactileEncoder."""

    def __init__(self, llm_dim: int, pool_grid: int = 3, taxel_pad: np.ndarray | None = None, taxel_rc: np.ndarray | None = None,
                 d: int = 256, n_freq: int = 4, contact: float = 0.02):
        super().__init__()
        del pool_grid
        if taxel_pad is None:
            taxel_pad, _ = load_taxel_pads()
        if taxel_rc is None:
            taxel_rc = load_taxel_grid()
        idx, P = _pad_index(np.asarray(taxel_pad)); self.register_buffer("pad_idx", idx, persistent=False)
        rc = torch.as_tensor(taxel_rc, dtype=torch.float32); self.register_buffer("rc", rc, persistent=False)
        f = 2.0 ** torch.arange(n_freq, dtype=torch.float32) * torch.pi; ang = rc[:, :, None] * f              # (N, 2, n_freq)
        self.register_buffer("pe", torch.cat([ang.sin(), ang.cos()], -1).reshape(len(rc), -1), persistent=False)
        self.register_buffer("tax_pad", torch.as_tensor(np.asarray(taxel_pad), dtype=torch.long), persistent=False)
        self.n_taxel, self.n_pads, self.out_dim, self.d, self.contact = int(len(taxel_pad)), P, llm_dim, d, contact
        self.val = nn.Linear(2, d); self.pos = nn.Linear(self.pe.shape[1], d, bias=False); self.pad_in = nn.Parameter(torch.randn(P, d) * 0.02)
        self.tax = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * d))
        self.q = nn.Parameter(torch.randn(P, d) * 0.02); self.summ = nn.Linear(7, d)
        self.pad_norm = nn.LayerNorm(d); self.pad_out = nn.Linear(d, llm_dim); self.pad_emb = nn.Parameter(torch.randn(P, llm_dim) * 0.02)
        self.hq = nn.Parameter(torch.randn(d) * 0.02); self.hand_summ = nn.Linear(7 * P, d); self.hand_norm = nn.LayerNorm(d); self.hand_out = nn.Linear(d, llm_dim)
        self.hand_emb = nn.Parameter(torch.randn(1, llm_dim) * 0.02)

    @property
    def n_tokens(self) -> int:
        return 1 + self.n_pads

    def summaries(self, x: torch.Tensor) -> torch.Tensor:
        """(B, P, 7): total force, contact fraction, centroid row / col, spread row / col, peak force"""
        valid = (self.pad_idx >= 0).to(x.dtype); g = x[:, self.pad_idx.clamp(min=0)] * valid; rc = self.rc[self.pad_idx.clamp(min=0)]   # (B,P,L), (P,L,2)
        tot = g.sum(-1); w = g / tot.clamp(min=1e-6)[..., None]; cen = torch.einsum("bpl,plk->bpk", w, rc)
        spr = torch.einsum("bpl,bplk->bpk", w, (rc[None] - cen[:, :, None]) ** 2).clamp(min=0).sqrt()
        area = ((g > self.contact).to(x.dtype) * valid).sum(-1) / valid.sum(-1); on = (tot > 1e-6).to(x.dtype)[..., None]
        return torch.cat([torch.log1p(tot)[..., None], area[..., None], cen * on, spr * on, torch.log1p(g.amax(-1))[..., None]], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.val.weight.dtype).reshape(x.shape[0], -1).clamp(min=0); B = x.shape[0]
        h = self.val(torch.stack([x, torch.log1p(x)], -1)) + self.pos(self.pe)[None] + self.pad_in[self.tax_pad][None]   # (B, N, d)
        k, v = self.tax(h).chunk(2, -1)
        idx = self.pad_idx.clamp(min=0); mask = self.pad_idx >= 0
        kp, vp = k[:, idx], v[:, idx]                                                                                    # (B, P, L, d)
        lg = torch.einsum("bpld,pd->bpl", kp, self.q) / self.d ** 0.5; lg = lg.masked_fill(~mask[None], float("-inf"))
        pooled = torch.einsum("bpl,bpld->bpd", lg.softmax(-1), vp)
        s = self.summaries(x); pt = self.pad_norm(pooled + self.summ(s))                                                   # (B, P, d)
        a = torch.einsum("bpd,d->bp", pt, self.hq).softmax(-1)
        hand = self.hand_norm(torch.einsum("bp,bpd->bd", a, pt) + self.hand_summ(s.reshape(B, -1)))
        return torch.cat([self.hand_out(hand)[:, None] + self.hand_emb, self.pad_out(pt) + self.pad_emb], 1)


ENCODERS = {"grid": TaxelGridEncoder, "mlp": TaxelTactileEncoder}


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
