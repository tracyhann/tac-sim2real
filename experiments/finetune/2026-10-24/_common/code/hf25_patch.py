"""Touch head for the HF25 video + touch LoRAs (train_control_lora_hf25.py --tactile_head; user 2026-10-24: the touch is a
vector of values, never pixels). Design = UVA / UVTA: the video DiT is the shared backbone (LoRA, as the F / H runs); a light
head decodes the 880 glove taxels of every frame from the DiT's last-layer tokens, so the touch comes out of the same forward
pass as the generated video, and the touch loss trains the LoRA too.
  TactileHead     tokens of the video (f latent frames x hw) -> per latent frame one attention pooling over its hw tokens
                  (+ the known frame-0 touch, + the DiT's timestep embedding so the head knows the noise level it reads at)
                  -> temporal conv -> 4 x 880 per latent frame (frame 0: 1 x 880) -> (81, 880) in u = min(F / S, CAP) / CAP,
                  decoded as sigmoid^2 (the loss-table R1 decode: a near-zero floor for the ~96 % silent sensors). ~18 M params, fp32.
  attach_capture  forward pre-hook on the DiT's output Head: the tokens entering it (= the last block's output) and its timestep
                  embedding are kept; with VideoX-Fun's non-reentrant checkpointing the tokens carry the graph, so the touch loss
                  backpropagates into the LoRA.
  tactile_loss    R1 of the loss table (Huber + 1/2 L1, active taxels x10; TouchSight) + the per-frame TOTAL force L1 (units of S).
  v1 (10-24 morning: plain sigmoid, R1 only, no timestep input; run FHN_low_v1) tracked the force course (r 0.7) but at 2x the
  magnitude on every case incl. training rows and never released: 880 per-taxel errors of ~0.01 sum to ~1 S, which R1 ignores,
  and the sigmoid floor never reaches 0. v2 = this file.
  HF25Dataset     ImageVideoControlDataset + sample["tactile"] = the row's u array (81, 880) (tactile .npz key "u").
Units: CAP, THR_REL = panel v3 (dual_mode_lora/code/panel_v3.py), F / S with S the replay's touch scale."""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from videox_fun.data.dataset_image_video import ImageVideoControlDataset
except ImportError:                                               # unit tests without VideoX-Fun on the path
    ImageVideoControlDataset = object

CAP, THR_REL = 0.11, 0.0059                                       # relative cap / contact threshold, units of S (panel v3)
THR_U = THR_REL / CAP; N_TAXEL = 880


def u_from_rel(T):
    """relative touch F / S (frames, 880) -> head target u in [0, 1]"""
    return (np.minimum(np.clip(np.asarray(T, np.float32), 0, None), CAP) / CAP).astype(np.float32)


def rel_from_u(u):
    return np.clip(np.asarray(u, np.float32), 0, 1) * CAP


class TactileHead(nn.Module):
    def __init__(self, dim, d=1024, p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim); self.proj = nn.Linear(dim, d); self.q = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.t0 = nn.Linear(N_TAXEL, 256); self.mix = nn.Linear(d + 256, d); self.te = nn.Linear(dim, d); self.tconv = nn.Conv1d(d, d, 3, padding=1)
        self.drop = nn.Dropout(p); self.out = nn.Linear(d, 4 * N_TAXEL); self.d = d
        nn.init.constant_(self.out.bias, -3.0)                                     # start near silence: sigmoid(-3)^2 = 0.002 per sensor (0.2 S total), not 0.25

    def forward(self, x, hw, touch0, e=None):
        """x (b, f * hw, dim) video tokens in (f, h, w) order; touch0 (b, 880) frame-0 u; e = the DiT output head's timestep
        embedding (b, ..., dim), optional -> (b, 4 (f - 1) + 1, 880) u"""
        b, n, _ = x.shape; f = n // hw; assert f * hw == n, (n, hw); dt = self.proj.weight.dtype   # computed in the head's own dtype (fp32 in training, see make_trainer)
        h = self.proj(self.norm(x.to(dt))).view(b, f, hw, self.d)
        a = torch.softmax((h @ self.q) / self.d ** 0.5, -1)                        # (b, f, hw): where this frame's touch is read from
        g = torch.cat([(a.unsqueeze(-1) * h).sum(2), torch.relu(self.t0(touch0.to(dt)))[:, None].expand(-1, f, -1)], -1)
        g = torch.relu(self.mix(g))
        if e is not None:                                                          # noise-level calibration: training reads at random t, inference at the last step
            g = g + torch.relu(self.te(e.to(dt).reshape(b, -1, e.shape[-1]).mean(1)))[:, None]
        g = g + torch.relu(self.tconv(g.transpose(1, 2))).transpose(1, 2)
        y = (torch.sigmoid(self.out(self.drop(g))) ** 2).view(b, f, 4, N_TAXEL).float()   # squared: silent sensors can reach ~0
        return torch.cat([y[:, :1, 0], y[:, 1:].reshape(b, -1, N_TAXEL)], 1)       # latent frame 0 = frame 0; j -> frames 4j-3 .. 4j


def video_tokens(x, hw, f, ref):
    """the video's f * hw tokens out of the DiT sequence: [ref image tokens (hw, when full_ref is used)] + video [+ padding]"""
    off = hw if ref else 0; n = f * hw
    assert x.size(1) >= off + n, (tuple(x.shape), hw, f, ref)
    return x[:, off:off + n]


def attach_capture(transformer):
    """keep what enters the DiT's output Head: cap[0] = the tokens, cap[1] = the timestep embedding e (latest call only); the
    transformer is used unchanged"""
    from videox_fun.models.wan_transformer3d import Head as DiTHead
    heads = [m for m in transformer.modules() if isinstance(m, DiTHead)]; assert len(heads) == 1, len(heads)
    cap = []

    def pre(module, args):
        cap.clear(); cap.append(args[0]); cap.append(args[1] if len(args) > 1 else None)
    heads[0].register_forward_pre_hook(pre)
    return cap


def tactile_loss(pred, u, w_total=1.0):
    """R1 (Huber(beta 1) + 1/2 L1, active taxels (u > THR_U) weighted x10) + w_total * per-frame total-force L1 in units of S;
    pred, u (b, 81, 880) in u units. The total term supervises what the evaluation measures (sum over 880 sensors), which the
    per-taxel R1 alone does not see (v1 lesson)."""
    w = 1 + 9 * (u > THR_U).float()
    r1 = (w * (F.smooth_l1_loss(pred, u, beta=1.0, reduction="none") + 0.5 * (pred - u).abs())).sum() / w.sum()
    tot = ((pred - u).sum(-1) * CAP).abs().mean()
    return r1 + w_total * tot


class HF25Dataset(ImageVideoControlDataset):
    """ImageVideoControlDataset whose samples also carry the row's touch target (the instance is converted in place:
    train_dataset.__class__ = HF25Dataset, so the sampler / bucket code keeps working on the same object)."""

    def __getitem__(self, idx):
        sample = super().__getitem__(idx); row = self.dataset[sample["idx"] % len(self.dataset)]
        u = np.load(os.path.join(self.data_root, row["tactile"]))["u"].astype(np.float32)
        assert u.shape == (sample["pixel_values"].shape[0], N_TAXEL), (row.get("id"), u.shape, tuple(sample["pixel_values"].shape))
        sample["tactile"] = u
        return sample


if __name__ == "__main__":                                        # CPU unit test (small dims): shapes, gradient reaches the video tokens only, loss finite
    torch.manual_seed(0); b, f, hw, dim = 1, 21, 156, 512
    x = torch.randn(b, hw + f * hw, dim, requires_grad=True); u = torch.rand(b, 81, 880) * (torch.rand(b, 81, 880) > 0.9)
    head = TactileHead(dim); e = torch.randn(b, dim); p = head(video_tokens(x, hw, f, True), hw, u[:, 0], e); l = tactile_loss(p, u); l.backward()
    p0 = head(video_tokens(x, hw, f, True), hw, u[:, 0], None); assert p0.shape == p.shape      # e optional
    print("head params", sum(q.numel() for q in head.parameters()) / 1e6, "M; pred", tuple(p.shape), "loss", l.item(), "grad on tokens", float(x.grad.abs().sum()) > 0,
          "ref tokens untouched", float(x.grad[:, :hw].abs().sum()) == 0)
    assert p.shape == (b, 81, 880) and torch.isfinite(l) and x.grad[:, hw:].abs().sum() > 0 and x.grad[:, :hw].abs().sum() == 0
    assert np.allclose(rel_from_u(u_from_rel(np.array([0.0, 0.05, 0.11, 0.3]))), [0.0, 0.05, 0.11, 0.11])
    print("ok")
