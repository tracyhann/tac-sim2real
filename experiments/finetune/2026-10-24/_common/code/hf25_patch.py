"""Touch head for the HF25 video + touch LoRAs (train_control_lora_hf25.py --tactile_head; user 2026-10-24: the touch is a
vector of values, never pixels). Design = UVA / UVTA: the video DiT is the shared backbone (LoRA, as the F / H runs); a light
head decodes the 880 glove taxels of every frame from the DiT's last-layer tokens, so the touch comes out of the same forward
pass as the generated video, and the touch loss trains the LoRA too.
  TactileHead     tokens of the video (f latent frames x hw) -> per latent frame one attention pooling over its hw tokens
                  (+ the known frame-0 touch) -> temporal conv -> 4 x 880 per latent frame (frame 0: 1 x 880) -> (81, 880)
                  in u = min(F / S, CAP) / CAP (sigmoid), the loss-table target. ~13 M params, fp32.
  attach_capture  forward pre-hook on the DiT's output Head: the tokens entering it (= the last block's output) are kept; with
                  VideoX-Fun's non-reentrant checkpointing they carry the graph, so the touch loss backpropagates.
  tactile_loss    R1 of the loss table (Huber + 1/2 L1, active taxels x10; TouchSight), the best contact F1 there.
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
        self.t0 = nn.Linear(N_TAXEL, 256); self.mix = nn.Linear(d + 256, d); self.tconv = nn.Conv1d(d, d, 3, padding=1)
        self.drop = nn.Dropout(p); self.out = nn.Linear(d, 4 * N_TAXEL); self.d = d

    def forward(self, x, hw, touch0):
        """x (b, f * hw, dim) video tokens in (f, h, w) order; touch0 (b, 880) frame-0 u -> (b, 4 (f - 1) + 1, 880) u"""
        b, n, _ = x.shape; f = n // hw; assert f * hw == n, (n, hw); dt = self.proj.weight.dtype   # computed in the head's own dtype (fp32 in training, see make_trainer)
        h = self.proj(self.norm(x.to(dt))).view(b, f, hw, self.d)
        a = torch.softmax((h @ self.q) / self.d ** 0.5, -1)                        # (b, f, hw): where this frame's touch is read from
        g = torch.cat([(a.unsqueeze(-1) * h).sum(2), torch.relu(self.t0(touch0.to(dt)))[:, None].expand(-1, f, -1)], -1)
        g = torch.relu(self.mix(g)); g = g + torch.relu(self.tconv(g.transpose(1, 2))).transpose(1, 2)
        y = torch.sigmoid(self.out(self.drop(g))).view(b, f, 4, N_TAXEL).float()
        return torch.cat([y[:, :1, 0], y[:, 1:].reshape(b, -1, N_TAXEL)], 1)       # latent frame 0 = frame 0; j -> frames 4j-3 .. 4j


def video_tokens(x, hw, f, ref):
    """the video's f * hw tokens out of the DiT sequence: [ref image tokens (hw, when full_ref is used)] + video [+ padding]"""
    off = hw if ref else 0; n = f * hw
    assert x.size(1) >= off + n, (tuple(x.shape), hw, f, ref)
    return x[:, off:off + n]


def attach_capture(transformer):
    """keep the tokens entering the DiT's output Head (list with the latest capture); the transformer is used unchanged"""
    from videox_fun.models.wan_transformer3d import Head as DiTHead
    heads = [m for m in transformer.modules() if isinstance(m, DiTHead)]; assert len(heads) == 1, len(heads)
    cap = []

    def pre(module, args):
        cap.clear(); cap.append(args[0])
    heads[0].register_forward_pre_hook(pre)
    return cap


def tactile_loss(pred, u):
    """R1: Huber(beta 1) + 1/2 L1, active taxels (u > THR_U) weighted x10; pred, u (b, 81, 880) in u units"""
    w = 1 + 9 * (u > THR_U).float()
    return (w * (F.smooth_l1_loss(pred, u, beta=1.0, reduction="none") + 0.5 * (pred - u).abs())).sum() / w.sum()


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
    head = TactileHead(dim); p = head(video_tokens(x, hw, f, True), hw, u[:, 0]); l = tactile_loss(p, u); l.backward()
    print("head params", sum(q.numel() for q in head.parameters()) / 1e6, "M; pred", tuple(p.shape), "loss", l.item(), "grad on tokens", float(x.grad.abs().sum()) > 0,
          "ref tokens untouched", float(x.grad[:, :hw].abs().sum()) == 0)
    assert p.shape == (b, 81, 880) and torch.isfinite(l) and x.grad[:, hw:].abs().sum() > 0 and x.grad[:, :hw].abs().sum() == 0
    assert np.allclose(rel_from_u(u_from_rel(np.array([0.0, 0.05, 0.11, 0.3]))), [0.0, 0.05, 0.11, 0.11])
    print("ok")
