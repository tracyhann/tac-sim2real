"""N0-VTLA on the glove data: the repo's policy (third_party/N0-VTLA, commit 03a0ce4, imported, not copied) with ONE swap and
the paper's predictor objective added.

  model      N0VTLAPolicy = PaliGemma 3B prefix (head-camera frame + language with the discretised state, pi0.5 style) +
             Gemma 300M flow-matching action expert; the tactile path is the repo's: tactile tokens g -> predictor
             (predictor_arch "tactile_kv": g is the only key/value, the VL prefix only conditions the queries) -> n_latent = 5
             latent tokens z -> z_proj -> zero-initialised gate -> prepended to the action suffix (actions attend z and the
             prefix; the prefix never sees z; z positions are sliced off before the action head).
  swap       tactile_encoder = TaxelTactileEncoder (glove values -> 16 tokens) instead of the frozen DINOv2 over optical
             tactile images. Everything else, incl. the pretrained predictor / z_proj / z_gate weights, loads from the
             N0-VTLA base checkpoint (NeoteAI/n0-vtla-base).
  added      aux_loss = the paper's Stage-1 predictor grounding, which the repo does not ship: symmetric InfoNCE between the
             pooled z and z* = f_enc(tac_{t+H} - tac_t) over the batch (f_enc = GloveTargetFeatures, frozen) + an L1
             reconstruction of the coarse (per-pad) future touch change from z (recon_head).
  stages     set_stage(model, "pred" | "align" | "joint"): the paper's three stages. pred = only the tactile stack
             (encoder, predictor, recon head) on aux_loss; align = z_proj, gate and the action expert on the action loss
             with the VL prefix masked for the action queries (actions see only z); joint = everything on the action loss
             (+ optional aux), the tactile stack at lr x 0.1 (the repo's VTLA_PREDICTOR_LR_SCALE)."""
from __future__ import annotations

import collections
import contextlib
import functools
import logging

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import n0vtla.models_pytorch.gemma_pytorch as _gp
import n0vtla.models_pytorch.pi0_pytorch as _pp
import n0vtla.models_pytorch.tactile_encoder as _te
from n0vtla.models_pytorch.n0vtla_policy import N0VTLAConfig, N0VTLAPolicy
from taxel_encoder import GloveTargetFeatures, TaxelTactileEncoder, load_taxel_pads

TACTILE_KEY = "glove_left"
TACT_PREFIXES = ("tactile_encoder.", "tactile_predictor.", "recon_head.")
ALIGN_PREFIXES = ("z_proj.", "z_gate", "paligemma_with_expert.gemma_expert.", "action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.")
LOW_LR_PREFIXES = TACT_PREFIXES + ("z_proj.",)


def make_config(horizon: int, dummy: bool = False, dtype: str = "bfloat16") -> N0VTLAConfig:
    """the repo's vtla_tactile_posttrain model config, with our horizon and one tactile view (the left glove)"""
    kw = dict(pi05=True, action_dim=32, action_horizon=horizon, pytorch_compile_mode=None, tactile_predictor_enabled=True, tactile_mode="latent",
              n_latent=5, predictor_arch="tactile_kv", z_gate_zero_init=True, vl_dropout_prob=0.0, predictor_loss_weight=0.0,
              tactile_image_keys=(TACTILE_KEY,), dtype=dtype)
    if dummy:                                                     # tiny Gemma widths for CPU tests (no checkpoint fits it)
        kw.update(paligemma_variant="dummy", action_expert_variant="dummy")
    return N0VTLAConfig(**kw)


class _TinyPaliGemmaWithExpert(_gp.PaliGemmaWithExpertModel):
    """tests only (dummy widths): the repo builds PaliGemma's projector and top-level hidden_size for gemma_2b (2048)"""

    def __init__(self, vlm_config, *args, **kw):
        super().__init__(vlm_config, *args, **kw)
        pg = self.paligemma; pg.config.hidden_size = vlm_config.width
        for name, mod in pg.named_modules():
            if name.endswith("multi_modal_projector") and hasattr(mod, "linear"):
                mod.linear = nn.Linear(mod.linear.in_features, vlm_config.width).to(mod.linear.weight.dtype)


class GloveVTLA(N0VTLAPolicy):
    def __init__(self, config, taxel_pad=None):
        if taxel_pad is None:
            taxel_pad, _ = load_taxel_pads()
        orig, orig_pg = _te.FrozenDINOv2TactileEncoder, _pp.PaliGemmaWithExpertModel   # names looked up at call time
        _te.FrozenDINOv2TactileEncoder = functools.partial(TaxelTactileEncoder, taxel_pad=taxel_pad)
        if config.paligemma_variant == "dummy":
            _pp.PaliGemmaWithExpertModel = _TinyPaliGemmaWithExpert
        try:
            super().__init__(config)
        finally:
            _te.FrozenDINOv2TactileEncoder, _pp.PaliGemmaWithExpertModel = orig, orig_pg
        D = self.paligemma_with_expert.paligemma.config.hidden_size
        self.target_features = GloveTargetFeatures(D, taxel_pad=taxel_pad)
        self.recon_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, self.target_features.n_pads))

    def aux_loss(self, observation, tac_delta, tau: float = 0.1, w_rec: float = 1.0, min_change: float = 1e-3):
        """Stage-1 predictor objective. tac_delta (B, N_taxel): (tac_{t+H} - tac_t) / tactile scale.
        Returns (loss, parts); rows without a touch change are left out of the InfoNCE (their target has no direction)."""
        images, img_masks, lang_tokens, lang_masks, _state, _ = self._preprocess_observation(observation, train=self.training)
        vlm_trains = any(p.requires_grad for p in self.paligemma_with_expert.paligemma.parameters())
        with contextlib.nullcontext() if vlm_trains else torch.no_grad():
            vl_ctx, _pe, prefix_pad_masks, _pa, _kv = self._prefix_forward(images, img_masks, lang_tokens, lang_masks, use_cache=False)
        z, _g, _has = self._compute_z(vl_ctx.detach(), prefix_pad_masks)
        zp = z.float().mean(1)
        tgt, pad_d = self.target_features(tac_delta.to(zp.device))
        rec = (self.recon_head(zp) - pad_d).abs().mean()
        valid = tgt.norm(dim=-1) > min_change
        parts = dict(rec=float(rec.detach()), n_change=int(valid.sum()))
        loss = w_rec * rec
        if int(valid.sum()) >= 2:
            p = F.normalize(zp[valid], dim=-1); t = F.normalize(tgt[valid], dim=-1); logits = p @ t.T / tau
            lab = torch.arange(len(p), device=p.device)
            nce = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            loss = loss + nce
            parts.update(nce=float(nce.detach()), top1=float((logits.argmax(1) == lab).float().mean()))
        return loss, parts

    @torch.no_grad()
    def latent(self, observation):
        """z (B, n_latent, llm_dim) for an observation (probes / eval)"""
        images, img_masks, lang_tokens, lang_masks, _state, _ = self._preprocess_observation(observation, train=False)
        vl_ctx, _pe, prefix_pad_masks, _pa, _kv = self._prefix_forward(images, img_masks, lang_tokens, lang_masks, use_cache=False)
        return self._compute_z(vl_ctx, prefix_pad_masks)[0]


def build_model(config, device, taxel_pad=None) -> GloveVTLA:
    m = GloveVTLA(config, taxel_pad=taxel_pad).to(device)
    m.tactile_encoder.float(); m.tactile_predictor.float(); m.recon_head.float(); m.z_proj.float()   # small heads in fp32 (the repo runs the predictor in fp32)
    return m


def _group(keys, depth=2):
    c = collections.Counter(".".join(k.split(".")[:depth]) for k in keys)
    return dict(sorted(c.items()))


def load_weights(model: nn.Module, path: str) -> dict:
    """strict=False load that also skips shape mismatches; returns a report of what loaded and what did not"""
    sd = safetensors.torch.load_file(path, device="cpu"); own = model.state_dict()
    for old, new in (("tactile_prior.", "tactile_predictor."),):  # the repo's legacy key rename
        for k in [k for k in sd if k.startswith(old)]:
            sd[new + k[len(old):]] = sd.pop(k)
    mism = [k for k in sd if k in own and sd[k].shape != own[k].shape]
    for k in mism:
        sd.pop(k)
    res = model.load_state_dict(sd, strict=False)
    rep = dict(loaded=len([k for k in sd if k in own]), missing=_group(res.missing_keys), unexpected=_group(res.unexpected_keys, 3), shape_mismatch=mism)
    logging.info("weights %s: %s", path, rep)
    return rep


def set_stage(model: nn.Module, stage: str) -> list[dict]:
    """requires_grad per stage; returns AdamW param groups ({'params', 'lr_scale'})"""
    for n, p in model.named_parameters():
        if stage == "pred":
            p.requires_grad = n.startswith(TACT_PREFIXES)
        elif stage == "align":
            p.requires_grad = n.startswith(ALIGN_PREFIXES)
        elif stage == "joint":
            p.requires_grad = True
        else:
            raise ValueError(stage)
    model._vl_dropout_override = 1.0 if stage == "align" else None    # align: action queries see no VL prefix (paper stage 2)
    hi = [p for n, p in model.named_parameters() if p.requires_grad and not (stage == "joint" and n.startswith(LOW_LR_PREFIXES))]
    lo = [p for n, p in model.named_parameters() if p.requires_grad and stage == "joint" and n.startswith(LOW_LR_PREFIXES)]
    return [g for g in ({"params": hi, "lr_scale": 1.0}, {"params": lo, "lr_scale": 0.1}) if g["params"]]


def count(groups) -> int:
    return sum(p.numel() for g in groups for p in g["params"])
