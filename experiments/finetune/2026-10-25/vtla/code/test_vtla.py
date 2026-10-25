"""CPU unit test, no weights, no data (run by setup.sh): the tiny-Gemma ("dummy") GloveVTLA on a fake batch.
Checks the taxel encoder's token layout, that the repo's tactile path uses it, the action loss + the predictor aux loss are
finite and train what each stage says, the z gate opens under the action loss, sampling returns (B, H, 32), and a
save / load round trip reproduces the actions."""
import tempfile
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

import vtla_data as D
import vtla_model as M
from taxel_encoder import ENCODERS, TaxelGridEncoder, TaxelTactileEncoder, load_taxel_pads


def fake_batch(B=4, H=8, L=48, seed=0):
    r = np.random.default_rng(seed); tac = np.abs(r.normal(0, 0.3, (B, 880))).astype(np.float32) * (r.random((B, 880)) < 0.1)
    return dict(image=r.integers(0, 255, (B, D.IMG, D.IMG, 3), dtype=np.uint8), state=r.uniform(-1, 1, (B, 32)).astype(np.float32),
                tokens=r.integers(1, 1000, (B, L)).astype(np.int64), token_mask=np.ones((B, L), bool), tactile=tac,
                tac_delta=r.normal(0, 0.2, (B, 880)).astype(np.float32), actions=r.normal(0, 0.5, (B, H, 32)).astype(np.float32))


def main():
    torch.manual_seed(0); pads, names = load_taxel_pads(); assert len(pads) == 880 and len(names) == 15, (len(pads), len(names))
    for name, cls in ENCODERS.items():
        tok = cls(64, taxel_pad=pads)(torch.rand(3, 880)); assert tok.shape == (3, 16, 64) and torch.isfinite(tok).all(), (name, tok.shape)
    g = TaxelGridEncoder(64, taxel_pad=pads); x = torch.zeros(1, 880); j = int(np.flatnonzero(pads == 3)[0])     # one press on index_distal
    x[0, j] = 1.0; s = g.summaries(x)[0, 3]; assert s[0] > 0 and abs(float(s[2]) + 1) < 1e-5 and abs(float(s[3]) + 1) < 1e-5, s   # centroid = that taxel's (row, col) = (-1, -1)
    x2 = torch.zeros(1, 880); x2[0, j + 8] = 1.0; d1 = (g(x) - g(x2)).norm(); x3 = torch.zeros(1, 880); x3[0, int(np.flatnonzero(pads == 12)[0])] = 1.0
    print(f"grid encoder: one-row shift moves the tokens {float(d1):.3f}, a press on another finger {float((g(x) - g(x3)).norm()):.3f}")
    H = 8
    for enc in ("mlp", "grid"):
        m = M.build_model(M.make_config(H, dummy=True, dtype="float32"), torch.device("cpu"), encoder=enc)
        assert isinstance(m.tactile_encoder, ENCODERS[enc]), type(m.tactile_encoder)
        run_stages(m, H, enc)
    print("test_vtla: all checks passed")


def run_stages(m, H, enc):
    b = fake_batch(H=H); obs, act, tdel = D.to_obs(b, "cpu")
    for stage in ("pred", "align", "joint"):
        groups = M.set_stage(m, stage); names_tr = {n for n, p in m.named_parameters() if p.requires_grad}
        opt = torch.optim.AdamW(groups, lr=1e-3); m.zero_grad(set_to_none=True)
        loss = m.aux_loss(obs, tdel)[0] if stage == "pred" else m(obs, act).mean()
        assert torch.isfinite(loss), (stage, loss); opt.zero_grad(); loss.backward(); opt.step()
        got = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        assert got <= names_tr, (stage, sorted(got - names_tr)[:5])
        if stage == "pred":
            assert all(n.startswith(M.TACT_PREFIXES) for n in got) and any(n.startswith("tactile_encoder.") for n in got), sorted(got)[:5]
        if stage == "align":
            assert not any(n.startswith("paligemma_with_expert.paligemma.") for n in got) and "z_gate" in got, sorted(got)[:5]
        print(f"[{enc}] stage {stage}: loss {float(loss):.4f}, {len(got)} tensors got gradients, trainable {M.count(groups) / 1e6:.1f} M")
    assert abs(float(m.z_gate)) > 0, "z gate did not move"
    m.eval(); noise = torch.randn(4, H, 32)
    x = m.sample_actions(torch.device("cpu"), D.to_obs(b, "cpu")[0], noise=noise, num_steps=3); assert x.shape == (4, H, 32) and torch.isfinite(x).all()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "model.safetensors"; safetensors.torch.save_model(m, str(p))
        m2 = M.build_model(M.make_config(H, dummy=True, dtype="float32"), torch.device("cpu"), encoder=enc); rep = M.load_weights(m2, str(p)); m2.eval()
        x2 = m2.sample_actions(torch.device("cpu"), D.to_obs(b, "cpu")[0], noise=noise, num_steps=3)
        assert torch.allclose(x, x2, atol=1e-5), float((x - x2).abs().max()); assert not rep["shape_mismatch"], rep
    z = m.latent(D.to_obs(b, "cpu")[0]); assert z.shape[:2] == (4, 5), z.shape


if __name__ == "__main__":
    main()
