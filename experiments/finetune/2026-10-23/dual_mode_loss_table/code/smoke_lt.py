import os, sys, torch; os.environ["GE_DEV"] = "cpu"; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loss_table as L
torch.manual_seed(0); g = (torch.rand(2, 81, 880) ** 6).clamp(0, 1); g[:, :10] = 0
for v in list(L.GRID) + list(L.EXTRA):
    m = L.Net(L.OUT_DIM[v], sp=v.endswith('sp')); z = torch.randn(2, 16, 21, 60, 104)
    y = m(z); l = L.loss_fn(v, y, g, ({**L.GRID, **L.EXTRA}[v][1] if len({**L.GRID, **L.EXTRA}[v]) > 1 else 0)); l.backward()
    u = L.decode(v, y); print(v, tuple(y.shape), "loss", round(float(l), 4), "u", tuple(u.shape), round(float(u.min()), 4), round(float(u.max()), 4), "grad ok", m.out.weight.grad is not None)
print("units", L.UM.shape, "cost", L.CM.shape, "centres", [round(float(c), 3) for c in L.CENTRES])
