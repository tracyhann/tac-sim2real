"""Tactile scores of a decoded prediction P (F, 880) N against a target T (F, 880) N (both clipped at F_MAX):
taxel MAE, MAE on contact taxels (T > 0.15 N), contact F1 (P > 0.15 vs T > 0.15, over frames with any contact),
per-finger total force: Pearson r over time (mean of the six) and MAE, total force r / MAE, onset error (first frame
with total > 5 N)."""
import numpy as np

from tactile_panel import F_MAX, finger_totals

THR, ONSET_N = 0.15, 5.0


def scores(P, T):
    P, T = np.minimum(np.clip(P, 0, None), F_MAX), np.minimum(np.clip(T, 0, None), F_MAX); n = min(len(P), len(T)); P, T = P[:n], T[:n]
    on_t, on_p = T > THR, P > THR; fr = on_t.any(1) | on_p.any(1)
    tp, fp_, fn = (on_t & on_p)[fr].sum(1), (~on_t & on_p)[fr].sum(1), (on_t & ~on_p)[fr].sum(1)
    f1 = float(np.mean(2 * tp / np.maximum(2 * tp + fp_ + fn, 1))) if fr.any() else float("nan")
    ft, fp = np.array([finger_totals(t) for t in T]), np.array([finger_totals(p) for p in P])
    rs = [float(np.corrcoef(ft[:, i], fp[:, i])[0, 1]) for i in range(ft.shape[1]) if ft[:, i].std() > 1e-3 and fp[:, i].std() > 1e-3]
    on = lambda x: int(np.argmax(x > ONSET_N)) if (x > ONSET_N).any() else -1
    ot, op = on(T.sum(1)), on(P.sum(1))
    return dict(taxel_mae=float(np.abs(P - T).mean()), contact_mae=float(np.abs(P - T)[on_t].mean()) if on_t.any() else float("nan"), contact_f1=f1,
                finger_r=float(np.mean(rs)) if rs else float("nan"), finger_mae=float(np.abs(fp - ft).mean()),
                total_r=float(np.corrcoef(T.sum(1), P.sum(1))[0, 1]) if P.sum(1).std() > 1e-3 else float("nan"), total_mae=float(np.abs(P.sum(1) - T.sum(1)).mean()),
                onset_err_frames=(op - ot) if ot >= 0 and op >= 0 else None)
