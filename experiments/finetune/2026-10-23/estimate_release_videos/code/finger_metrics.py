"""Contact state and timing at the finger level (user 2026-10-24, metric list "Contact state and timing"). Inputs: relative touch
(F / S, 880 glove taxels, frames x 880), clipped at the panel v3 cap like eval_ge.metrics. Parts = panel_v3.FINGER (thumb,
index, middle, ring, little, palm). A part is in contact on a frame when its total relative force > FIN_THR (default 0.04 S,
about 1 N at the median real S of 25.5 N); the same threshold on both sides.
  finger_acc       accuracy of the (frame, part) contact states
  finger_f1        F1 of the in-contact (frame, part) cells (pooled over frames and parts of the video)
  press_dur_err    per part in contact in the truth or the prediction: |predicted - true number of contact frames|; mean (frames)
  contact_end_err  per part in contact in both: |predicted - true last contact frame|; mean (frames). A part the prediction
                   never touches is not in this mean (it costs finger_f1 and press_dur_err)
  load_share_tvd   eval_ge share_tv: per frame where both totals exceed TOT_REL, 0.5 * sum |share pred - share true|; mean"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "2026-10-22/dual_mode_lora/code"))
from eval_ge import metrics, touch_scale  # noqa: E402,F401
from panel_v3 import F_MAX, finger_totals  # noqa: E402

FIN_THR = 0.04
KEYS = ["finger_acc", "finger_f1", "press_dur_err", "contact_end_err", "load_share_tvd"]


def finger_metrics(P, T, thr=FIN_THR):
    P, T = np.minimum(np.clip(P, 0, None), F_MAX), np.minimum(np.clip(T, 0, None), F_MAX); n = min(len(P), len(T)); P, T = P[:n], T[:n]
    cp, ct = np.array([finger_totals(x) for x in P]) > thr, np.array([finger_totals(x) for x in T]) > thr      # (frames, 6)
    tp, fp, fn = (cp & ct).sum(), (cp & ~ct).sum(), (~cp & ct).sum()
    dur = [abs(int(cp[:, j].sum()) - int(ct[:, j].sum())) for j in range(cp.shape[1]) if cp[:, j].any() or ct[:, j].any()]
    last = lambda c: int(len(c) - 1 - np.argmax(c[::-1]))
    end = [abs(last(cp[:, j]) - last(ct[:, j])) for j in range(cp.shape[1]) if cp[:, j].any() and ct[:, j].any()]
    return dict(finger_acc=float((cp == ct).mean()), finger_f1=float(2 * tp / max(2 * tp + fp + fn, 1)),
                press_dur_err=float(np.mean(dur)) if dur else float("nan"), contact_end_err=float(np.mean(end)) if end else float("nan"),
                load_share_tvd=metrics(P, T)["share_tv"])
