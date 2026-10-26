"""Canonical vs D77 yaw-fixed twin of the held-out recordings, side by side (2026-10-26, for the CoRL LBR abstract): the physics
test (policy_gt.sh summaries policy_gtr4.json / policy_gtyf4.json) and the real-glove judge (glove_judge_gtr4.json /
glove_judge_gtyf4.json) per arm, mean ± sd over the 10 noise seeds (references n = 1). 55494490: both twins; 4397541c: canonical
only (its yaw-fixed twin fails its own replay and the hold tuner's pose guard refuses it).
  python yawfix_table.py <results dir>   -> prints the markdown table"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
ARMS = (("T-cmd", "T-cmd", "T-cmd"), ("V-cmd", "V-cmd", "V-cmd"), ("T-obs", "T-obs", "T-obs"),
        ("T-obs + median offset (all frames)", "T-obs-off", "T-obs-off"), ("T-obs + median offset (loaded frames)", "T-obs-offL", "T-obs-offL"),
        ("ref: recording replay", None, "recording replay"), ("ref: cmd", "cmd", "cmd"), ("ref: L-cmd", "label_cmd_16fps", "L-cmd"),
        ("ref: L-obs", "label_obs_16fps", "L-obs"), ("ref: meas", "meas", "meas"))
ROWS = (("55494490", "canonical", "gtr4"), ("55494490", "yaw-fixed", "gtyf4"), ("4397541c", "canonical twin only (yaw ~104 deg off)", "gtr4"))


def ms(v, d):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "n/a"
    return f"{np.mean(v):.{d}f}" if len(v) == 1 else f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}"


def main():
    pol = {t: json.loads((D / f"policy_{t}.json").read_text()) for t in ("gtr4", "gtyf4")}
    glv = {t: json.loads((D / f"glove_judge_{t}.json").read_text())["results"] for t in ("gtr4", "gtyf4")}
    head = ["recording", "twin", "arm", "success", "carried", "in place", "carry force vs recording", "glove r total", "glove r loaded fingers",
            "release error ms", "agreement"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for ep, tw, t in ROWS:
        for name, pk, gk in ARMS:
            p = pol[t]; s = (p["runs"].get(pk) or p["reference"].get(pk) or {}).get(ep, {}) if pk else {}
            g = glv[t].get(ep, {}).get(gk, [])
            if not g and not s:
                continue
            cells = [ep, tw, name, s.get("success", "n/a") if pk else "n/a", s.get("carried", "n/a") if pk else "n/a", s.get("final_ok", "n/a") if pk else "n/a",
                     f"{s['carry_force_ratio_median']:.2f}" if pk and s.get("carry_force_ratio_median") is not None else "n/a",
                     ms([r["r_total"] for r in g], 2), ms([r["r_finger"] for r in g], 2), ms([r["offset_abs_ms"] for r in g], 0), ms([r["agreement"] for r in g], 3)]
            lines.append("| " + " | ".join(cells) + " |")
    print("\n".join(lines)); print()
    for ep, tw, t in ROWS:
        r = glv[t][ep]["recording replay"][0]; print(f"{ep} {tw}: glove-loaded {r['glove_loaded']}, loaded in the recording's replay {r['sim_loaded']}")


if __name__ == "__main__":
    main()
