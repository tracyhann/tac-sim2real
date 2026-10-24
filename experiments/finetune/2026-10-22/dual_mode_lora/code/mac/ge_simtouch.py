"""Simulated glove touch at the canvas times for every sim row of the dual-mode (generate / estimate) video + touch LoRA
(Vision experiments/R2S2V_mug/lora_ge). One (81, 880) array per replay: the replay's txc30 (30 Hz, one column per placed
taxel of the hand asset) linearly interpolated to t0 + k / 16 s and scattered into the glove's 880 columns through that
hand's taxels.json (863 placed taxels for hand_actor05_left_mano_v1, 880 for ..._880; picked by the column count), the
same rule as r2s2v/lora_tactile_data.clip_tactile. Variants: the four twins' own replays, the edf79415 palm-press p8
(round 5) and the round-6 counterfactuals of edf79415 (visgap scenes; same t0 = 11.0 s as the clip).
Alignment check (D68): every array's times are the canvas times of its clip's control (views5 uses the same t0 + k / 16),
and the twins' arrays must equal runs/clips/r2s2v_lora/dataset_ho/tactile/<clip>.npz["sim"] (built by the old builder).
  .venv/bin/python r2s2v/ge_simtouch.py OUTDIR"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
from cliprep.core import Clip  # noqa: E402

N, FPS = 81, 16
CF = "runs/counterfactual/edf79415_11-16_mano_visgap"
VARIANTS = {  # name: (clip config, replay dir)
    "tsmug-edf79415": ("edf79415_11-16_mano", "runs/clips/edf79415_11-16_mano/sim"),
    "tsmug-53780879": ("53780879_auto_mano", "runs/clips/53780879_auto_mano/sim"),
    "tsmug-55494490": ("55494490_auto_mano", "runs/clips/55494490_auto_mano/sim"),
    "tsmug-4397541c": ("4397541c_auto_mano", "runs/clips/4397541c_auto_mano/sim"),
    "edf-p8": ("edf79415_11-16_mano", "runs/counterfactual/edf79415_11-16_mano/palm_press/variants/p8/sim"),
    "edf-c6recorded": ("edf79415_11-16_mano", f"{CF}/motion/variants/base/sim"),
    **{f"edf-c6{n}": ("edf79415_11-16_mano", f"{CF}/posture/variants/{n}/sim") for n in ("claw", "pinch2", "rimpinch", "wrap")},
    "edf-c6grip030": ("edf79415_11-16_mano", f"{CF}/grip_scale/variants/s0.3_off/sim"),
    "edf-c6fric05": ("edf79415_11-16_mano", f"{CF}/object_props/variants/m0_f0.5_g1/sim"),
    "edf-c6water300": ("edf79415_11-16_mano", f"{CF}/object_props/variants/m300_f1_g1/sim"),
    "edf-c6t10": ("edf79415_11-16_mano", f"{CF}/motion/variants/t10.00/sim"),
    "edf-c6shake60": ("edf79415_11-16_mano", f"{CF}/motion/variants/shake_lat_60mm_4hz/sim")}
CTRL = {"edf-p8": "views5_p8", **{f"edf-c6{n}": f"views6_{n}" for n in ("recorded", "claw", "pinch2", "rimpinch", "wrap", "grip030", "fric05", "water300", "t10", "shake60")}}
HANDS = [HERE / "assets/hand_actor05_left_mano_v1", HERE / "assets/hand_actor05_left_mano_v1_880"]


def taxel_idx(ncol):
    for h in HANDS:
        tj = json.loads((h / "taxels.json").read_text())["taxels"]
        if len(tj) == ncol:
            return np.array([t["idx"] for t in tj]), h.name
    raise SystemExit(f"no hand asset with {ncol} taxels")


def sim_touch(z, ts):
    idx, hand = taxel_idx(z["txc30"].shape[1]); S30 = np.maximum(z["txc30"], 0); S = np.zeros((len(ts), 880), np.float32)
    assert ts[0] >= z["t30"][0] - 1e-3 and ts[-1] <= z["t30"][-1] + 1 / 30 + 1e-3, ("canvas times outside the replay", ts[[0, -1]], z["t30"][[0, -1]])
    for k, t in enumerate(ts):
        j = np.searchsorted(z["t30"], t); j0, j1 = max(j - 1, 0), min(j, len(z["t30"]) - 1); w = 0 if j0 == j1 else (t - z["t30"][j0]) / (z["t30"][j1] - z["t30"][j0])
        S[k, idx] = (1 - w) * S30[j0] + w * S30[j1]
    return S, hand


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); rep = {}
    for name, (cfg, rd) in VARIANTS.items():
        rp = HERE / rd / "replay.npz"
        if not rp.exists():
            print("skip (no replay)", name, rd); continue
        c = Clip(HERE / f"configs/clips/{cfg}.yaml"); ts = c.t0 + np.arange(N) / FPS; z = np.load(rp)
        S, hand = sim_touch(z, ts); r = dict(t0=float(ts[0]), hand=hand, replay=str(rd))
        ctl = HERE / "runs/clips" / c.name.replace("_visgap", "") / "wm" / CTRL.get(name, "views5") / "head" / "fun_ctrl.mp4"
        if not ctl.exists():
            print("skip (no control yet)", name, ctl.relative_to(HERE)); continue
        r["replay_older_than_control"] = rp.stat().st_mtime < ctl.stat().st_mtime
        if not r["replay_older_than_control"]:
            # the replay was rewritten after its control (and the zero-shot videos) were rendered: the touch must come from
            # the replay the control shows. The two LoRA datasets built before (10-21 20:47) and after (10-22 11:45) that
            # render hold its sim touch at these times; use it only if both agree (the replay did not change in between).
            a, b = (np.load(HERE / "runs/clips/r2s2v_lora" / d / "tactile" / f"{name}.npz") for d in ("dataset", "dataset_ho"))
            files = [HERE / "runs/clips/r2s2v_lora" / d / "tactile" / f"{name}.npz" for d in ("dataset", "dataset_ho")]
            assert files[0].stat().st_mtime < ctl.stat().st_mtime < files[1].stat().st_mtime < rp.stat().st_mtime, (name, "no saved touch brackets the control render")
            assert np.abs(a["sim"] - b["sim"]).max() < 1e-5 and np.abs(b["t"] - ts).max() < 1e-6, (name, "the replay changed between the two saved datasets")
            r["source"] = "dataset_ho (replay as rendered in the control)"; r["current_replay_maxdiff_N"] = float(np.abs(b["sim"] - S).max()); S = b["sim"].astype(np.float32)
        else:
            r["source"] = "replay"
            old = HERE / "runs/clips/r2s2v_lora/dataset_ho/tactile" / f"{name}.npz"
            if old.exists():                                     # same replay -> the old builder's array must match
                o = np.load(old); r["vs_dataset_ho_sim_maxdiff"] = float(np.abs(o["sim"] - S).max())
                assert r["vs_dataset_ho_sim_maxdiff"] < 1e-4 and np.abs(o["t"] - ts).max() < 1e-6, (name, r)
        r.update(total_max_N=round(float(S.sum(1).max()), 1), contact_frames=int((S.sum(1) > 1).sum()))
        np.savez_compressed(out / f"{name}.npz", sim=S, t=ts, replay=str(rd), hand=hand, source=r["source"])
        rep[name] = r; print(name, r)
    (out / "simtouch.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
