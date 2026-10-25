"""AUTHORS' BUILD SCRIPT (needs the simulation replays, which are not public): per dataset row, the state and the two action
targets of the VTLA arms, written to <dataset>/vtla/<id>.npz (+ vtla/check.json). Everyone else downloads the result.

Per video frame (81 at 16 fps; the replay is sampled at the frame's simulation time, hand_pose.npz t_sim):
  state    (81, 32)  [wrist xyz (3) | wrist rot6d (6) | closure (1) | 22 MEASURED finger joints]   (both arms' proprioception)
  act_vts  (81, 32)  = state                                                                     (VTS: the measured state chunk)
  act_vtf  (81, 32)  [wrist xyz | wrist rot6d | grip force (1) | 22 COMMANDED finger joints]       (VTF: force-informed action)
  q_cmd    (81, 22)  the replay's position-servo targets: grasp pose qg + CLOSE_W[f][j] * delta_f(t) on the loaded fingers' closing
                     joints, + the pulp offset on their distal flexion joint (cliprep.replay: d.ctrl = qg + w (delta + kp e) +
                     pulp; e is always 0 there). delta_f is the PI loop's closing offset that makes the simulated normal force of
                     finger f follow the glove, so q_cmd - q carries the grip force (F ~ kp (q_cmd - q), DexForce-style).
                     Silent (unloaded) fingers comply to contacts with offsets the replay does not record: q_cmd = q there.
  f_cmd, f_sim (81, 6)  per part (thumb, index, middle, ring, little, palm) the force target the grip controller tracks and the
                     simulated normal force (Fsim), N. The replay records the UNSCALED glove force (Ftgt, = f_glove); the
                     grip_scale counterfactual multiplies the target by s after that record (its gate hook; the carry floor is 0
                     in these configs), so f_cmd = s x Ftgt with s = the row's params grip_scale for those rows; the finger batch
                     (jitter_fg) scales the glove program before the record, so its Ftgt already is the command (s = 1).
                     grip force (act_vtf[:, 9]) = sum of f_cmd over the loaded fingers.
  closure  = the CLOSE_W-weighted mean flexion of the 5 fingers' closing joints (rad, measured) - the kinematic gripper analogue.
  tactile  (81, 880) the row's glove touch (N, = tactile/<id>.npz touch), t (81,) video time, loaded (5,) bool.
Replays: v2 rows via release_cases.json (control -> cam.json sim); expansion rows via the row's `sim` (scene_mug_dg relative).
Checks (check.json; a failing row stops the build): t_sim inside the replay, finite values, |q_cmd - q| < 60 deg (99th percentile)
on the loaded fingers' closing joints while the glove pushes. Reported: that offset's percentiles (it is NOT always positive: in
the claw / rim-pinch postures the mug wall holds a finger more flexed than its command, e.g. rim pinch index -46 deg at 10.6 N;
the hand pose matches the replay's joint angles exactly there, 0.0 deg), and corr(grip force, total touch) (the leak to watch).
  R2S2R=<workspace root> python build_vtla_labels.py [--dataset <root>/outputs/R2S2V_mug/hf25/dataset] [--assets] [--only id ...]"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

R = Path(os.environ.get("R2S2R", ".")).resolve(); SM = (R / "experiments/scene_mug_dg", R / "experiments/scene_mug")   # the authors' workspace root
HERE = Path(__file__).resolve().parent; ASSETS = HERE.parent / "assets"
FINGERS = ["thumb", "index", "middle", "ring", "little"]; PARTS = FINGERS + ["palm"]
CLOSE_W = {"thumb": {"mano_thumb_0_flex": 0.5, "mano_thumb_1_flex": 1.0, "mano_thumb_2_flex": 1.0},          # = cliprep.replay.CLOSE_W
           **{f: {f"mano_{f}_0_flex": 1.0, f"mano_{f}_1_flex": 1.0, f"mano_{f}_2_flex": 0.8} for f in ("index", "middle", "ring", "little")}}
ALIAS = {"runs/clips/r2s2v_c6/sims/water300": "runs/counterfactual/edf79415_11-16_mano_visgap/object_props/variants/m300_f1_g1/sim",   # Mac symlinks
         "runs/clips/r2s2v_c6/sims/fric05": "runs/counterfactual/edf79415_11-16_mano_visgap/object_props/variants/m0_f0.5_g1/sim"}
N = 81


def rot6d(Rm):
    return np.concatenate([Rm[..., :, 0], Rm[..., :, 1]], -1)


def sim_dir_of(row, cases):
    s = row.get("sim")
    if not s:
        c = cases[row["id"]]; cam = json.loads((R / Path(c["control"]).parent / "cam.json").read_text()); s = cam.get("sim") or ""
        s = s[s.index("runs/"):] if "runs/" in s else s
    s = ALIAS.get(s.rstrip("/"), s.rstrip("/"))
    for root in SM:
        if (root / s / "replay.npz").exists():
            return root / s
    raise FileNotFoundError(f"{row['id']}: replay {s} not found under {[str(x) for x in SM]}")


def write_assets(raw_tactile):
    z = np.load(raw_tactile); pads = [int(p) for p in z["pad_ids"]]; idx = {p: k for k, p in enumerate(pads)}
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "glove_taxels.json").write_text(json.dumps(dict(
        note="TouchScale left glove, 880 taxels in the dataset's channel order; taxel_pad_index = position of the taxel's pad in pad_names",
        pad_names=[str(x) for x in z["pad_names"]], pad_ids=pads, taxel_pad_index=[idx[int(p)] for p in z["taxel_pad"]],
        taxel_names=[str(x) for x in z["taxel_names"]]), indent=0))
    print("wrote", ASSETS / "glove_taxels.json")


def build_row(ds, row, cases):
    cid = row["id"]; sd = sim_dir_of(row, cases); vd = sd.parent; ck = dict(sim=str(sd.relative_to(R)))
    rp = np.load(sd / "replay.npz"); rj = json.loads((sd / "replay.json").read_text()) if (sd / "replay.json").exists() else {}
    g = np.load(vd / "grasp.npz"); qg = dict(zip([str(n) for n in g["joint_names"]], g["q"]))
    hp = np.load(ds / row["hand_pose"]); names = [str(x) for x in hp["finger_q_names"]]; q = hp["finger_q"].astype(np.float64); ts = hp["t_sim"].astype(np.float64)
    loaded = rj.get("loaded_fingers") or [f for k, f in enumerate(FINGERS) if np.abs(rp["delta"][:, k]).max() > 1e-6]
    t = rp["t"]; ck["t_sim_in_replay"] = bool(ts[0] >= t[0] - 0.02 and ts[-1] <= t[-1] + 0.02); ck["loaded"] = loaded
    at = lambda a: np.stack([np.interp(ts, t, a[:, k]) for k in range(a.shape[1])], 1)      # noqa: E731  (holds the end values outside)
    delta, pulp, f_glove, f_sim = at(rp["delta"]), at(rp["pulp"]), at(rp["Ftgt"]), at(rp["Fsim"])
    # grip_scale counterfactuals scale the target AFTER the record (gate hook); the finger batch (kind jitter_fg) scales the glove
    # program BEFORE it (fill_glove_gaps hook: hold_scale x per-finger factor), so its Ftgt is already the command
    s_grip = float((row.get("params") or {}).get("grip_scale", 1.0)) if row.get("kind") in (None, "grip_scale") else 1.0
    f_cmd = s_grip * f_glove; ck["grip_scale_applied"] = s_grip
    q_cmd = q.copy()
    for f in loaded:
        k = FINGERS.index(f)
        for j, nm in enumerate(names):
            if not nm.startswith(f"mano_{f}_"):
                continue
            q_cmd[:, j] = qg[nm] + CLOSE_W[f].get(nm, 0.0) * delta[:, k] + (pulp[:, k] if nm == f"mano_{f}_2_flex" else 0.0)
    wsum = sum(sum(w.values()) for w in CLOSE_W.values())
    closure = sum(w * q[:, names.index(j)] for f in FINGERS for j, w in CLOSE_W[f].items()) / wsum
    Tw = hp["wrist_T_world"]; grip = f_cmd[:, [FINGERS.index(f) for f in loaded]].sum(1)
    state = np.concatenate([Tw[:, :3, 3], rot6d(Tw[:, :3, :3]), closure[:, None], q], 1).astype(np.float32)
    act_vtf = np.concatenate([Tw[:, :3, 3], rot6d(Tw[:, :3, :3]), grip[:, None], q_cmd], 1).astype(np.float32)
    tz = np.load(ds / row["tactile"]); touch = tz["touch"].astype(np.float32); tot = touch.sum(1)
    push = f_cmd[:, :5] > 1.0; off = []
    for f in loaded:
        k = FINGERS.index(f)
        for nm in CLOSE_W[f]:
            j = names.index(nm); off.append((q_cmd[:, j] - q[:, j])[push[:, k]])
    off = np.concatenate(off) if off else np.zeros(1)
    ck.update(frames=int(len(ts)), finite=bool(np.isfinite(state).all() and np.isfinite(act_vtf).all()), cmd_minus_q_deg=dict(
        p5=float(np.degrees(np.percentile(off, 5))), median=float(np.degrees(np.median(off))), p95=float(np.degrees(np.percentile(off, 95)))),
        grip_max_N=float(grip.max()), touch_total_max_N=float(tot.max()), grip_vs_touch_r=float(np.corrcoef(grip, tot)[0, 1]) if grip.std() > 0 and tot.std() > 0 else None)
    ck["abs_cmd_minus_q_p99_deg"] = float(np.degrees(np.percentile(np.abs(off), 99)))
    ck["ok"] = bool(ck["frames"] == N and ck["finite"] and ck["t_sim_in_replay"] and ck["abs_cmd_minus_q_p99_deg"] < 60.0 and touch.shape == (N, 880))
    (ds / "vtla").mkdir(exist_ok=True)
    np.savez_compressed(ds / "vtla" / f"{cid}.npz", state=state, act_vts=state.copy(), act_vtf=act_vtf, q_cmd=q_cmd.astype(np.float32), q=q.astype(np.float32),
                        f_cmd=f_cmd.astype(np.float32), f_glove=f_glove.astype(np.float32), f_sim=f_sim.astype(np.float32), part_names=np.array(PARTS), closure=closure.astype(np.float32),
                        grip_N=grip.astype(np.float32), tactile=touch, t=(np.arange(N) / 16).astype(np.float32), t_sim=ts.astype(np.float32),
                        loaded=np.array([f in loaded for f in FINGERS]), joint_names=np.array(names))
    return ck


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default=str(R / "outputs/R2S2V_mug/hf25/dataset")); ap.add_argument("--only", nargs="*")
    ap.add_argument("--cases", default=str(R / "outputs/R2S2V_mug/hf25/release_cases.json")); ap.add_argument("--assets", action="store_true")
    a = ap.parse_args(); ds = Path(a.dataset)
    if a.assets or not (ASSETS / "glove_taxels.json").exists():
        write_assets(R / "data/mug-sim2video-pose-tactile/r6_head_recorded/tactile.npz")
    cases = {c["id"]: c for c in json.loads(Path(a.cases).read_text())} if Path(a.cases).exists() else {}
    rows = json.loads((ds / "rows.json").read_text()); rows = [r for r in rows if not a.only or r["id"] in a.only]; checks = {}; bad = []
    for r in rows:
        ck = build_row(ds, r, cases); checks[r["id"]] = ck; bad += [] if ck["ok"] else [r["id"]]
        c = ck["cmd_minus_q_deg"]
        print(f"{r['id']:30s} {'ok ' if ck['ok'] else 'BAD'} loaded {','.join(ck['loaded']):18s} q_cmd-q {c['p5']:+5.1f}/{c['median']:+5.1f}/{c['p95']:+5.1f} deg  "
              f"grip max {ck['grip_max_N']:5.1f} N  touch max {ck['touch_total_max_N']:5.1f} N  r(grip, touch) {ck['grip_vs_touch_r'] if ck['grip_vs_touch_r'] is None else round(ck['grip_vs_touch_r'], 3)}", flush=True)
    old = json.loads((ds / "vtla" / "check.json").read_text()) if (ds / "vtla" / "check.json").exists() else {}
    old.update(checks); (ds / "vtla" / "check.json").write_text(json.dumps(old, indent=1))
    print(f"{len(rows) - len(bad)} / {len(rows)} rows ok" + (f"; FAILED {bad}" if bad else ""))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
