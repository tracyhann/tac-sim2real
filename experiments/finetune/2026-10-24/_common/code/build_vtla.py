"""Action-model view of the HF25 dataset (2026-10-25; user: "按这个模型的架构，把训练数据的格式先调整一下，然后把 prompt 对齐"):
the format a vision-tactile-language-action model (N0-VTLA-style: images + language + state in the prefix, the tactile reading as
its own tokens, an action chunk as the output, the future tactile change as an auxiliary target) consumes. Runs on an existing
dataset folder (build_hf25.py output + handpose/) and adds, without touching the videos:
  action/<id>.npz   per video frame (81 at 16 fps):
      state        (81, 31) float32  the hand state = action target: wrist position in the world (= desk) frame (3), wrist
                                     rotation as rot6d (the first two columns of the 3 x 3 rotation, 6), 22 finger joint angles
                                     (rad); state_names gives the order. A policy predicts a CHUNK of future states, written
                                     relative to the chunk's first frame (done in the data loader, not stored).
      tactile      (81, 880) float32 the glove's touch above each sensor's baseline (N); tactile_rel = tactile / S (S = the
                                     row's touch scale), tactile_u = min(tactile_rel, 0.11) / 0.11. The current reading is an
                                     input; the change over the coming chunk is the auxiliary prediction target.
      t (81,), fps, S, schema (text)
  metadata_vtla.json   one manifest with every modality per row: video, start frame, handpose (21 keypoints, camera pose),
                       action, tactile, the three control videos (not used by an action model), text, params, camera, group
  PROMPTS.md           the table of all rows' prompts and parameters (for review)
and rewrites `text` + `params` in rows.json / metadata_FHN|SVO|CDS.json from prompts.py (v2: task instruction + ONE quantitative
clause per perturbation; no touch-panel sentence). Checks: handpose frames = video frames = 81, rot6d columns orthonormal,
finger_q within [-pi, pi], tactile shapes, every file present.
  python build_vtla.py [DATASET_DIR]     (default $R2S2R/outputs/R2S2V_mug/hf25/dataset)"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from prompts import ACTION, params_for, prompt_for  # noqa: E402

N, FPS, CAP = 81, 16, 0.11
STATE_NAMES = (["wrist_x", "wrist_y", "wrist_z"] + [f"wrist_rot6d_{i}" for i in range(6)]
               + ["index_0_abd", "index_0_flex", "index_1_flex", "index_2_flex", "middle_0_abd", "middle_0_flex", "middle_1_flex", "middle_2_flex",
                  "little_0_abd", "little_0_flex", "little_1_flex", "little_2_flex", "ring_0_abd", "ring_0_flex", "ring_1_flex", "ring_2_flex",
                  "thumb_0_abd", "thumb_0_flex", "thumb_0_roll", "thumb_1_abd", "thumb_1_flex", "thumb_2_flex"])
SCHEMA = ("state[t] = [wrist position xyz (m, world = desk frame), wrist rotation rot6d (R[:, 0], R[:, 1] of wrist_T_world), 22 finger joint "
          "angles (rad, order of state_names)]; action chunk = state[t+1 : t+1+H] expressed relative to state[t] (positions and rotation relative "
          "to the chunk's first wrist pose, joint angles absolute); tactile[t] = glove normal force above baseline (N) per sensor, tactile_rel = / S, "
          "tactile_u = min(tactile_rel, 0.11) / 0.11; auxiliary target = tactile_rel[t+H] - tactile_rel[t]")


def rot6d(R):
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def build_row(ds, r):
    hp = np.load(ds / r["hand_pose"]); tz = np.load(ds / r["tactile"]); cid = r["id"]
    T = hp["wrist_T_world"]; q = hp["finger_q"]; assert T.shape == (N, 4, 4) and q.shape == (N, 22), (cid, T.shape, q.shape)
    names = [str(x).replace("mano_", "") for x in hp["finger_q_names"]]; assert names == STATE_NAMES[9:], (cid, names[:3])
    R = T[:, :3, :3]; ortho = float(np.abs(np.einsum("tij,tik->tjk", R, R) - np.eye(3)).max()); assert ortho < 1e-3, (cid, ortho)
    assert np.abs(q).max() < np.pi, (cid, float(np.abs(q).max()))
    state = np.concatenate([T[:, :3, 3], rot6d(R), q], 1).astype(np.float32)
    touch = tz["touch"].astype(np.float32); S = float(tz["S"]); assert touch.shape == (N, 880), (cid, touch.shape)
    rel = touch / S; u = np.minimum(np.clip(rel, 0, None), CAP) / CAP
    (ds / "action").mkdir(exist_ok=True)
    np.savez_compressed(ds / "action" / f"{cid}.npz", state=state, state_names=np.array(STATE_NAMES), tactile=touch, tactile_rel=rel.astype(np.float32),
                        tactile_u=u.astype(np.float32), S=np.float32(S), t=hp["t_video"].astype(np.float32), fps=np.int64(FPS), schema=np.array(SCHEMA))
    path_m = float(np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1).sum())
    return dict(action=f"action/{cid}.npz", wrist_path_m=round(path_m, 3), contact_frames=int((touch.sum(1) > 1.0).sum()))


def main(ds):
    ds = Path(ds); rows = json.loads((ds / "rows.json").read_text()); vt = []
    for r in rows:
        v = r["variant"]; r["text"] = prompt_for(v); r["params"] = params_for(v); r.update(build_row(ds, r))
        vt.append(dict(id=r["id"], group=r["group"], camera=r.get("camera"), perturbation=r.get("perturbation"), video=r["file_path"], start=r["start"],
                       handpose=r["hand_pose"], action=r["action"], tactile=r["tactile"], controls=dict(own=r["ctrl_own"], sim=r["ctrl_sim"], cube=r["ctrl_cube"]),
                       text=r["text"], params=r["params"], S=r["S"], frames=N, fps=FPS, wrist_path_m=r["wrist_path_m"], contact_frames=r["contact_frames"]))
        print(f"{r['id']:32s} {v:9s} path {r['wrist_path_m']:.2f} m  contact {r['contact_frames']}/81  text: {r['text'][:70]}...", flush=True)
    (ds / "rows.json").write_text(json.dumps(rows, indent=1))
    for arm, col in (("FHN", "ctrl_own"), ("SVO", "ctrl_sim"), ("CDS", "ctrl_cube")):
        (ds / f"metadata_{arm}.json").write_text(json.dumps([dict(r, control_file_path=r[col], arm=arm) for r in rows], indent=1))
    (ds / "metadata_vtla.json").write_text(json.dumps(dict(schema=SCHEMA, instruction=ACTION, state_dim=31, tactile_dim=880, frames=N, fps=FPS,
                                                           note="prompts v2 (2026-10-25): task instruction + one quantitative clause per perturbation; params = the simulation's settings",
                                                           rows=vt), indent=1))
    lines = ["# HF25 rows: prompts and parameters (v2, 2026-10-25)", "", f"Shared instruction: *{ACTION}*", "",
             "| # | id | group / camera | variant | params (non-default) | prompt |", "|---|---|---|---|---|---|"]
    for k, r in enumerate(rows, 1):
        nd = {a: b for a, b in r["params"].items() if a != "variant" and b not in (1.0, 0, 0.0, "recorded")}
        cam = (r.get("camera") or "").split(" (")[0]
        lines.append(f"| {k} | {r['id']} | {r['group']} / {cam} | {r['variant']} | {json.dumps(nd) if nd else '-'} | {r['text']} |")
    (ds / "PROMPTS.md").write_text("\n".join(lines) + "\n")
    print(f"rows {len(rows)}; distinct prompts {len({r['text'] for r in rows})}; action/ written; PROMPTS.md written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(os.environ.get("R2S2R", "/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r")) / "outputs/R2S2V_mug/hf25/dataset")
