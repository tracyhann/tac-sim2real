"""Dataset for the three HF25 video + touch LoRAs (user 2026-10-24; FHN / SVO / CDS). One set of 25 training pairs, three
metadata files that differ ONLY in the control video, so the control is the only difference between the arms.

Rows = the 25 pairs of the public release (data/mug-sim2video-pose-tactile, fetch_hf.sh, commit 6cf129c): the zero-shot A14B
video (832 x 480, 81 frames, 16 fps, copied as is) + the replay's simulated touch of the left glove (normal_N, 880 channels,
sampled at the video frames). The touch stays a VALUE array (hf25_patch.py, user 10-24: never pixels): tactile/<id>.npz with
touch (N), touch_rel = F / S, S, t and u = min(F / S, CAP) / CAP, the head's target; S = median total force over the frames with
total > 1 N of that replay (its 81 frames span the whole 5 s window) = the normalisation the F / H runs lacked.
Prompt = prompts.py (one shared sentence + one visual-feature sentence where the video has one).
Controls (854 x 480 sources resized to 832 x 480, as at generation):
  FHN  ctrl_own   the control the video was generated from = the F / H format: static scene depth (inverse-depth grey) +
                  shadowless sim hand / mug / coaster pixels; r6 head = fused-twin depth (views6), r8b = G1 cubes (head) or
                  the sim desk box (top photo) (views9), r10 = cubes from the user frame (views10)
  SVO  ctrl_sim   the sim render alone: hand / mug / coaster pixels on black, no depth (sim.mp4 of views9 / views10; the r6
                  rows take views9 of the same variant: same replay, same head camera, because views6's sim.mp4 also draws
                  the textured twin mesh)
  CDS  ctrl_cube  cube-scene depth (no body) + the sim render on top (fun_ctrl.mp4 of views9 / views10; r6 rows -> views9)
  NOTE: for the 19 r8b / r10 rows CDS = FHN (those videos were generated from cube controls); they differ on the 6 r6 rows.
Test cases (test_cases.json, used by predict_hf25.py / regr_hf25.py / score_hf25.py):
  held-out  the twins 55494490 / 4397541c (never in any HF25 row), head view: start = real frame 0 + its touch (target =
            the real touch, relative to the episode's S) and start = the zero-shot generated frame 0 + sim touch (target =
            the sim touch); starts / touch from lora_ge/dataset (the canvases cropped to the video); control per arm: FHN
            views5/head fun_ctrl (twin depth + sim), SVO / CDS views9_recorded/head sim.mp4 / fun_ctrl.mp4 (mac/twin_cubes.sh)
  in-sample 3 training rows (one per group), own start and touch: did the arm fit its data
Checks (the build stops on a failure; check.json): HF video = the Vision generation it came from (MAE), 81 frames at 16 fps for
every stream, touch times = video times, finite non-negative touch, the r6 -> views9 swap shows the same replay (cam.json sim),
SVO controls mostly black (no scene drawn). Reported: S, active fraction, how much of the active touch the cap clips, HF touch
vs the dual-mode builder's sim touch of the same replay.
  envs/armA/bin/python build_hf25.py   -> outputs/R2S2V_mug/hf25/dataset/"""
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); HERE = Path(__file__).resolve().parent
DM = R / "experiments/finetune/2026-10-22/dual_mode_lora/code"          # io helpers (imported, not copied)
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(DM))
from ge_data import read_video, touch_scale, write_video  # noqa: E402
from hf25_patch import CAP, THR_U, u_from_rel  # noqa: E402
from prompts import BASE, PANEL, prompt_for  # noqa: E402

HFD = R / "data/mug-sim2video-pose-tactile"; O = R / "outputs/R2S2V_mug/hf25"; DS = O / "dataset"
SIM = R / "outputs/B_sim2wm/sim"; GEN = R / "outputs/A_baseline/gen"; LG = R / "outputs/R2S2V_mug/lora_ge"
N, FPS, W, H = 81, 16, 832, 480
ARMS = {"FHN": "ctrl_own", "SVO": "ctrl_sim", "CDS": "ctrl_cube"}
INSAMPLE = ("r6_head_recorded", "r8b_topphoto_shake60mm", "r10_closeup_rimpinch")
TWINS = ("55494490", "4397541c")
CTRL_CACHE = {}


def control(src):
    """src (854 x 480 control or sim render, 81 frames) -> dataset ctrl/<clip>_<vdir>_<view>_<kind>.mp4 (832 x 480)"""
    src = Path(src); key = str(src)
    if key not in CTRL_CACHE:
        kind = "sim" if src.name == "sim.mp4" else "fun"; clip = src.parents[2].name
        p = DS / "ctrl" / f"{clip}_{src.parent.parent.name}_{src.parent.name}_{kind}.mp4"
        fr, n = read_video(src); assert n == N, (src, n)
        if not p.exists():
            write_video(p, list(fr))
        CTRL_CACHE[key] = (str(p.relative_to(DS)), float((fr.max(-1) <= 10).mean()))
    return CTRL_CACHE[key]


def cam_sim(d):
    """the replay a view was rendered from, as a scene_mug-relative path (some cam.json files hold the absolute Mac path)"""
    s = json.loads((d / "cam.json").read_text()).get("sim") or ""
    return s[s.index("runs/"):] if "runs/" in s else s


def save_touch(path, T, S, t, **extra):
    """touch (frames, 880) N + the replay / episode scale S -> npz with the relative touch and the head target u"""
    Tr = (T / S).astype(np.float32); u = u_from_rel(Tr)
    np.savez_compressed(path, touch=T.astype(np.float32), touch_rel=Tr, S=np.float32(S), t=t, u=u, **extra)
    act = Tr > THR_U * CAP
    return dict(S_N=float(S), total_max_N=float(T.sum(1).max()), active_frac=float(act.mean()), clip_frac_active=float((Tr[act] > CAP).mean()) if act.any() else 0.0,
                u_mean=float(u.mean()), contact_frames=int((T.sum(1) > 1.0).sum()))


def build_row(c, meta, checks):
    cid = c["id"]; own = R / c["control"]; vd = own.parent; vdir, view = vd.parent.name, vd.name; variant = vdir.split("_")[1]
    cube_dir = vd.parent.parent / f"views9_{variant}" / "head" if vdir.startswith("views6_") else vd
    ck = dict(group=c["group"], variant=variant, view=view, own=str(own.relative_to(R)), cube_dir=str(cube_dir.relative_to(R)), replay=cam_sim(vd))
    ck["same_replay"] = cam_sim(cube_dir) == cam_sim(vd)
    hv, nh = read_video(HFD / cid / "video.mp4"); gv, ng = read_video(R / c["gen"])
    ck["hf_frames"], ck["hf_vs_gen_mae"] = nh, float(np.abs(hv.astype(np.float32) - gv.astype(np.float32)).mean()) if ng == nh else None
    z = np.load(HFD / cid / "tactile.npz"); T = z["normal_N"].astype(np.float32); tv = z["t_video"]
    ck["touch_shape"] = list(T.shape); ck["touch_dt_err"] = float(np.abs(np.diff(tv) - 1 / FPS).max()); ck["touch_finite"] = bool(np.isfinite(T).all() and (T >= 0).all())
    ck["uncovered_channels"] = int((~z["sim_covered"]).sum())
    st = LG / "simtouch" / f"edf-c6{variant}.npz"                          # the dual-mode builder's touch of the same replay (r6 variants)
    if st.exists():
        Ts = np.load(st)["sim"].astype(np.float32); n_ = min(len(Ts), len(T))
        ck["vs_lora_ge_simtouch"] = dict(max_abs_N=float(np.abs(Ts[:n_] - T[:n_]).max()), total_r=float(np.corrcoef(Ts[:n_].sum(1), T[:n_].sum(1))[0, 1]))
    shutil.copy2(HFD / cid / "video.mp4", DS / "video" / f"{cid}.mp4")       # the video as released (= the Vision generation)
    cv2.imwrite(str(DS / "start" / f"{cid}.png"), hv[0][..., ::-1])
    ck.update(save_touch(DS / "tactile" / f"{cid}.npz", T, touch_scale(T), tv, sim_covered=z["sim_covered"]))
    (co, _), (cs, black), (cc, _) = control(own), control(cube_dir / "sim.mp4"), control(cube_dir / "fun_ctrl.mp4")
    ck["svo_black_frac"] = black
    ok = (ck["same_replay"] and nh == N and ng == N and ck["hf_vs_gen_mae"] is not None and ck["hf_vs_gen_mae"] < 3.0 and ck["touch_shape"] == [N, 880]
          and ck["touch_dt_err"] < 1e-6 and ck["touch_finite"] and ck["contact_frames"] > 10 and black > 0.3)
    ck["ok"] = bool(ok); checks[cid] = ck
    print(cid, "ok" if ok else "FAILED", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ck.items() if k not in ("own", "cube_dir", "replay")}, flush=True)
    return dict(id=cid, file_path=f"video/{cid}.mp4", control_file_path=None, text=prompt_for(variant), type="video", width=W, height=H,
                group=c["group"], variant=variant, view=view, perturbation=meta.get("perturbation"), camera=meta.get("camera"), S=ck["S_N"],
                tactile=f"tactile/{cid}.npz", start=f"start/{cid}.png", ctrl_own=co, ctrl_sim=cs, ctrl_cube=cc), ok


def test_cases(rows):
    out, notes = [], []
    (DS / "test" / "start").mkdir(parents=True, exist_ok=True); (DS / "test" / "tactile").mkdir(parents=True, exist_ok=True)
    for ep8 in TWINS:
        clip = f"tsmug-{ep8}"; ctl = {"FHN": SIM / clip / "views5/head/fun_ctrl.mp4", "SVO": SIM / clip / "views9_recorded/head/sim.mp4",
                                      "CDS": SIM / clip / "views9_recorded/head/fun_ctrl.mp4"}
        if ctl["SVO"].exists():
            assert cam_sim(ctl["SVO"].parent).endswith(f"{ep8}_auto_mano/sim"), (clip, cam_sim(ctl["SVO"].parent))
        cmap = {a: (control(p)[0] if p.exists() else None) for a, p in ctl.items()}
        notes += [f"{clip}: no {a} control yet ({p.relative_to(R)}): run mac/twin_cubes.sh, then rebuild" for a, p in ctl.items() if not p.exists()]
        for src, rid in (("real", f"real_{ep8}"), ("sim", f"sim_{ep8}_head_s0")):
            canvas = cv2.imread(str(LG / "dataset" / "start" / f"{rid}.png")); assert canvas.shape[1] > W, canvas.shape
            cv2.imwrite(str(DS / "test" / "start" / f"{rid}.png"), canvas[:, :W])                         # the video part of the old canvas
            z = np.load(LG / "dataset" / "tactile" / f"{rid}.npz"); info = save_touch(DS / "test" / "tactile" / f"{rid}.npz", z["touch"], float(z["S"]), z["t"])
            out.append(dict(case=f"G_{src}_{ep8}", kind="heldout", clip=clip, source=src, start=f"test/start/{rid}.png", tactile=f"test/tactile/{rid}.npz",
                            control=cmap, text=BASE + PANEL, S=info["S_N"], target="real touch (relative, episode S)" if src == "real" else "sim touch (relative, replay S)"))
    for rid in INSAMPLE:
        r = next(x for x in rows if x["id"] == rid)
        out.append(dict(case=f"G_train_{rid}", kind="insample", clip="tsmug-edf79415", source="gen", start=r["start"], tactile=r["tactile"],
                        control={a: r[col] for a, col in ARMS.items()}, text=r["text"], S=r["S"], target="sim touch of the training row"))
    return out, notes


def main():
    for d in ("video", "start", "tactile", "ctrl"):
        (DS / d).mkdir(parents=True, exist_ok=True)
    cases = json.loads((HERE / "release_cases.json").read_text())
    meta = {m["id"]: m for m in csv.DictReader((HFD / "metadata.csv").open())}
    assert set(meta) == {c["id"] for c in cases} and len(cases) == 25, "release cases != HF metadata"
    rows, checks, bad = [], {}, []
    for c in cases:
        r, ok = build_row(c, meta[c["id"]], checks); rows.append(r); bad += [] if ok else [c["id"]]
    for arm, col in ARMS.items():
        (DS / f"metadata_{arm}.json").write_text(json.dumps([dict(r, control_file_path=r[col], arm=arm) for r in rows], indent=1))
    tc, notes = test_cases(rows)
    (DS / "test_cases.json").write_text(json.dumps(tc, indent=1)); (DS / "rows.json").write_text(json.dumps(rows, indent=1))
    summ = dict(rows=len(rows), hf_revision=(HFD / "REVISION").read_text().strip() if (HFD / "REVISION").exists() else None,
                touch="values: tactile/<id>.npz u = min(F / S, CAP) / CAP, CAP 0.11; S = median total force over frames with total > 1 N of the replay",
                distinct_prompts=len({r["text"] for r in rows}), base_prompt_rows=sum(r["text"] == BASE + PANEL for r in rows),
                distinct_controls={a: len({r[col] for r in rows}) for a, col in ARMS.items()}, fhn_equals_cds_rows=sum(r["ctrl_own"] == r["ctrl_cube"] for r in rows),
                S_N=dict(min=min(r["S"] for r in rows), max=max(r["S"] for r in rows)), active_frac_mean=float(np.mean([c["active_frac"] for c in checks.values()])),
                clip_frac_active_max=max(c["clip_frac_active"] for c in checks.values()), test_cases=len(tc), notes=notes, failed=bad)
    (DS / "check.json").write_text(json.dumps(dict(summary=summ, rows=checks), indent=1))
    print(json.dumps(summ, indent=1))
    if bad:
        raise SystemExit(f"checks failed for {bad}: see {DS / 'check.json'}")


if __name__ == "__main__":
    main()
