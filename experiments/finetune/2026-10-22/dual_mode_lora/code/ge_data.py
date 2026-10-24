"""Dataset for the dual-mode video + touch LoRA (Wan2.2-Fun-A14B-Control, canvas 1088 x 480 = video 832 x 480 | touch
panel 256 x 480, panel v3 = RELATIVE touch F / S (lora_ge/panel_v3.py; S per episode, stored in every row, never shown); user 2026-10-22: one model that generates video + paired touch from sim
(mode G) and estimates touch from any given video (mode E); the job is picked by the inpaint mask at inference).

Rows (one per video + touch pair; every stream resampled to the canvas times t0 + k / 16 s, k = 0..80, LESSONS D68):
  real  TouchScale head video + the left glove's touch (lora_ge/episodes.json, fetched by fetch_eps.py)
        video: the head frame nearest each canvas time; touch: N above each taxel's episode baseline (20th percentile),
        max over +-34 ms (scene_mug cliprep.core.touch); window = the twin's window for the four twin episodes, else
        1.0 s before the left glove's first sustained contact (total > max(5 N, 25 % of its 95th percentile) for 0.2 s)
        control: Depth Anything V2 depth of the clip itself (ctrl_da/, written by ge_depth.py); twin episodes also have
        the sim control (head view of views5: twin depth + shadowless sim hand / mug + coaster / mark)
  sim   zero-shot A14B videos generated under a sim control (round 5: 7 views of the edf79415 / 53780879 replays and the
        edf79415 palm-press p8; round 6: edf79415 counterfactuals, head + front3) + that replay's simulated touch
        (scene_mug r2s2v/ge_simtouch.py -> lora_ge/simtouch/); control: that sim control, alt = Depth Anything of the video
Split by episode: test = lora_ge/episodes.json split "test" (participant actor_01 entirely + the held-out twins
55494490 / 4397541c, whose zero-shot head videos are test-only sim rows). Every row may be used in both modes (modes "GE");
the trainer (train_control_lora_ge.py) intersects that with the run's modes.
Alignment check (automatic, the build stops on a failure; report in check.json):
  - every stream's sample times = the canvas times (head frames within half a head frame, glove within its window, sim
    touch arrays carry their times), every video / control has 81 frames at 16 fps;
  - each written real canvas is decoded back: frame k must match the source head frame chosen for canvas time k better
    than its neighbours, and not the frame a frame-index copy would give (the D68 bug: frames[:81] of a 30 fps video);
  - the touch panel decodes back to the touch array (finger totals r > 0.99);
  - twin real rows must equal the old builder's arrays (dataset_ho tactile gt; scene_mug cliprep touch()).
  envs/armA/bin/python ge_data.py   -> outputs/R2S2V_mug/lora_ge/dataset/"""
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); X = R / "experiments/R2S2V_mug"; LG = R / "outputs/R2S2V_mug/lora_ge"
DS = LG / "dataset"; GEN = R / "outputs/A_baseline/gen"; SIM = R / "outputs/B_sim2wm/sim"; HO = R / "outputs/R2S2V_mug/lora_tactile/dataset_ho"
sys.path.insert(0, str(X / "lora_ge")); sys.path.insert(1, str(X / "lora_tactile"))
from panel_v3 import BAR_MAX, F_MAX, PADS, PANEL_VERSION, PH, PW, canvas, decode, finger_totals  # noqa: E402  (v3: RELATIVE touch F / S)
CONTACT_N = 1.0                                              # contact frames for the episode scale S

N, FPS, W, H = 81, 16, 832, 480
PANEL = " On the right, a grey touch map of the left glove: each cell is one sensor, brighter is pressed harder; the bars below are the total force per finger."
VIEWS5 = ["head", "wrist", "left3", "right3", "top3", "front3", "bottom3"]
C6 = {"recorded": "motion/variants/base/sim", "claw": "posture/variants/claw/sim", "pinch2": "posture/variants/pinch2/sim",
      "rimpinch": "posture/variants/rimpinch/sim", "wrap": "posture/variants/wrap/sim", "grip030": "grip_scale/variants/s0.3_off/sim",
      "fric05": "object_props/variants/m0_f0.5_g1/sim", "water300": "object_props/variants/m300_f1_g1/sim", "t10": "motion/variants/t10.00/sim",
      "shake60": "motion/variants/shake_lat_60mm_4hz/sim"}


# ------------------------------------------------------------------ io
def read_video(p, idx=None, size=(W, H)):
    """frames (RGB uint8) of p resized to size; idx = the frame numbers to keep (sequential decode)"""
    c = av.open(str(p)); out = {}; want = None if idx is None else set(int(i) for i in idx); n = 0
    for k, f in enumerate(c.decode(video=0)):
        n += 1
        if want is None or k in want:
            a = f.to_ndarray(format="rgb24"); out[k] = a if size is None or a.shape[1::-1] == tuple(size) else cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    c.close()
    return (np.stack([out[k] for k in range(n)]) if idx is None else np.stack([out[int(i)] for i in idx])), n


def write_video(p, fr, crf=14):
    p.parent.mkdir(parents=True, exist_ok=True); final = p; p = p.with_name("." + p.stem + ".tmp.mp4"); o = av.open(str(p), "w"); st = o.add_stream("h264", rate=FPS)
    st.width, st.height, st.pix_fmt = fr[0].shape[1], fr[0].shape[0], "yuv420p"; st.options = {"crf": str(crf)}
    for f in fr:
        for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            o.mux(pk)
    for pk in st.encode():
        o.mux(pk)
    o.close(); p.replace(final)                                   # atomic: a reader sees the old or the new file, never half


def with_panel(frames):
    return [np.concatenate([f, np.zeros((PH, PW, 3), np.uint8)], 1) for f in frames]


# ------------------------------------------------------------------ real episodes
class Ep:
    def __init__(self, ep):
        self.dir = LG / "episodes" / ep; self.info = json.loads((self.dir / "json").read_text())
        rows = (self.dir / "rgb_head.csv").read_text().splitlines()[1:]; t = np.array([float(r.split(",")[1]) for r in rows])
        self.T0 = t[0]; self.th = t - t[0]

    def glove(self):
        g = json.loads((R / "data/TouchScale/metadata/glove_geometry_left.json").read_text())["regions"]
        assert [r["id"] for r in g] == [p[0] for p in PADS] and all((r["rows"], r["cols"]) == (p[2], p[3]) for r, p in zip(g, PADS)), "glove pad order != panel"
        z = np.load(self.dir / "left_hand_data.npz"); tg = z["timestamps"].astype(np.float64) - self.T0
        F = np.concatenate([z[f"tactile_{r['id']}"].astype(np.float32).reshape(len(tg), -1) for r in g], 1)
        return tg, F

    def touch(self, ts):
        """(len(ts), 880) N: above each taxel's episode baseline (20th pct), max over +-34 ms (cliprep.core.touch)"""
        tg, F = self.glove(); Rr = np.clip(F - np.percentile(F, 20, axis=0), 0, None)
        out = []
        for t in ts:
            i = np.nonzero((tg >= t - 0.034) & (tg <= t + 0.034))[0]
            out.append(Rr[i].max(0) if len(i) else Rr[np.argmin(np.abs(tg - t))])
        return np.stack(out).astype(np.float32), tg

    def window(self):
        """t0 of an 81-frame window: 1.0 s before the left glove's first sustained contact"""
        tg, F = self.glove(); tot = np.clip(F - np.percentile(F, 20, axis=0), 0, None).sum(1)
        k = max(1, int(round(0.1 * len(tg) / max(tg[-1] - tg[0], 1e-3)))); sm = np.convolve(tot, np.ones(k) / k, "same")
        thr = max(5.0, 0.25 * float(np.percentile(sm, 95))); on = sm > thr; hold = max(1, int(round(0.2 * len(tg) / max(tg[-1] - tg[0], 1e-3))))
        run = np.convolve(on.astype(float), np.ones(hold), "valid") >= hold; i = int(np.argmax(run)) if run.any() else int(np.argmax(sm))
        t_on = float(tg[i]); t0 = min(max(self.th[0], t_on - 1.0), self.th[-1] - (N - 1) / FPS - 0.05)
        return float(t0), t_on, thr


def touch_scale(T):
    """S (N) = median total force over the contact frames (total > 1 N) of the given frames (panel v3, user 2026-10-22)"""
    tot = T.sum(1); on = tot > CONTACT_N; return float(np.median(tot[on])) if on.any() else float("nan")


def build_real(r, twin_t0, rows, checks):
    e = Ep(r["episode_id"]); ep8 = r["ep8"]
    if r["twin"]:
        t0, t_on, thr = twin_t0[ep8], None, None
    else:
        t0, t_on, thr = e.window()
    ts = t0 + np.arange(N) / FPS; assert ts[-1] <= e.th[-1] + 1e-6, (ep8, "window past the end")
    hi = np.array([int(np.argmin(np.abs(e.th - t))) for t in ts]); dt = np.abs(e.th[hi] - ts)
    T, tg = e.touch(ts); src, nsrc = read_video(e.dir / "rgb_head.mp4")
    S = touch_scale(e.touch(e.th)[0]); Tr = T / S                               # the WHOLE episode's scale, shared by all its windows
    assert nsrc == len(e.th), (ep8, "head video frames != timestamps", nsrc, len(e.th))
    fr = [canvas(src[hi[k]], Tr[k]) for k in range(N)]; rid = f"real_{ep8}"; vid = DS / "video" / f"{rid}.mp4"; write_video(vid, fr)
    cv2.imwrite(str(DS / "start" / f"{rid}.png"), fr[0][..., ::-1]); np.savez_compressed(DS / "tactile" / f"{rid}.npz", touch=T, touch_rel=Tr, S=S, t=ts, head_idx=hi)
    # ---- checks: written frames vs source frames (D68), touch panel round trip, twin rows vs the old builder
    back, nb = read_video(vid, size=(W + PW, H)); assert nb == N, (rid, nb)
    v = back[:, :, :W].astype(np.float32); mae = lambda k, j: float(np.abs(v[k] - src[int(np.clip(j, 0, nsrc - 1))].astype(np.float32)).mean())
    own = np.array([mae(k, hi[k]) for k in range(N)]); nb_ = np.array([min(mae(k, hi[k] - 1), mae(k, hi[k] + 1)) for k in range(N)])
    naive = np.array([mae(k, hi[0] + k) for k in range(N)])                      # what a frame-index copy would have written
    moving = nb_ > own + 0.3                                                     # frames whose neighbours differ visibly
    c = dict(t0=round(t0, 3), contact_onset_s=t_on, thr_N=thr, max_dt_head_s=float(dt.max()), own_mae=float(own.mean()),
             own_beats_neighbours=float((own[moving] < nb_[moving]).mean()) if moving.any() else None, n_moving=int(moving.sum()),
             naive_mae_late=float(naive[40:].mean()), own_mae_late=float(own[40:].mean()))
    D = np.stack([decode(f[:, W:])[0] for f in back]); ft, fd = np.array([finger_totals(x) for x in np.minimum(Tr, F_MAX)]), np.array([finger_totals(x) for x in D])
    c["panel_finger_r"] = float(np.corrcoef(ft.ravel(), fd.ravel())[0, 1]) if ft.std() > 1e-3 else None
    c["S_N"] = S; c["touch_total_max_N"] = float(T.sum(1).max()); c["clip_rel_taxel"] = float((Tr[Tr > 0.006] > F_MAX).mean()) if (Tr > 0.006).any() else 0.0; c["contact_frac"] = float((T.sum(1) > 5).mean())
    c["glove_gap_max_s"] = float(max(np.min(np.abs(tg - t)) for t in ts))
    if r["twin"]:
        o = np.load(HO / "tactile" / f"tsmug-{ep8}.npz"); c["vs_dataset_ho_touch_maxdiff_N"] = float(np.abs(o["gt"] - T).max()); c["vs_dataset_ho_t"] = float(np.abs(o["t"] - ts).max())
        hv = HO / ("train" if r["split"] == "train" else "test") / f"tsmug-{ep8}_gt.mp4"
        if hv.exists():
            old, _ = read_video(hv, size=(W + PW, H)); c["vs_dataset_ho_video_mae"] = float(np.abs(old[:, :, :W].astype(np.float32) - v).mean())
    ok = (c["max_dt_head_s"] <= 0.5 / 30 + 1e-3 and c["n_moving"] >= 5 and c["own_beats_neighbours"] >= 0.9 and c["own_mae"] < 4.0
          and c["naive_mae_late"] > c["own_mae_late"] + 1.0 and (c["panel_finger_r"] is None or c["panel_finger_r"] > 0.99) and c["glove_gap_max_s"] < 0.05
          and (not r["twin"] or (c["vs_dataset_ho_touch_maxdiff_N"] < 0.05 and c["vs_dataset_ho_t"] < 1e-6 and c.get("vs_dataset_ho_video_mae", 0) < 4.0)))
    c["ok"] = bool(ok); checks[rid] = c
    print(rid, r["split"], r["actor"], "ok" if ok else "FAILED", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in c.items()}, flush=True)
    clip = f"tsmug-{ep8}" if r["twin"] else ep8
    text = (TWIN_TEXT[clip] if r["twin"] else f"First-person view of a person wearing beige knitted tactile gloves with black wrist straps at a desk: the left hand {r['phrase']}. Realistic video.{PANEL}")
    ctrl = f"ctrl_sim/{clip}_head.mp4" if r["twin"] else f"ctrl_da/{rid}.mp4"
    rows.append(dict(id=rid, file_path=f"video/{rid}.mp4", control_file_path=ctrl, control_alt_file_path=f"ctrl_da/{rid}.mp4" if r["twin"] else None,
                     text=text, type="video", width=W + PW, height=H, kind="real", split=r["split"], actor=r["actor"], episode=r["episode_id"], task=r["task"],
                     clip=clip, view="head", tactile=f"tactile/{rid}.npz", modes="GE", start=f"start/{rid}.png", panel=PANEL_VERSION, S=S))
    return ok


def sim_control(clip, vdir, view):
    """the sim control (854 x 480) resized to 832 x 480 + a black panel, cached per (clip, vdir, view)"""
    p = DS / "ctrl_sim" / f"{clip}_{'' if vdir == 'views5' else vdir + '_'}{view}.mp4"
    if not p.exists():
        fr, n = read_video(SIM / clip / vdir / view / "fun_ctrl.mp4"); assert n == N, (clip, vdir, view, n); write_video(p, with_panel(fr))
    return str(p.relative_to(DS))


def build_sim(name, clip, vdir, view, tag, seed, touch_name, split, rows, checks):
    g = GEN / f"wan22fun-a14b-{view}-{tag}" / clip / f"seed{seed}.mp4"; cj = SIM / clip / vdir / view / "cam.json"
    if not g.exists():
        return None
    meta = json.loads(g.with_suffix(".json").read_text()); cam = json.loads(cj.read_text()); ctl = SIM / clip / vdir / view / "fun_ctrl.mp4"
    st = np.load(LG / "simtouch" / f"{touch_name}.npz"); Tsim = st["sim"].astype(np.float32); ts = st["t"]
    S = touch_scale(Tsim); Tr = Tsim / S                                         # the replay spans the window: its own scale
    c = dict(frames=meta["frames"], fps=meta["fps"], control_in_meta=f"{vdir}/{view}/fun_ctrl.mp4" in meta["control"], cam_sim=cam.get("sim"),
             video_newer_than_control=g.stat().st_mtime > ctl.stat().st_mtime, touch_source=str(st["source"]) if "source" in st.files else "replay",
             touch_dt=float(np.abs(np.diff(ts) - 1 / FPS).max()))
    if vdir.startswith("views6_"):                                  # round 6: the control must show this variant's replay (bash 3.2 export bug)
        c["cam_sim_matches"] = str(cam.get("sim", "")).endswith(C6[vdir[len("views6_"):]])
    ok = c["frames"] == N and c["fps"] == FPS and c["control_in_meta"] and c["video_newer_than_control"] and c["touch_dt"] < 1e-6 and c.get("cam_sim_matches", True)
    rid = f"sim_{name}_{view}_s{seed}"; c["ok"] = bool(ok); checks[rid] = c
    print(rid, split, "ok" if ok else "SKIPPED (check failed)", c, flush=True)
    if not ok:
        return False
    fr, n = read_video(g); assert n == N
    cf = [canvas(f, Tr[k]) for k, f in enumerate(fr)]; write_video(DS / "video" / f"{rid}.mp4", cf); cv2.imwrite(str(DS / "start" / f"{rid}.png"), cf[0][..., ::-1])
    np.savez_compressed(DS / "tactile" / f"{rid}.npz", touch=Tsim, touch_rel=Tr, S=S, t=ts)
    rows.append(dict(id=rid, file_path=f"video/{rid}.mp4", control_file_path=sim_control(clip, vdir, view), control_alt_file_path=f"ctrl_da/{rid}.mp4",
                     text=meta["prompt"] + PANEL, type="video", width=W + PW, height=H, kind="sim", split=split, actor="sim", episode=clip, task=name,
                     clip=clip, view=view, tactile=f"tactile/{rid}.npz", modes="GE", start=f"start/{rid}.png", gen=str(g.relative_to(R)), panel=PANEL_VERSION, S=S))
    return True


TWIN_TEXT = {}


def main():
    for d in ("video", "start", "tactile", "ctrl_sim", "ctrl_da"):
        (DS / d).mkdir(parents=True, exist_ok=True)
    for m in ("metadata_mixed.json", "metadata_test.json"):
        for x in json.loads((HO / m).read_text()):
            TWIN_TEXT.setdefault(x["clip"], x["text"])
    eps = json.loads((LG / "episodes.json").read_text()); rows, checks = [], {}
    twin_t0 = {k[6:]: float(np.load(LG / "simtouch" / f"{k}.npz")["t"][0]) for k in TWIN_TEXT}
    for r in eps:
        if r["twin"]:
            sim_control(f"tsmug-{r['ep8']}", "views5", "head")
        build_real(r, twin_t0, rows, checks)
    for ep8, split in (("edf79415", "train"), ("53780879", "train"), ("55494490", "test"), ("4397541c", "test")):
        clip = f"tsmug-{ep8}"
        for view in (VIEWS5 if split == "train" else ["head"]):
            for seed in ((0, 1) if view == "head" and split == "train" else (0,)):
                build_sim(ep8, clip, "views5", view, "c5", seed, clip, split, rows, checks)
    for view in VIEWS5:
        build_sim("edf79415p8", "tsmug-edf79415", "views5_p8", view, "c5p8", 0, "edf-p8", "train", rows, checks)
    for n in C6:
        if (LG / "simtouch" / f"edf-c6{n}.npz").exists():
            for view in ("head", "front3"):
                build_sim(f"edf79415c6{n}", "tsmug-edf79415", f"views6_{n}", view, f"c6{n}", 0, f"edf-c6{n}", "train", rows, checks)
    bad = [k for k, c in checks.items() if k.startswith("real_") and not c["ok"]]
    from panel_v3 import encode
    lay = encode(np.zeros(880)) > 0; full = np.zeros((H, W + PW), bool); full[:, W:] = lay            # pads + bars, at the latent grid (8 px)
    lat = full.reshape(H // 8, 8, (W + PW) // 8, 8).any((1, 3)); np.save(DS / "panel_layout_latent.npy", lat)
    print("panel taxel / bar latent cells", int(lat.sum()), f"= {lat.mean():.1%} of latent positions")
    (DS / "panel.json").write_text(json.dumps(dict(version=PANEL_VERSION, relative=True, F_MAX_rel=F_MAX, BAR_MAX_rel=BAR_MAX, module="lora_ge/panel_v3.py",
                                                   scale="S = median total force over contact frames (total > 1 N) of the whole episode (real) / replay (sim)"), indent=1))
    (DS / "check.json").write_text(json.dumps(checks, indent=1)); (DS / "rows.json").write_text(json.dumps(rows, indent=1))
    tr = [x for x in rows if x["split"] == "train"]; te = [x for x in rows if x["split"] == "test"]
    assert not {x["episode"] for x in tr} & {x["episode"] for x in te}, "an episode is in both splits"
    (DS / "metadata_train.json").write_text(json.dumps(tr, indent=1)); (DS / "metadata_train_real.json").write_text(json.dumps([x for x in tr if x["kind"] == "real"], indent=1))
    (DS / "metadata_test.json").write_text(json.dumps(te, indent=1))
    print("rows", len(rows), "train", len(tr), f"(real {sum(x['kind'] == 'real' for x in tr)}, sim {sum(x['kind'] == 'sim' for x in tr)})",
          "test", len(te), f"(real {sum(x['kind'] == 'real' for x in te)}, sim {sum(x['kind'] == 'sim' for x in te)})", "real checks failed:", bad)
    if bad:
        raise SystemExit(f"alignment check failed for {bad}: see {DS / 'check.json'}")


if __name__ == "__main__":
    main()
