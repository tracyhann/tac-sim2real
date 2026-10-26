"""AUTHORS' BUILD SCRIPT (needs the replays): the touch-data comparison sets (README section 11; user 2026-10-26: "每组挑 100 个，
凑一个 300 个的数据集，训这 3 个"). Three policy arms, same model / steps / inputs, only the action labels' source differs:
  T-obs   dataset_cmp_t, arm obs   the observed hand pose as the action (what a video of a human hand gives)
  T-cmd   dataset_cmp_t, arm cmd   the commanded hand pose of the TOUCH-calibrated replay (servo targets that make the simulated grip
                                    follow the glove)
  V-cmd   dataset_cmp_v, arm cmd   the commanded hand pose of the VISION-only replay of the same perturbations (clip config
                                    touch: false: fixed closing offset, release when the tracked mug stops)
Sets:
  test           36 touch-on rows in neither training set: 6 per batch x clip (jitter, jitter_fg, relocate x edf, 538); the fg ones are
                 the hardest (most water / lowest friction / lightest grip), the others a seeded draw
  dataset_cmp_t  300 touch-on rows: 50 per batch x clip, seeded draw from the rest; relocate has only 44 per clip left after the test,
                 so jitter tops up to 300 (56 per clip) (+ the 36 test rows, which train_vtla leaves out via --val_ids)
  dataset_cmp_v  300 valid vision-only rows with the composition the touch-off yield allows (valid: edf jitter 150, fg 86,
                 relocate 50; 538 jitter 0, fg 67, relocate 0 - the failures are the mug tipping at the vision-timed release):
                 every valid jitter and relocate twin whose touch-on row is not a test row, then fg (alternating clips) up to 300
Rows reference dataset_v3's generated videos by absolute path; a vision-only row uses its touch-on twin's video (same scene, same
wrist path; the fingers differ by a few degrees). Labels per row: vtla/<id>.npz from build_vtla_labels.build_row (state = that
rollout's measured hand pose; act_obs; act_cmd with q_cmd = the rollout's logged servo targets, cf_log ctrl_f).
  R2S2R=<root> envs/scenemug/bin/python build_cmp.py"""
import json
import os
import shutil
from pathlib import Path

import numpy as np

import build_vtla_labels as B

R = Path(os.environ.get("R2S2R", ".")).resolve(); H = R / "outputs/R2S2V_mug/hf25"; V3 = H / "dataset_v3"; S = R / "experiments/scene_mug_dg"
BATCH = {"jitter": "dg_jit", "jitter_fg": "dg_fg", "relocate": "dg_rl"}; N_TEST, N_TRAIN = 6, 50


def absrow(r, base, **kw):
    out = dict(r)
    for k in ("video", "file_path", "hand_pose", "tactile", "start"):
        if out.get(k) and not Path(out[k]).is_absolute():
            out[k] = str(base / out[k])
    out["video"] = out.get("video") or out["file_path"]; out.update(kw)
    return out


def hardness(r):
    p = r["params"]; return (p.get("water_g", 0) + 50) / p.get("friction_scale", 1.0) / ((p.get("fingers") or {}).get("hold_scale", 1.0))


def write_set(ds, rows):
    if ds.exists():
        shutil.rmtree(ds / "vtla", ignore_errors=True)
    (ds / "vtla").mkdir(parents=True, exist_ok=True); checks = {}
    for r in rows:
        checks[r["id"]] = B.build_row(ds, r, {})
    (ds / "rows.json").write_text(json.dumps(rows, indent=1)); (ds / "metadata_vtla.json").write_text(json.dumps(dict(rows=rows), indent=1))
    (ds / "vtla" / "check.json").write_text(json.dumps(checks, indent=1))
    src = {k: sum(1 for c in checks.values() if c.get("q_cmd_source", "").startswith(k)) for k in ("cf_log", "formula")}
    bad = [i for i, c in checks.items() if not c["ok"]]
    print(f"{ds.name}: {len(rows)} rows, q_cmd from {src}, label checks failed: {bad[:5]}")
    return checks


def main():
    rng = np.random.default_rng(20261026)
    v3 = {r["id"]: r for r in json.loads((V3 / "rows.json").read_text())}
    pool = {(k, c): sorted([r for r in v3.values() if r.get("group") == g and r["id"].startswith(c + "_")], key=lambda r: r["id"]) for k, g in BATCH.items() for c in ("edf", "538")}
    test, train_t, left = [], [], {}
    for (k, c), rows in pool.items():
        cand = sorted(rows, key=hardness, reverse=True)[:N_TEST] if k == "jitter_fg" else list(rng.choice(rows, N_TEST, replace=False))
        tid = {r["id"] for r in cand}; test += cand
        rest = [r for r in rows if r["id"] not in tid]; take = list(rng.choice(rest, min(N_TRAIN, len(rest)), replace=False)); train_t += take
        left[(k, c)] = [r for r in rest if r["id"] not in {x["id"] for x in take}]
    while len(train_t) < 6 * N_TRAIN:                                           # relocate has 50 rows per clip: 44 after the test -> top up from jitter
        for c in ("edf", "538"):
            if len(train_t) < 6 * N_TRAIN and left[("jitter", c)]:
                train_t.append(left[("jitter", c)].pop(0))
    test_ids = {r["id"] for r in test}
    vs = [r for r in json.loads((S / "runs/datagen/manifest_vsim.json").read_text())["rows"] if r["accepted"] and r["pair"] in v3 and r["pair"] not in test_ids]
    pick = [r for r in vs if r["kind"] in ("jitter", "relocate")]
    fg = {c: [r for r in vs if r["kind"] == "jitter_fg" and r["clip"] == c] for c in ("edf", "538")}
    for c in fg:
        rng.shuffle(fg[c])
    while len(pick) < 300 and (fg["edf"] or fg["538"]):
        for c in ("538", "edf"):
            if fg[c] and len(pick) < 300:
                pick.append(fg[c].pop())
    pick = pick[:300]
    rows_t = [absrow(r, V3, comparison="T") for r in train_t + test]
    rows_v = []
    for r in pick:
        tw = v3[r["pair"]]; d = S / "runs/datagen" / r["id"]
        rows_v.append(dict(id=r["id"], pair=r["pair"], video=str(V3 / tw["file_path"]), hand_pose=str(d / "hand_pose.npz"), tactile=str(d / "tactile.npz"), sim=r["sim"],
                           kind=r["kind"], variant=r["variant"], group=BATCH[r["kind"]], params=r["params"], text=r["text"], text_priv=r.get("text_priv"), comparison="V", touch="off"))
    comp = lambda rows: {f"{k}/{c}": sum(1 for r in rows if r["kind"] == k and r["id"].startswith(c + "_")) for k in BATCH for c in ("edf", "538")}   # noqa: E731
    write_set(H / "dataset_cmp_t", rows_t); write_set(H / "dataset_cmp_v", rows_v)
    info = dict(test=sorted(test_ids), train_t=comp(train_t), train_v=comp(rows_v), test_comp=comp(test), note=__doc__.split("Sets:")[0].strip())
    (H / "dataset_cmp_t" / "test.json").write_text(json.dumps(info, indent=1)); (H / "dataset_cmp_t" / "test_ids.txt").write_text("\n".join(sorted(test_ids)) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "note" and k != "test"}))


if __name__ == "__main__":
    main()
