# SVO · sim video only as control, normalised touch, HF25 data (2026-10-24)

**Goal (user 10-24).** As FHN (normalised touch, 880-channel panel, 25 HF pairs, video + touch out), but the control is the
simulation's rendered animation ONLY, no depth.
**Control.** The sim render: hand / mug / coaster as shadowless MuJoCo pixels on black (`sim.mp4` of views9 / views10; the r6
rows use views9 of the same variant = same replay and head camera, because views6's `sim.mp4` also draws the textured twin
mesh). ~97 % of each control frame is black. Held-out twins: their cube-scene sim render (`test/` of the dataset).
**Touch.** An (81, 880) value array; a head on the low-noise expert's tokens outputs it (no pixel panel; see shared README).
**Data, prompts, training, eval.** Shared: [../_common/README.md](../_common/README.md). 500 steps per expert.
**Run.** `bash ../_common/code/train.sh SVO_high` and `SVO_low` (one GPU each, rerun until exit 0), then `ev_SVO_500`; see the
[runbook](../README.md). On the authors' cluster: `bash code/launch.sh` ([CLUSTER.md](../_common/CLUSTER.md)).
**Outputs.** `$R2S2R/outputs/R2S2V_mug/hf25/{SVO_high, SVO_low (+ tactile_head-*.pt), out/SVOc500, out/SVOc500_regr}`; table `hf25/eval/table.md`.
**Status.** Code ready, not launched; runbook for the person running it: [../README.md](../README.md).
