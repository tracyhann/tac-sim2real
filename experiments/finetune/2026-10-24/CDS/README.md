# CDS · cube-scene depth + sim video as control, normalised touch, HF25 data (2026-10-24)

**Goal (user 10-24).** As FHN, first frame as input, but the control = depth of a CUBE scene (static boxes only, no body) +
the simulation's rendered animation on top.
**Control.** Round-8b format: G1 primitive boxes (head) / the sim desk box (top photo) / cubes from the user frame (r10) as
inverse-depth grey, sim hand / mug / coaster pixels pasted on top (`fun_ctrl.mp4` of views9 / views10; r6 rows -> views9 of
the same variant). Held-out twins: their cube-scene depth + sim render (`test/` of the dataset).
**Overlap with FHN.** The 19 r8b / r10 videos were generated from these cube controls, so their CDS control = their FHN
control; the arms differ on the 6 r6 rows (twin depth in FHN, cubes here) and on the held-out twins' control.
**Touch.** An (81, 880) value array; a head on the low-noise expert's tokens outputs it (no pixel panel; see shared README).
**Data, prompts, training, eval.** Shared: [../_common/README.md](../_common/README.md). 500 steps per expert.
**Run.** `bash ../_common/code/train.sh CDS_high` and `CDS_low` (one GPU each, rerun until exit 0), then `ev_CDS_500`; see the
[runbook](../README.md). On the authors' cluster: `bash code/launch.sh` ([CLUSTER.md](../_common/CLUSTER.md)).
**Outputs.** `$R2S2R/outputs/R2S2V_mug/hf25/{CDS_high, CDS_low (+ tactile_head-*.pt), out/CDSc500, out/CDSc500_regr}`; table `hf25/eval/table.md`.
**Status.** Code ready, not launched; runbook for the person running it: [../README.md](../README.md).
