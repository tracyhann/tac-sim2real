# FHN · F / H video + touch LoRA, normalised touch, HF25 data (2026-10-24)

**Goal (user 10-24).** The F / H arms were trained on absolute touch drawn as a pixel panel. Redo them on the 25 HF pairs with
normalised touch (F / S per replay) output as values by a touch head; everything else as F / H.
**Control.** The control each video was generated from (F / H format): static scene depth (inverse-depth grey) + shadowless
sim hand / mug / coaster pixels. r6 rows: fused-twin depth (views6); r8b: G1 cubes / sim desk box (views9); r10: cubes from
the user frame (views10). Held-out twins: fused-scan depth + sim (the control the F / H runs were tested with; `test/` of the dataset).
**Touch.** An (81, 880) value array; a head on the low-noise expert's tokens outputs it (no pixel panel; see shared README).
**Data, prompts, training, eval.** Shared: [../_common/README.md](../_common/README.md). 500 steps per expert.
**Run.** `bash ../_common/code/train.sh FHN_high` and `FHN_low` (one GPU each, rerun until exit 0), then `ev_FHN_500`; see the
[runbook](../README.md). On the authors' cluster: `bash code/launch.sh` ([CLUSTER.md](../_common/CLUSTER.md)).
**Outputs.** `$R2S2R/outputs/R2S2V_mug/hf25/{FHN_high, FHN_low (+ tactile_head-*.pt), out/FHNc500, out/FHNc500_regr}`; table `hf25/eval/table.md`.
**Status.** Code ready, not launched; runbook for the person running it: [../README.md](../README.md).
