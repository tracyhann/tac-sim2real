# f2 / f3 · e2 / e3 retrained on time-aligned real rows (2026-10-22)

**Goal.** Same as e2 (mixed, 6 rows) / e3 (real only, 2 rows) with ONLY the D68 fix: `dataset_fix` = dataset rebuilt by
`lora_tactile_data.py` with real rows resampled to the canvas times (control, sim rows, touch, prompts identical).
**Training.** rank 64 / alpha 32, lr 1e-4; f2 600 steps, f3 to 600 (step 200 = 100 passes, matched to f2 step 600);
ran under the temporary 8-card raise (user, 10-22 ~14:35).
**Evals.** in-sample `f2_ev*` / `f3_ev*`; held-out `f2_hoev*` / `f3_hoev*` on 55494490 / 4397541c (`after_ck_fho.sh`).
Board: Mac `scene_mug/r2s2v/lora_fix_pull.sh` -> `lora_fix_grid.py` (snapshots in `code/mac/`).
**Results (held-out, outputs/R2S2V_mug/lora_tactile/heldout_table.json, real frame-0 start, absolute N, panel v1).**
contact F1 @ 6 mm: f2 step 600 = 0.66, f3 step 200 = 0.65 (baseline: real frame-0 touch repeated 0.64; twin sim touch 0.38).
**Code.** `code/` = snapshots (2026-10-23); live code `experiments/R2S2V_mug/lora_tactile/`.
**Outputs.** `out/` -> outputs/R2S2V_mug/lora_tactile/{dataset_fix, f2_*, f3_*}; decoded predictions in lora_tactile/out/.
