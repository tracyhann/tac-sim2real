# h2 / h3 · held-out test of the video + touch LoRA (2026-10-22)

**Goal.** The first held-out test: train on edf79415 + 53780879 (time-aligned real rows, D68 fix), score on 55494490 +
4397541c (never trained on). h2 = mixed real + sim rows (6), h3 = real rows only (2). Control = the star control
(views5/head/fun_ctrl.mp4: episode-fused depth + shadowless sim hand / mug + coaster / mark).
**Training.** rank 64 / alpha 32, lr 1e-4; h2 600 steps, h3 200 steps (= h2 step 600 in passes per sample).
**Evals.** `h2_ev<ck>` / `h3_ev<ck>` (after_ck.sh), `h2_ev_direct.sbatch` / `h2_ev400_direct.sbatch` when the pool starved them;
table `ho_table.py` -> heldout_table.json; grids Mac `scene_mug/r2s2v/lora_grid.py`.
**Results (heldout_table.json, real frame-0 start, absolute N, panel v1).** contact F1 @ 6 mm: h2 step 600 = 0.59
(step 400: 0.65), h3 step 200 = 0.66; baselines: real frame-0 touch repeated 0.64, twin sim touch 0.38 (total r 0.91).
Video: zero-shot A14B mug / hand IoU vs sim 0.81 / 0.36; h2 / h3 lower (see table).
**Code.** `code/` = snapshots (2026-10-23, incl. run_lora.sh backups); live code `experiments/R2S2V_mug/lora_tactile/`
(still used by the running `h3_ev400`).
**Outputs.** `out/` -> outputs/R2S2V_mug/lora_tactile/{dataset_ho, h2_*, h3_*, heldout_table.json}.
