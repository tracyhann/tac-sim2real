# Dual-mode LoRA · one model that generates (G) and estimates (E) touch (started 2026-10-22, runs 10-23)

**Goal (user 10-22).** One Wan2.2-Fun-A14B-Control LoRA on the 1088 x 480 canvas (video 832 | touch panel 256) that
(G) generates video + paired touch from a sim control, and (E) estimates touch from any given video. The inpaint mask picks
the job: G = frame 0 known (with / without its touch, 50 / 50), E = all 81 video frames known, the panel unknown.
**Touch = RELATIVE (user 10-23), panel v3.** The panel draws F / S, S = the episode's median total force over its contact frames
(whole episode; stored in every row, never shown to the model); caps 0.11 S per taxel, 1.1 S per finger bar (< 0.5 % clip).
Panel v2 (4.2 N / 50 N, absolute) was used for one build and dropped before training.
**Data (code/ge_data.py; dataset/rows.json; automatic D68 alignment check on every row).**
- real: 50 left-hand-primary TouchScale episodes (code/select_eps.py, HF range fetch code/fetch_eps.py).
  train 40: actor_09 mug x14 (incl. twins edf79415 / 53780879), actor_10 cup x8, actor_05 spray bottle x9, actor_04 cup / egg x9.
  test 10: actor_01 x8 (held-out participant) + twins 55494490 / 4397541c.
- sim: 39 train rows = zero-shot A14B videos under sim controls (round 5: 7 views of edf / 538 / palm-press p8; round 6
  counterfactuals head + front3) + that replay's sim touch (Mac `scene_mug/r2s2v/ge_simtouch.py`, snapshot in code/mac/);
  test 2 = the held-out twins' head videos. Train 79 rows, test 12.
- controls: sim render (twins / sim rows) and Depth Anything V2 depth of the clip (code/ge_depth.py) for every row.
**Trainer.** `third_party/VideoX-Fun/scripts/wan2.2_fun/train_control_lora_ge.py` = a COPY of train_control_lora.py written by
code/make_trainer.py (+ code/ge_mask.py copied next to it): --ge_mask / --ge_modes, resume that skips used batches (exact
passes), saves every 15 steps + every epoch, keeps every 2nd epoch. Tested by part `prep` (code/test_ge_cpu.py, tiny model
code/make_tiny.py + code/verify_tiny.py).
**Runs (8 passes, rank 64 / alpha 32, lr 1e-4, both experts).** U = both jobs (P(G) 0.5), 632 steps; G = G only, 632; E = E only,
632; R = E only on the 40 real rows, 320 (what sim data adds to estimation).
**Eval (code/predict_ge.py -> code/eval_ge.py, code/compose_ge.py).** Held-out: E on every test clip (Depth Anything control;
twins also with the sim control); G on the twins (real frame 0 with / without its touch). Protocol after the 10-23 diagnostic
(../../2026-10-23/dual_mode_infer_diag): E without guidance (CFG 1), G at CFG 6 with the start canvas as reference image.
Baselines: R6 regressor (../../2026-10-23/dual_mode_loss_table), twin sim touch, real frame-0 touch repeated.
**Launch / scheduling.** `bash code/launch_ge.sh <part>` (pool_retry loop per part, shared card cap); `code/dispatch_ge.sh`
(priority), `code/cap_guard_ge.sh` (pauses my lowest-priority parts above the cap); `code/stop_ge.sh` stops a part incl. its
srun step. Parts: prep, U|G|E|R_high|low, ev_<RUN>_<epoch>, regr, regrsp, diag.
**Outputs.** `out/` -> outputs/R2S2V_mug/lora_ge (dataset/, <RUN>_<expert>/ checkpoints, out/ predictions, eval/ tables,
videos/); markers outputs/R2S2V_mug/done_exp_lora_ge_<part>; logs `logs/` -> logs/lora_ge + logs/r2s2v/pool_cfix_r2s2r_ge_*.log.
**Status (2026-10-23 16:30 CDT).** U, G_high, E_high trained; E_low / G_low / R running or queued; first U eval re-running
with the fixed protocol. Running notes: side-session notes `dual_mode_lora_notes.md`.
**Live code.** `code/` here IS the live code; `experiments/R2S2V_mug/lora_ge` is a symlink to it (old paths keep working).
