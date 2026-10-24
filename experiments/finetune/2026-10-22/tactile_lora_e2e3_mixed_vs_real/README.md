# e2 / e3 · video + touch LoRA, mixed vs real-only (2026-10-22)

**Goal.** e2 = the 6 canvases of e1 with one untagged prompt (real + sim rows mixed); e3 = only the 2 real rows (what the
sim-generated rows add). Panel v1 (absolute N).
**Fixes made here.** `--inpaint_first_frame_only` (patch_trainer.py: no VideoX-Fun random masks, which leaked the target);
`--resume_lora_weights` (patch_trainer2.py: resume from checkpoint-N.safetensors with a fresh optimizer). Both patch the
shared `third_party/VideoX-Fun/scripts/wan2.2_fun/train_control_lora.py` (opt-in flags).
**Training.** rank 64 / alpha 32, lr 1e-4, 600 steps (e3 step 200 = the same 100 passes as e2 step 600).
**Known bug (D68).** Real rows took the first 81 frames of the 30 fps real.mp4 (0-2.7 s, 1.9x slow) under a 5 s control and
touch panel -> retrained as f2 / f3.
**Evals.** e2_eval / e3_eval (steps 300 / 600; e3 200 / 300 / 600), step-200 board (`run_ck200.sh`), e2_ev400, e3_ev*;
contact F1 at 6 mm + per pad by `multiscale_f1.py` (snapshots in `code/`).
**Code.** `code/` = snapshots (2026-10-23); live code `experiments/R2S2V_mug/lora_tactile/` + `run_ck200.sh`, `multiscale_f1.py`.
**Outputs.** `out/` -> outputs/R2S2V_mug/lora_tactile/{dataset, e2_*, e3_*, out (decoded tactile + canvases), loss_curves.json,
multiscale_f1.json}.
**Status.** Done; results are in-sample and on misaligned real rows (D68): use f2 / f3 / h2 / h3 instead.
