# Loss comparison R0-R6 on the E regression head (2026-10-23)

**Goal (user via the parent, Methods tab sections 5-6).** Which loss to carry into the diffusion LoRA for relative touch.
**Setup.** Frozen Wan VAE latents of the 832 x 480 video -> the same 3D-conv + temporal trunk -> 880 taxels per frame, panel v3
relative touch; per-taxel losses on u = (F/S)/0.11. Same 79 training rows, optimiser (AdamW 3e-4), 150 epochs, batch 4.
The extra-term weight lam is chosen per variant on an inner split by participant (actor_10 held out of the TRAINING rows);
then seeds 0 / 1 / 2 train on all training rows and are scored on the held-out cases (dual_mode_lora/code/eval_ge.metrics).
R0 L1 · R1 Huber + 1/2 L1, active x10 · R2 + Pearson (finger curves, frame maps) · R3 + temporal (VDA TGM) and spatial
gradients · R4 BCE contact + gated L1 · R5 16 log bins + bin-distance CE · R6 total L1 + Sinkhorn OT on the load share.
Sensitivity (part regrsp): R0 / R3 on a softplus output head.
**Results (eval/loss_table.md, loss_table_sp.md).** R0 collapses to all-zero output with both output heads (sparse targets);
R3 collapses or sits at a constant level. R6 wins the inner split and leads on total r, relative MAE, load-share error and
N error (actor_01: total r 0.79, rel. MAE 0.23, share TV 0.26, 12.8 N); R1 has the best contact F1@6mm on the twins' real
video (0.65). Recommendation: R6 (+ R1's active weighting) as the auxiliary touch loss for the diffusion LoRA (D2).
**Code.** `code/` = loss_table.py, regress_ge.py (latents), smoke_lt.py (snapshots; the live copies are in
../../2026-10-22/dual_mode_lora/code, run as parts `regr` / `regrsp` of run_ge.sh).
**Outputs.** `out/` -> outputs/R2S2V_mug/lora_ge/{eval/loss_table*.{md,json}, out/lt_<V>_s<k>/, out/regr_*/, regr/lat/}.
