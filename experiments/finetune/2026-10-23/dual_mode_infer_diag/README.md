# Inference diagnostic: why the first U eval broke (2026-10-23)

**Symptom.** U epoch 8, CFG 6: in E the GIVEN video came back colour-distorted (known-part MAE 35-55 grey levels) and the panel
inflated (rel. MAE 1.8); in G the video drifted off the sim control (mug / hand IoU 0.08 / 0.06-0.09).
**Checks.** The pipeline matches training: 81-frame masks, mask polarity and 4-frame packing, y = [control | mask | masked
video]; known latents are conditioning only (no paste-back for the 8x VAE).
**Runs (part `diag` of run_ge.sh, predict_ge.py flags).** E on training rows at CFG 6 / CFG 1; E on test rows at CFG 1, CFG 6 with
an empty negative prompt, and the base model without the LoRA at CFG 1; G with the start canvas as reference image, and at CFG 1.
**Result.** CFG is the cause: known-part MAE CFG 6 = 22-39 even on training rows (empty negative 21-24), CFG 1 = 2-3 (base 4-5);
CFG 1 also fixes the panel scale (rel. MAE 1.50 -> 0.31). G: CFG 6 without a reference image invents objects; CFG 1 follows the
motion but blurs; CFG 6 + start as reference image is sharp. -> protocol: E at CFG 1 without reference image (it would leak the
frame-0 touch), G at CFG 6 with the reference image. Even at CFG 1 U's touch estimate is weak, also in-sample (total r 0.07-0.23 on
training rows): the panel is under-learned by the plain latent MSE (-> D1 panel loss weight).
**Outputs.** `out/` -> outputs/R2S2V_mug/lora_ge/out/{dg*, Up8_cfg6noref}; the old labeladh scores in labeladh/<clip>/old_cfg6noref/.
