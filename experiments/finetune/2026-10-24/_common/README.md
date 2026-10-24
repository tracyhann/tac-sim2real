# HF25 shared code · FHN / SVO / CDS (2026-10-24)

Shared data, training, inference and scoring code for the three video + touch LoRAs of 2026-10-24
([FHN](../FHN/README.md), [SVO](../SVO/README.md), [CDS](../CDS/README.md)). The arms differ ONLY in the control video;
data, prompts, normalisation, touch head, trainer settings, steps, test cases and metrics are identical.

**Status: code ready, dataset built, trainer smoke-tested on a GPU; the real runs are not launched.** Runbook for whoever
runs them (downloads, layout, dataset build, launch, monitoring, eval): [../README.md](../README.md).

## Common setup (user 2026-10-24)
- **Model.** Wan2.2-Fun-A14B-Control, LoRA on both experts (high / low noise), as the F / H runs.
- **Data.** The 25 pairs of the public release https://huggingface.co/datasets/tracyhan816/mug-sim2video-pose-tactile
  (commit 6cf129c; `fetch_hf.sh` -> Vision `data/mug-sim2video-pose-tactile/`): zero-shot A14B video (832 x 480, 81 frames)
  of a counterfactual replay of clip edf79415 + that replay's simulated touch of the left glove (880 channels per frame).
  Groups: r6 head (6), r8b head + top photo (12), r10 user frames (7). Variants: recorded, grip 30 %, friction x0.5,
  +300 g water, 10x fast, +-60 mm shake, claw, rim pinch.
- **Touch = values, not pixels (user 10-24).** The F / H runs drew the touch as a grey panel beside the video and read it back
  from pixels; that is gone. Here the video is the plain 832 x 480 clip and the touch is an (81, 880) array. A light head
  (`hf25_patch.TactileHead`, ~13 M params; UVA / UVTA-style decoupled head) reads the low-noise expert's last-layer video
  tokens: per latent frame one attention pooling over its tokens (+ the known frame-0 touch) -> temporal conv -> 880 taxels
  of each frame. It is trained together with the LoRA (loss = video flow matching + 1.0 x touch loss, the touch gradient
  flows into the LoRA); at inference it reads the tokens of the last denoising step, so the touch comes out of the same
  forward pass as the generated video. The high-noise expert is a plain video LoRA.
- **Normalisation (new vs F / H).** F / S with S = median total force over frames with total > 1 N of the row's replay (81
  frames = the whole 5 s window); head target u = min(F / S, 0.11) / 0.11 (the loss-table target). F / H used absolute N.
- **Touch loss.** R1 of the loss table (Huber + 1/2 L1, active taxels x10; TouchSight) — the best contact F1 there.
- **Prompts** (`prompts.py`): one shared action sentence + one visual-feature sentence where the video has one (water in the
  mug, claw grip, rim pinch, shake, very fast) + the touch sentence. 9 rows (recorded / grip 30 % / friction x0.5) get the
  shared prompt alone. Scene, view and hand look are not described: the first frame carries them.
- **Training** (`train.sh`, = F / H `run_lora.sh`): rank 64 / alpha 32, lr 1e-4 constant, batch 1, first-frame-only inpaint
  mask (frame 0 known), control_ref + first_frame, 500 steps per expert (= 20 passes per row), checkpoint every 50 (multiples
  of 100 kept, head saved with each). ~45 s/step on an H200 -> ~6.5 h per expert; a pool allocation (4 h) ends mid-run and
  the retry resumes from the newest LoRA weights + head with a fresh Adam (as F / H). 6 cards in parallel.

## Controls (`build_hf25.py`, one dataset, `metadata_<ARM>.json` differ only in `control_file_path`)
| arm | control | source per group |
|---|---|---|
| FHN | the control the video was generated from (F / H format: scene depth + shadowless sim hand / mug / coaster pixels) | r6: views6 (fused-twin depth); r8b: views9 (G1 cubes / sim desk box); r10: views10 (cubes from the frame) |
| SVO | sim render only: hand / mug / coaster pixels on black, no depth | views9 / views10 `sim.mp4`; r6 rows take views9 of the same variant (same replay + head camera; views6 `sim.mp4` also draws the textured twin) |
| CDS | cube-scene depth (no body) + the sim render on top | views9 / views10 `fun_ctrl.mp4`; r6 rows -> views9 |

Note: on the 19 r8b / r10 rows CDS = FHN (those videos were generated from cube controls); FHN and CDS differ on the 6 r6 rows
and on the held-out twins' control.

## Evaluation (`predict_hf25.py`, `regr_hf25.py`, `score_hf25.py`; part `ev_<ARM>_500`)
- Held-out: twins 55494490 / 4397541c (never in any HF25 row), head view; start = real frame 0 + its touch (target real touch,
  relative to the episode S) and start = zero-shot generated frame 0 + sim touch (target sim touch). Controls in each arm's
  own format: FHN views5/head (twin depth + sim), SVO / CDS views9_recorded/head (built by `mac/twin_cubes.sh`, 10-24).
- In-sample: 3 training rows (r6_head_recorded, r8b_topphoto_shake60mm, r10_closeup_rimpinch).
- Generation = dual-mode G protocol: CFG 6, start frame as reference image, 50 steps, seed 0; touch from the head at the last step.
- Metrics = `dual_mode_lora/code/eval_ge.metrics` (full set) + mug / hand IoU vs sim (labeladh.py). Reference rows: R6 regressor
  trained on the same 25 pairs applied to each arm's generated video (`regr_hf25.py`: "video + external estimator" vs "own
  head"), frame-0 touch repeated, twin sim touch, dual-mode G epoch 8, h2 step 600 (absolute, r only), zero-shot / real video IoU.

## Files
Runbook for whoever runs it: [../README.md](../README.md) (downloads, layout, dataset build, launch, monitoring, eval).
`fetch_hf.sh` · `build_hf25.py` (+ checks -> dataset/check.json) · `prompts.py` · `hf25_patch.py` (head, loss, dataset) ·
`make_trainer_hf25.py` (-> VideoX-Fun `train_control_lora_hf25.py`) · `train.sh` (parts) · `launch.sh` (pool loops, eval waiter) ·
`predict_hf25.py` · `regr_hf25.py` · `score_hf25.py` · `videoxfun_patches/` (the two shared-trainer patches a fresh VideoX-Fun needs first) · `smoke_tiny.sh` / `launch_smoke.sh`
(tiny-model trainer test, one card, ~3 min; PASSED 10-24 after a dtype fix) · `release_cases.json` · `mac/twin_cubes.sh`.
Outputs: Vision `outputs/R2S2V_mug/hf25/{dataset, <ARM>_high, <ARM>_low, out/<ARM>c<CK>[_regr], regr, eval}`; logs `logs/hf25/`.

## History
- 10-24 v1 drew the touch as a pixel panel on the video canvas (copied from F / H); rejected by the user ("touch is values, not
  a video / RGB problem") after the dataset had been built. v2 = this value head; the panel dataset was deleted and rebuilt.
