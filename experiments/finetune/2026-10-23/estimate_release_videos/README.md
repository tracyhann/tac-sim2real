# Touch estimation on the released generated videos (2026-10-23)

**Goal (user 10-23).** "Submit a job on Vision: let our fine-tuned world model infer the tactile of these videos, try it", then
"you may also try our earlier video + paired tactile model; prefer the checkpoints we already have". The videos are the 27 of the
public release https://huggingface.co/datasets/tracyhan816/mug-sim2video-pose-tactile; their simulated tactile (880 glove
channels) is the reference.

**Models (existing checkpoints, nothing trained).**
- `U8`: dual-mode LoRA run U, epoch 8 (outputs/R2S2V_mug/lora_ge/U_{low,high}/checkpoint-632), job E exactly as its held-out
  eval: all 81 video frames known, touch panel unknown, CFG 1, no reference image, 30 steps. Panel v3 = relative touch (F / S).
- `f2`: video + touch LoRA f2 step 600 (outputs/R2S2V_mug/lora_tactile/f2_{low,high}/checkpoint-600; held-out contact F1 @ 6 mm
  0.66). Trained to generate video + touch from frame 0; here given the same full-video mask as U8 (never seen in training),
  same settings. Panel v1 = touch in N.
- Both: control = each video's own sim control (832 x 480 + black panel), prompt = its generation prompt + the panel sentence.

**Split.** r10 (9) and r8b (12) are new to both models; the 6 r6 head videos are U's training rows (sim rows of lora_ge).

**Steps.** Mac: cases.json from the release bundle. Vision: `bash code/launch_est.sh est_<model>_<group>` (pool_retry, shared cap)
-> `code/run_est.sh` -> `code/predict_release.py`. Mac: compare with the sim tactile (total-force correlation, contact timing,
per-finger shares; U8 on the relative scale, f2 in N).

**Outputs.** `out/` -> outputs/R2S2V_mug/release_estimate/<model>/<id>{.mp4,_tactile.npy,.json}; markers
outputs/R2S2V_mug/done_exp_est_<model>_<group>; logs logs/r2s2v/retry_exp_est_*.log and pool_cfix_r2s2r_exp_est_*.log.

**Results (2026-10-23, Mac `compare_est.py`; sim tactile = reference; U8 put into N with each clip's true scale S).**

| group | n | total-force r | total MAE (N) | contact IoU | release error | finger-share TV | pad contact F1 |
|---|---|---|---|---|---|---|---|
| new: r10 (other scenes) | 7 | 0.31 | 12.8 | 0.84 | 786 ms | 0.27 | 0.73 |
| new: r8b (box scene / top photo) | 12 | 0.26 | 11.3 | 0.84 | 822 ms | 0.28 | 0.67 |
| U training rows: r6 head | 6 | 0.12 | 12.0 | 0.84 | 822 ms | 0.28 | 0.65 |
| baseline: always in contact | 25 | - | - | 0.84 | 812 ms | - | - |

- U8 finds contact from frame 0 and holds roughly the right level, but predicts contact on EVERY frame (100 %): the release
  (sim frame 66-70, ~4.2 s) is never estimated, so its contact timing equals the always-in-contact baseline.
- Force over time correlates weakly (0.08-0.51 per video); it starts high (~45 N at frame 0 vs ~10 N sim) and stays flat where
  the sim force changes (rim pinch climbs to 70 N). Its output barely changes across perturbations (relative touch: the 30 %
  grip's lower force shows only through the oracle S). Training rows are not better (r 0.12).
- f2 FAILED: under the full-video mask it paints the desk scene into the panel (no touch map; decoded totals 300-976 N).
  Stopped after r10 + part of r8b / r6 (14 outputs kept). f2 needs a frame-0 touch map to draw the panel at all.
- Files: Mac scene_mug/runs/clips/release_estimate_pull/compare/{scores.md, scores.json, fig_total.png}.
