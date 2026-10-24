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

**First-pass results (2026-10-23, Mac `compare_est.py`; reduced metric set, SUPERSEDED by the full table below).**

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

**Full metrics (2026-10-24, user: "the metrics are incomplete").** `code/compare_full.py` scores with the held-out eval's own
metric set (dual_mode_lora/code/eval_ge.metrics: relative touch, panel v3 thresholds) + baselines + a perturbation score.
New method: R6 regressor (the loss table's best head, frozen VAE latents -> touch; trained as there on the 79 lora_ge rows,
seeds 0-2; `code/regr_release.py`, part est_R6, 6 min on one card). Baselines: frame-0 sim touch repeated (knows frame 0);
recorded replay's touch (the unperturbed sim's touch for every video = ignores the perturbation). delta_r = r of the
perturbation-induced change in relative total force (video minus the recorded video of the same view, 15 videos).
Onset is 0 ms for every method: the sim is already in contact at frame 0, so onset carries no information here.

All 19 new videos (r10 + r8b); R6 = mean of 3 seeds:

| method | F1@6mm | pad F1 | pad-time F1 | time IoU | release ms | total r | finger r | share TV | rel. MAE | N MAE (oracle S) | delta_r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U8 (dual-mode, job E) | 0.29 | 0.66 | 0.89 | 0.82 | 839 | 0.28 | 0.32 | 0.29 | 0.34 | 11.9 | -0.07 |
| R6 regressor | 0.63 | 0.89 | 0.91 | 0.92 | 232 | 0.80 | 0.73 | 0.12 | 0.18 | 6.5 | -0.06 |
| f2 (n = 11) | 0.19 | 0.37 | 0.77 | 0.82 | 835 | -0.21 | -0.02 | 0.31 | 9.15 | 292 | - |
| frame-0 sim touch repeated | 0.65 | 0.70 | 0.64 | 0.82 | 839 | - | - | 0.30 | 0.62 | 22.0 | - |
| recorded replay's touch (n = 17) | 0.86 | 0.96 | 0.93 | 0.99 | 40 | 0.96 | 0.93 | 0.08 | 0.07 | 2.7 | - |

- U8 loses to the R6 regressor on every metric (as on the held-out real clips) and to frame-0-repeated on contact location.
- Nothing beats "recorded replay's touch": on these videos the perturbations barely change the relative touch, and neither
  model follows the change they do make (delta_r about 0 on new videos; R6 0.43 only on its own training rows).
- The posture videos (r10 claw x2, rim pinch) are the hardest: U8 total r 0.07-0.12; R6 seed 0 -0.12 / 0.15 / 0.83.
- The test set is narrow: all 25 videos come from one clip (edf79415), whose other renders are training rows of both U8 and
  R6, so R6's "new" scores are partly recall of that clip's touch.
- Files: Mac scene_mug/runs/clips/release_estimate_pull/compare_full/{scores_full.md (group + per-video), scores_full.json};
  Vision outputs/R2S2V_mug/release_estimate/R6_s{0,1,2}/, regr_lat/.

**E8 / R8 added, f2 completed (2026-10-24, user: "do you have E, R and f2 results?").** The dual-mode estimate-only runs at
epoch 8: E8 (real + sim rows, E_{low,high}/checkpoint-632; r6 head videos are its training rows) and R8 (real rows only,
R_{low,high}/checkpoint-320; all 25 videos new), same job-E protocol as U8 (predict_release.py --model E8|R8); f2 finished on
all 25 (parts est_f2_r8b / est_f2_r6 relaunched). 8 parts on 8 cards, 01:53-02:41 CDT. compare_full.py now scores all of them
and writes fig_total_full.png (relative total per video: sim, recorded replay, U8, E8, R8, R6 seed 0).

All 19 new videos (r10 + r8b):

| method | F1@6mm | pad F1 | time IoU | onset ms | release ms | total r | finger r | share TV | rel. MAE | N MAE (oracle S) | delta_r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U8 (both jobs) | 0.29 | 0.66 | 0.82 | 0 | 839 | 0.28 | 0.32 | 0.29 | 0.34 | 11.9 | -0.07 |
| E8 (estimate only, real + sim) | 0.39 | 0.60 | 0.75 | 454 | 839 | -0.13 | 0.04 | 0.35 | 0.50 | 17.1 | -0.12 |
| R8 (estimate only, real) | 0.30 | 0.53 | 0.73 | 56 | 839 | -0.29 | -0.07 | 0.49 | 1.06 | 36.6 | -0.04 |
| f2 (video + touch LoRA) | 0.18 | 0.35 | 0.82 | 0 | 839 | -0.14 | 0.03 | 0.30 | 14.2 | 422 | 0.04 |
| R6 regressor (3 seeds) | 0.63 | 0.89 | 0.92 | 7 | 232 | 0.80 | 0.73 | 0.12 | 0.18 | 6.5 | -0.06 |
| frame-0 sim touch repeated | 0.65 | 0.70 | 0.82 | 0 | 839 | - | - | 0.30 | 0.62 | 22.0 | - |
| recorded replay's touch | 0.86 | 0.96 | 0.99 | 0 | 40 | 0.96 | 0.93 | 0.08 | 0.07 | 2.7 | - |

- Same ranking as on the held-out twins' generated videos with sim control (lora_ge eval/table.md: Ep8 r -0.34, Rp8 -0.28,
  Up8 0.11, R6 0.76), so this is not a run error.
- No LoRA ever releases (release error = the always-on value, 839 ms). E8 starts contact ~0.45 s late and rises to ~1.5 S at the
  end where the sim lets go; R8 draws the same curve on every video (peaks ~3.3 S at 2.2 s and at 3.6-5 s; mean pairwise r of
  its 25 total curves 0.98) and overshoots ~2x (rel. MAE 1.06). E8 has the best LoRA contact location (F1@6mm 0.39).
- Contact location does not follow the video for any model: the sim's time-averaged taxel maps differ across the 25 videos
  (mean pairwise r 0.75: claw / rim pinch move the contacts), the predictions do not (U8 0.99, E8 0.95, R8 0.99, R6 0.99).
- E8 on its own training rows (r6) is no better (r -0.22): the LoRAs did not even fit these sim rows' touch.

**Mixed vs real-only at equal steps (2026-10-24, user: "can we conclude mixed training is better?").** E8 (mixed) beats R8
(real only) above, but E8 had 632 steps and R8 320. E4 = run E at checkpoint-316 (R8's step count), same protocol
(predict_release.py --model E4, parts est_E4_<group>). All 19 new videos:

| method | steps | F1@6mm | pad F1 | time IoU | onset ms | total r | finger r | share TV | rel. MAE | N MAE (oracle S) |
|---|---|---|---|---|---|---|---|---|---|---|
| E8 (mixed) | 632 | 0.39 | 0.60 | 0.75 | 454 | -0.13 | 0.04 | 0.35 | 0.50 | 17.1 |
| E4 (mixed) | 316 | 0.31 | 0.56 | 0.82 | 0 | -0.34 | -0.03 | 0.51 | 1.15 | 39.6 |
| R8 (real only) | 320 | 0.30 | 0.53 | 0.73 | 56 | -0.29 | -0.07 | 0.49 | 1.06 | 36.6 |

- At equal steps mixed = real-only (E4 ~ R8 on every metric, within what one seed can tell), even though these videos are
  in-domain for the sim rows. E8's lead comes from the extra steps, as on the held-out clips (actor_01: Ep4 r 0.61 / F1 0.36 /
  63 N vs Rp8 0.59 / 0.38 / 67 N). No evidence here that sim rows help the LoRA estimate touch.

**Contact state and timing (2026-10-24, user's metric list).** `code/finger_metrics.py` + `code/compare_contact.py` ->
scores_contact.md. Parts thumb / index / middle / ring / little / palm; a part is in contact when its total relative force
> 0.04 S (~1 N at the median S), both sides. finger_acc / finger_f1 over (frame, part); press_dur_err = |pred - true contact
frames| per part; contact_end_err = |pred - true last contact frame| per part in contact in both; load_share_tvd = eval_ge
share_tv. All 19 new videos:

| method | finger acc | finger F1 | press-duration err (frames) | contact-end err (frames) | load-share TVD |
|---|---|---|---|---|---|
| U8 | 0.66 | 0.71 | 30.6 | 13.4 | 0.29 |
| E8 | 0.59 | 0.65 | 31.2 | 13.4 | 0.35 |
| E4 | 0.58 | 0.63 | 39.6 | 17.1 | 0.51 |
| R8 | 0.53 | 0.58 | 30.2 | 13.4 | 0.49 |
| f2 | 0.49 | 0.62 | 41.7 | 13.5 | 0.30 |
| R6 regressor (3 seeds) | 0.94 | 0.94 | 7.0 | 5.5 | 0.12 |
| frame-0 sim touch repeated | 0.80 | 0.75 | 30.4 | 13.2 | 0.30 |
| recorded replay's touch | 0.96 | 0.96 | 4.0 | 0.4 | 0.08 |

- The sim grasps with thumb, index and middle only; the LoRAs also press ring and little (and U8 / R8 the little finger
  hard), so their finger accuracy is below frame-0-repeated. Contact-end error ~13 frames = never letting go (the sim ends
  at frame ~67 of 80). Same ranking at 0.02 S and 0.08 S (sensitivity tables in scores_contact.md).
