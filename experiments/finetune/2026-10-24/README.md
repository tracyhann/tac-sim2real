# HF25 runbook · three video + touch LoRAs on Wan2.2-Fun-A14B-Control (FHN / SVO / CDS)

Everything needed to run the 2026-10-24 experiment: what it is, where the data and weights come from, how the training set is
prepared, how to launch, monitor and evaluate. Written for someone who did not build it. Design details and history live in
[_common/README.md](_common/README.md); per-arm notes in [FHN](FHN/README.md), [SVO](SVO/README.md), [CDS](CDS/README.md).

**Status (2026-10-24 06:30 CDT).** Code ready, dataset built and checked on Vision, trainer smoke-tested on a GPU (tiny model:
train, checkpoint, resume, head save / load all pass). The real runs are NOT launched: start them with the commands in §5.

## 1. The experiment in one paragraph
Fine-tune Wan2.2-Fun-A14B-Control (a 14B video diffusion model with two experts, high / low noise) with LoRA so that, from a
first frame + a control video + a text prompt, it generates a 5 s video of a hand carrying a mug AND the paired tactile reading
of the TouchScale glove (880 sensors x 81 frames). The touch is a value array, never an image: a small head on the low-noise
expert's last-layer tokens outputs it; it is trained together with the LoRA. Three arms are trained on the same 25 pairs with
the same prompts, steps and settings; only the control video differs:

| arm | control video |
|---|---|
| **FHN** | the control each video was generated from: static scene depth (inverse-depth grey) + the simulated hand / mug / coaster rendered on top (the F / H format) |
| **SVO** | the simulation render only: hand / mug / coaster on black, no depth |
| **CDS** | depth of a cube scene (static boxes, no body) + the simulation render on top |

Note: 19 of the 25 videos were generated from cube controls, so for those rows CDS = FHN; the arms differ on the 6 r6 rows and
on the held-out test controls.

## 2. Where everything is on the Vision cluster (nothing to download there)
Workspace root `R = /scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r`.

| what | path |
|---|---|
| this experiment's code | `$R/experiments/finetune/2026-10-24/` (`_common/code/` = all scripts; `<ARM>/code/launch.sh` = one-line launchers) |
| HF release (25 pairs, raw) | `$R/data/mug-sim2video-pose-tactile/` (42 MB; `REVISION` = the pinned commit) |
| prepared training set | `$R/outputs/R2S2V_mug/hf25/dataset/` (65 MB) and its tarball `$R/outputs/R2S2V_mug/hf25/hf25_dataset_v2.tar.gz` (63 MB, md5 `35b0b5dac0a09c98f561d36229f36ab6`) |
| backbone weights | `$R/models/Wan2.2-Fun-A14B-Control/` (65 GB) |
| VideoX-Fun (trainer + pipeline) | `$R/third_party/VideoX-Fun/` (commit `13bb3c6`, shared trainer carries two local patches; our trainer copy `scripts/wan2.2_fun/train_control_lora_hf25.py` + `hf25_patch.py`) |
| Python envs | `$R/envs/videoxfun/bin/python` (training / inference), `$R/envs/armA/bin/python` (data build / scoring) |
| run outputs | `$R/outputs/R2S2V_mug/hf25/{<ARM>_high, <ARM>_low, out/, regr/, eval/}` |
| logs | `$R/logs/hf25/` (loops, waiters, builds), `$R/logs/r2s2v/pool_cfix_r2s2r_hf25_<part>.log` (the GPU step's stdout) |
| done markers | `$R/outputs/R2S2V_mug/done_exp_hf25_<part>` |

The sim controls (`$R/outputs/B_sim2wm/sim/tsmug-*/views*/{fun_ctrl.mp4, sim.mp4}`) are the only input that exists nowhere
else; the prepared dataset / tarball already contains the resized copies the training needs.

## 3. Setting up elsewhere (fresh machine, 1 x 80 GB+ GPU per training part)
Recreate the layout under a root of your choice and export `R2S2R=<root>` (train.sh reads it; the eval scripts are Vision-only,
see §6). Steps:

1. **Backbone weights (65 GB).** Hugging Face repo `alibaba-pai/Wan2.2-Fun-A14B-Control` (Apache-2.0):
   ```bash
   hf download alibaba-pai/Wan2.2-Fun-A14B-Control --local-dir $R2S2R/models/Wan2.2-Fun-A14B-Control
   ```
   Expected layout: `high_noise_model/` (27 GB, `config.json` + `diffusion_pytorch_model.safetensors`), `low_noise_model/` (27 GB),
   `models_t5_umt5-xxl-enc-bf16.pth` (11 GB), `Wan2.1_VAE.pth` (485 MB), `google/` (the umt5 tokenizer, 21 MB).
2. **VideoX-Fun + environment.**
   ```bash
   git clone https://github.com/aigc-apps/VideoX-Fun.git $R2S2R/third_party/VideoX-Fun
   git -C $R2S2R/third_party/VideoX-Fun checkout 13bb3c625e8b1e457d7124d92207b49e00e4a6fe
   uv venv --python 3.12 $R2S2R/envs/videoxfun && source $R2S2R/envs/videoxfun/bin/activate
   pip install -r $R2S2R/third_party/VideoX-Fun/requirements.txt
   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # if requirements picked another build
   pip install peft==0.21.2 tensorboard av decord opencv-python-headless           # the trainer imports tensorboard; peft >= 0.21 for inject_adapter_in_model
   ```
   Versions in use on Vision: Python 3.12.12, torch 2.8.0+cu128, peft 0.21.2, diffusers 0.35.1, accelerate 1.15.0,
   transformers 4.56.2, av 19, decord 0.6, opencv 5.0, einops 0.8.2, omegaconf 2.4.
3. **Trainer patches (in this order; each is idempotent and asserts its anchors).** The shared VideoX-Fun trainer needs two opt-in
   flags we rely on (`--inpaint_first_frame_only`: the inpaint mask keeps only frame 0, as at inference; `--resume_lora_weights`:
   resume from `checkpoint-N.safetensors` without accelerate state dirs), then our copy with the touch head is generated:
   ```bash
   C=$R2S2R/experiments/finetune/2026-10-24/_common/code; V=$R2S2R/third_party/VideoX-Fun
   python $C/videoxfun_patches/patch_trainer.py  $V/scripts/wan2.2_fun/train_control_lora.py
   python $C/videoxfun_patches/patch_trainer2.py $V/scripts/wan2.2_fun/train_control_lora.py
   python $C/make_trainer_hf25.py $V      # -> scripts/wan2.2_fun/train_control_lora_hf25.py + hf25_patch.py (train.sh also does this if missing)
   ```
4. **Data.** Public release (CC BY-NC 4.0) `tracyhan816/mug-sim2video-pose-tactile`, pinned commit `6cf129c` — 25 folders, each
   `video.mp4` (832 x 480, 81 frames, 16 fps, a zero-shot Wan2.2-Fun generation of a counterfactual replay of clip edf79415),
   `tactile.npz` (`normal_N` (81, 880): the left glove's simulated normal force per sensor at the video frames, `t_video`,
   `sim_covered`), `hand_pose.npz`, plus `metadata.csv` (camera, first frame, perturbation per video):
   ```bash
   bash $C/fetch_hf.sh        # -> $R2S2R/data/mug-sim2video-pose-tactile (42 MB); download the whole repo, --include dropped files once
   ```
5. **Prepared training set.** Either copy the tarball from Vision (recommended: it has the controls) and extract it:
   ```bash
   mkdir -p $R2S2R/outputs/R2S2V_mug/hf25 && tar -xzf hf25_dataset_v2.tar.gz -C $R2S2R/outputs/R2S2V_mug/hf25   # -> hf25/dataset/
   ```
   or rebuild it with `build_hf25.py` (§4) — that needs the sim controls, i.e. the Vision tree.

## 4. The training set and how it is prepared (`_common/code/build_hf25.py`)
Run on Vision (armA env, CPU, ~10 min; stops on any failed check):
```bash
cd $R/experiments/finetune/2026-10-24/_common/code && rm -rf $R/outputs/R2S2V_mug/hf25/dataset && $R/envs/armA/bin/python build_hf25.py
```
What it does per HF row (`release_cases.json` maps each HF id to the Vision generation it came from and its control):
- copies `video.mp4` as is (verified frame-by-frame against the Vision generation, MAE 0.0), writes `start/<id>.png` (frame 0);
- touch -> `tactile/<id>.npz`: `touch` (N), `S` (= median total force over the frames with total > 1 N of that replay; the
  normalisation), `touch_rel = touch / S`, `u = min(touch_rel, 0.11) / 0.11` (the head's target in [0, 1]), `t`, `sim_covered`;
- the three controls, resized 854 x 480 -> 832 x 480: `ctrl/<clip>_<views>_<view>_fun.mp4` (depth + sim render) and `..._sim.mp4`
  (sim render only); the 6 r6 rows take the views9 cube renders of the same replay (same head camera) for SVO / CDS;
- prompt from `prompts.py`: one shared action sentence + a visual-feature sentence where the video has one (water in the mug,
  claw grip, rim pinch, shaken, very fast) + the touch sentence; 9 rows get the shared prompt alone.
Outputs: `metadata_FHN.json` / `metadata_SVO.json` / `metadata_CDS.json` (identical rows except `control_file_path`; the trainer
reads `file_path`, `control_file_path`, `text`, `tactile`), `rows.json`, `test_cases.json` (4 held-out twin cases + 3 in-sample
rows with per-arm controls; starts / touch under `test/`), `check.json` (per-row checks + summary). Checks: 81 frames at 16 fps for
every stream, touch times = video times, finite touch, the r6 -> views9 swap shows the same replay, SVO controls mostly black.
Current summary: 25/25 ok, S 11-42 N, 4 % of sensor readings active, the cap clips at most 2.8 % of a row's active readings.

## 5. Training
**Settings** (`_common/code/train.sh`, identical to the earlier F / H runs except the touch head and normalisation): LoRA rank 64 /
alpha 32 on q, k, v, ffn.0, ffn.2 of both experts, lr 1e-4 constant, batch 1, 81 frames at 480 x 832, first-frame-only inpaint
mask, control_ref + first_frame, reference image in self-attention, 10 % control dropout (trainer default), 500 steps per expert
(= 20 passes over the 25 rows), checkpoint every 25 steps (multiples of 100 kept + the newest). The LOW-noise expert also trains
the touch head (`--tactile_head --tactile_lambda 1.0`: loss = video flow matching + 1.0 x R1 touch loss; head in fp32, LoRA in
bf16; saved as `tactile_head-<step>.pt` beside each `checkpoint-<step>.safetensors`). The high-noise expert is a plain video LoRA.
Parts: `FHN_high FHN_low SVO_high SVO_low CDS_high CDS_low` (one GPU each, ~45 s/step on an H200 -> ~6.5 h of compute each),
then `ev_FHN_500 ev_SVO_500 ev_CDS_500` (one GPU, ~1 h each).

**On Vision (pool cards, the intended way).** Each arm's launcher starts one retry loop per expert and a waiter that starts the
evaluation once both experts have step 500:
```bash
bash /scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/experiments/finetune/2026-10-24/FHN/code/launch.sh
bash /scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/experiments/finetune/2026-10-24/SVO/code/launch.sh
bash /scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/experiments/finetune/2026-10-24/CDS/code/launch.sh
```
Run them on the login node (`ssh vision`); they return at once. Pool steps run inside 4 h allocations and can be preempted, so
every training part is EXPECTED to be killed at least once (6.5 h > 4 h): `pool_retry.sh` relaunches it and the trainer resumes
from the newest checkpoint + head with a fresh Adam (as F / H did). Each kill costs < 25 steps + ~10 min of model loading;
expect 8-9 h wall-clock per expert, more when the pool is busy. The card cap (`$R/experiments/R2S2V_mug/pool_cap`, 8) counts all
`cfix_r2s2r` steps of the user, so keep other pool work in mind: the 6 training parts fit, the 3 eval parts queue behind them.

**Elsewhere (one GPU, no pool).** With the layout of §3 and `R2S2R` exported:
```bash
bash $R2S2R/experiments/finetune/2026-10-24/_common/code/train.sh FHN_low     # exit 0 = checkpoint-500 (+ head) written; exit 1 = not finished: run it again, it resumes
```
Repeat per part (6 parts, any order, in parallel on separate GPUs). Ran on H200 141 GB cards with 44 GB of host RAM (the pool
request); the memory need on smaller GPUs was not measured.

**Monitoring.**
```bash
tail -f $R/logs/r2s2v/pool_cfix_r2s2r_hf25_FHN_low.log      # the step's stdout: "Steps: 12%|..| 60/500 [...] step_loss=.., tactile=.."
ls -la $R/outputs/R2S2V_mug/hf25/FHN_low/                   # checkpoint-N.safetensors + tactile_head-N.pt (low) every 25 steps
tail -3 $R/logs/hf25/loop_FHN_low.log                        # the retry loop: "run ... on <job> <node>" / "done ... rc N"
ls $R/outputs/R2S2V_mug/done_exp_hf25_*                      # finished parts
pgrep -af "pool_retry.sh cfix_r2s2r_hf25"                    # live loops
```
A loop that gives up writes `done_exp_hf25_<part>.gaveup` with the reason (a missing script, rc 126 / 127). A step that exits
rc 1 right after start: read the step log (the trainer's traceback is there), fix, and relaunch the same `launch.sh` (it skips
parts that are done or already looping).

**Smoke test (optional, 1 card, ~3 min).** `bash $R/experiments/finetune/2026-10-24/_common/code/launch_smoke.sh` trains a
tiny stand-in model for 3 steps, resumes to 4 and checks the head files; passed on 2026-10-24.

## 6. Evaluation (`ev_<ARM>_500`, started by the launcher's waiter; or `bash _common/code/launch.sh ev_FHN_500`)
1. `predict_hf25.py`: both experts' step-500 LoRAs merged into the pipeline + the low expert's head; for each test case
   (`dataset/test_cases.json`: twins 55494490 / 4397541c from the real frame 0 + its touch and from the zero-shot generated frame 0
   + sim touch; 3 training rows) generates the video (CFG 6, start frame as reference image, 50 steps, seed 0) and reads the
   touch from the head at the last denoising step -> `out/<ARM>c500/<case>_s0.{mp4, _tactile.npy (81 x 880, F / S units), json}`.
2. `regr_hf25.py`: the reference "video + separate estimator": an R6 regressor (frozen Wan VAE latents -> touch) trained on the
   same 25 pairs, applied to the arm's generated videos -> `out/<ARM>c500_regr/`.
3. `labeladh.py` (Vision tree): mug / hand IoU of the held-out videos vs the simulation silhouettes (SAM3).
4. `score_hf25.py` -> `eval/table.md` + `eval/scores.json`: the full held-out metric set (contact F1 @ 6 mm / pad / taxel,
   pad-time F1, time IoU, onset / release error, total and finger force correlation, load-share TV, relative and oracle-N MAE)
   for every arm, next to the reference rows (R6 regressor, frame-0 touch repeated, the twin's own sim touch, dual-mode G epoch 8,
   the earlier h2 LoRA, zero-shot / real video IoU). The score script runs on CPU in seconds and can be rerun any time.

Steps 1-4 import code and models from the Vision tree (`experiments/R2S2V_mug/fun_zeroshot.py`, the dual-mode metrics,
SAM3, the simulation masks), so run the evaluation on Vision. Trained LoRAs from another machine can be copied into
`$R/outputs/R2S2V_mug/hf25/<ARM>_{high,low}/` and evaluated there.

## 7. Reading the result
`eval/table.md`: one row per (case group, method). Compare the three arms' "own touch head" rows with each other and with the
"R6 regressor on the ... video" rows (does the backbone's head beat an external estimator on its own video?) and the baselines.
Held-out groups are the ones that matter; the in-sample group only shows whether an arm fitted its data. Caveats to carry into
any write-up: all 25 training pairs are replays of ONE real clip (edf79415), the test set is two twins, FHN and CDS share 19 of 25
controls, the twins' real touch is absolute glove data normalised by its own episode scale.

## 8. Checklist for the helper (Vision)
1. `ssh vision`; check the pool is not full: `ps -u $USER -o args | grep -c "[p]ool_run.py --name cfix_r2s2r"` (< 8).
2. `ls $R/outputs/R2S2V_mug/hf25/dataset/metadata_*.json` exists (else §4).
3. Run the three `launch.sh` commands of §5.
4. Over the next ~10 h: the monitoring commands of §5; relaunch after a `.gaveup`, never start a second loop by hand.
5. When `done_exp_hf25_ev_<ARM>_500` exists for all three: `cat $R/outputs/R2S2V_mug/hf25/eval/table.md`.
