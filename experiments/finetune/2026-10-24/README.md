# HF25 · three video + touch LoRAs on the video model W (FHN / SVO / CDS)

Self-contained runbook. Everything the experiment needs is public: the backbone weights and the data are on Hugging Face,
the trainer is VideoX-Fun at a pinned commit, the code is this folder. `setup.sh` fetches and installs all of it; `train.sh`
trains; `train.sh ev_*` evaluates. Design notes: [_common/README.md](_common/README.md); per arm:
[FHN](FHN/README.md), [SVO](SVO/README.md), [CDS](CDS/README.md).

**Status.** Code complete, training set built and checked, trainer smoke-tested end to end on a GPU (train, checkpoint, resume,
touch-head save / load). The six training runs have not been started.

## 1. The experiment
W is a video diffusion transformer with two experts (high noise, low noise) and control-video conditioning. Its identity is
withheld for this internal test: call it "W" everywhere — file names, labels, plots, reports. We fine-tune it with LoRA so
that, from a first frame + a control video + a text prompt, it generates a 5 s video of a hand carrying a mug AND the paired
reading of the TouchScale tactile glove: 880 sensors x 81 frames. The touch is a value array, never an image: a small head on
the low-noise expert's last-layer tokens outputs it and is trained together with the LoRA (video flow-matching loss + touch
loss: per-sensor R1 + per-frame total force; squared-sigmoid decode; the head also sees the DiT's timestep embedding). Three arms share the 25 training pairs, prompts, steps and settings; only the control video differs:

| arm | control video | what it tests |
|---|---|---|
| **FHN** | the control each video was generated from: static scene depth (inverse-depth grey) + the simulated hand / mug / coaster rendered on top | the format of our earlier runs, now with normalised touch and a value head |
| **SVO** | the simulation render only: hand / mug / coaster on black, no depth | is the sim animation alone enough |
| **CDS** | depth of a cube approximation of the scene (boxes, no body) + the simulation render | can crude scene geometry replace the real scan |

19 of the 25 videos were generated from cube controls, so FHN = CDS on those rows; the arms differ on 6 rows and on the
held-out test controls. Touch normalisation: per row, F / S with S = median total force over the frames in contact; the head's
target is u = min(F / S, 0.11) / 0.11. Result = the full touch metric table on two held-out clips (§6, §7).

## 2. What you need (all public)
| item | where | size |
|---|---|---|
| backbone weights (model W) | Hugging Face [latent-artist/861b8a78d34141a9ad659229049b6423](https://huggingface.co/latent-artist/861b8a78d34141a9ad659229049b6423) (Apache-2.0) | 69 GB |
| training framework (trainer + pipeline classes) | GitHub [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) at commit `13bb3c625e8b1e457d7124d92207b49e00e4a6fe` (Apache-2.0) | 50 MB |
| **prepared training set + test cases** | Hugging Face [tracyhan816/mug-video-touch-lora-hf25](https://huggingface.co/datasets/tracyhan816/mug-video-touch-lora-hf25) (CC BY-NC 4.0) — videos, touch arrays, first frames, the three control sets, held-out cases | 64 MB |
| raw public release (provenance only, not needed to train) | Hugging Face [tracyhan816/mug-sim2video-pose-tactile](https://huggingface.co/datasets/tracyhan816/mug-sim2video-pose-tactile), commit `6cf129c` — 25 videos + hand pose + touch | 42 MB |
| this code | GitHub [tracyhann/tac-sim2real](https://github.com/tracyhann/tac-sim2real/tree/for-amir), branch `for-amir`, `experiments/finetune/2026-10-24/` | |
| hardware | Linux, 1 NVIDIA GPU per training run — developed and tested on H200 141 GB only (see §5 / §6 for what is known about 80 GB cards), bf16 support (A100 / H100 class), ~45 GB host RAM, ~85 GB disk, CUDA 12.8 driver | |
| software | git, Python 3.12 (or [uv](https://github.com/astral-sh/uv)), ffmpeg libs come with the pip wheels | |

Python environment: [env/requirements.txt](env/requirements.txt) (the pins that matter: torch 2.8.0+cu128, diffusers 0.35.1,
transformers 4.56.2, accelerate 1.15.0, peft 0.21.2, ...) on top of VideoX-Fun's `requirements.txt`; the exact freeze of the
reference machine is [env/requirements-lock.txt](env/requirements-lock.txt).

## 3. Setup (one command)
Pick a root directory `R2S2R` for weights, VideoX-Fun, the env and outputs (the code can live anywhere; it finds itself):
```bash
git clone -b for-amir https://github.com/tracyhann/tac-sim2real.git
export R2S2R=/data/hf25                           # any writable path with ~85 GB free
bash tac-sim2real/experiments/finetune/2026-10-24/setup.sh         # add --no-weights to skip the 69 GB download, --raw-release to also fetch the raw release
```
`setup.sh` is idempotent and does, in order: clone VideoX-Fun at the pinned commit; create `$R2S2R/envs/videoxfun` (Python 3.12)
and install VideoX-Fun's requirements + `env/requirements.txt`; `hf download` the W weights to `$R2S2R/models/W`
and the training set to `$R2S2R/outputs/R2S2V_mug/hf25/dataset`; apply the two trainer patches
(`_common/code/videoxfun_patches/`: `--inpaint_first_frame_only`, `--resume_lora_weights`) and generate our trainer copy
`train_control_lora_hf25.py` + `hf25_patch.py` (inside the framework's `scripts/` folder); run the CPU unit test of the touch head. Layout it creates:
```
$R2S2R/
  third_party/VideoX-Fun/                 pinned checkout + patched trainer copy
  envs/videoxfun/                         the Python env ($R2S2R/envs/videoxfun/bin/python)
  models/W/                               high_noise_model/ low_noise_model/ (the two experts), the VAE and text-encoder files, google/ (tokenizer)
  outputs/R2S2V_mug/hf25/dataset/         the training set (see the dataset card for every file and format)
  outputs/R2S2V_mug/hf25/<ARM>_{high,low}/   training outputs (created by train.sh)
  cache/huggingface/                      HF cache
```
Manual equivalent: the steps above one by one; the scripts only need `R2S2R` exported (`PY=<python>` overrides the env path).

## 4. The data
**Prompts v2 and the action-model view (2026-10-25).** Every row's `text` is the shared task instruction plus ONE quantitative clause for its
perturbation or grasp (e.g. "shake the mug side to side, about 6 cm at 4 Hz"; "10 times the recorded speed"; "300 g extra"); the same
numbers are in `params`. `action/<id>.npz` holds the hand state per frame (wrist xyz + rot6d + 22 finger joints = 31-D) and the
tactile vector (880, baseline-subtracted, plus the relative version) — the format a vision-tactile-language-action model consumes
(state in, action chunk out, future tactile change as auxiliary target); `metadata_vtla.json` lists every modality per row and
`PROMPTS.md` is the review table. The FHN LoRA runs of 10-24 were trained with prompts v1 (qualitative feature sentence + a
touch-panel sentence); any new run uses v2.
Use the prepared training set as is; `setup.sh` downloads it. Its card lists every file and format. In short: 25 rows, each a
released video (832 x 480, 81 frames, 16 fps) + its first frame + a touch npz (`touch` N, `S`, `touch_rel`, `u`, `t`) + three control
videos; `metadata_<ARM>.json` are the three training manifests (identical except `control_file_path`); `test_cases.json` + `test/`
hold the 7 evaluation cases (4 held-out, from two real clips the training never sees, + 3 in-sample rows) with starts, touch and
a control per arm. Prompts come from `_common/code/prompts.py` (one shared action sentence + a visual-feature sentence where the
video shows one). `_common/code/build_hf25.py` is the script that produced the set from the raw release and our simulation
renders; it is included for provenance and needs the authors' simulation outputs, so you do not run it.

## 5. Training
Settings (`_common/code/train.sh`): LoRA rank 64 / alpha 32 on q, k, v, ffn.0, ffn.2 of both experts, lr 1e-4 constant, batch 1,
81 frames at 480 x 832, first-frame-only inpaint mask (frame 0 known), control_ref + first_frame, reference image in self-attention,
500 steps per expert (= 20 passes over the 25 rows), checkpoint every 25 steps (multiples of 100 kept + the newest). The LOW-noise
expert also trains the touch head (`--tactile_head --tactile_lambda 1.0`; head in fp32, LoRA in bf16; saved as
`tactile_head-<step>.pt` beside each `checkpoint-<step>.safetensors`). The high-noise expert is a plain video LoRA.

Six independent parts, one GPU each (~45 s/step on an H200 → ~6.5 h each; run in parallel on separate GPUs):
```bash
export R2S2R=/data/hf25
C=tac-sim2real/experiments/finetune/2026-10-24/_common/code
bash $C/train.sh FHN_high      # likewise FHN_low  SVO_high  SVO_low  CDS_high  CDS_low
```
Exit code 0 = finished (`checkpoint-500.safetensors`, + `tactile_head-500.pt` for `_low`). Exit code 1 = not finished: run the same
command again and it resumes from the newest checkpoint (fresh optimizer state, constant LR), so it is safe to run under any
scheduler that kills jobs — each kill costs at most 25 steps + ~10 min of model loading. Exit code 2 = `R2S2R` not set. Progress
is in the console (`Steps: 12%|...| 60/500 [..] step_loss=.. tactile=..`; `tactile` is the touch loss, nan for `_high`) and as files
in `$R2S2R/outputs/R2S2V_mug/hf25/<ARM>_<expert>/`. Logs: redirect the console yourself (`... > logs/FHN_high.log 2>&1`).
Run each part under `nohup` / `tmux` (6-9 h). GPU memory: measured only on H200 141 GB; one expert (28 GB bf16) + LoRA + activations
with gradient checkpointing should fit an 80 GB card, but this was not verified — if a part OOMs on 80 GB, report it rather than
changing flags. Host RAM ~45 GB.

## 6. Evaluation (per arm, one GPU, ~1 h)
```bash
bash $C/train.sh ev_FHN_500     # needs FHN_high and FHN_low at step 500; likewise ev_SVO_500, ev_CDS_500
```
It runs:
1. `predict_hf25.py` — both experts' LoRAs merged into the pipeline + the low expert's head; for each test case it generates the
   video (CFG 6, start frame as reference image, 50 steps, seed 0) and reads the touch from the head at the last denoising step →
   `$R2S2R/outputs/R2S2V_mug/hf25/out/<ARM>c500/<case>_s0.{mp4, _tactile.npy (81 x 880, relative F / S), json}`.
2. `score_hf25.py` — the full touch metric set on every evaluated arm vs the ground truth of each case (contact F1 at 6 mm / pad /
   sensor, pad-time F1, time IoU, onset / release error, total and per-finger force correlation, load-share error, relative and
   oracle-newton MAE), next to the baselines that need no extra data: "frame-0 touch repeated" and "twin sim touch" →
   `$R2S2R/outputs/R2S2V_mug/hf25/eval/{table.md, scores.json}`. CPU, seconds; rerun any time (`python $C/score_hf25.py`).

GPU memory for step 1: as configured, the whole pipeline (both experts + text encoder, ~70 GB of weights) sits on the GPU, which
needs > 80 GB. On an 80 GB card set `W_GPU_MEMORY_MODE=model_cpu_offload` (or `sequential_cpu_offload`, slower) before the
`ev_*` command: the framework's own offload helper then places the models (needs ~70 GB of host RAM). The authors ran only the
all-on-GPU mode.

Two optional reference rows run only on the authors' cluster and are skipped automatically elsewhere (the table simply lacks
them): an external R6 touch regressor on the generated videos (needs the glove geometry file) and mug / hand IoU of the generated
videos vs the simulation (needs SAM3 + the simulation masks). Trained LoRAs can be sent back for those.

## 7. Reading the result
`eval/table.md`: one row per (case group, method). The held-out groups decide; the in-sample group only shows whether an arm
fitted its data. Compare the arms' "own touch head" rows with each other and with the baselines (a method that does not beat
"frame-0 touch repeated" on contact F1 or "twin sim touch" on total-force correlation has not learned the touch). Caveats for any
write-up: all 25 training pairs are replays of ONE real clip; the test set is two clips; FHN and CDS share 19 of 25 controls; the
held-out real touch is absolute glove data normalised by its own episode scale.

## 8. For an agent: the whole thing as commands
```bash
git clone -b for-amir https://github.com/tracyhann/tac-sim2real.git && cd tac-sim2real/experiments/finetune/2026-10-24
export R2S2R=/data/hf25 && bash setup.sh                                   # ~1 h, dominated by the 69 GB download
for p in FHN_high FHN_low SVO_high SVO_low CDS_high CDS_low; do             # one GPU each: 6 in parallel (CUDA_VISIBLE_DEVICES=k) or sequentially; rerun a part until exit 0
  nohup bash -c "until bash _common/code/train.sh $p; do sleep 30; done" > train_$p.log 2>&1 &   # drop the & to run sequentially
done; wait
for a in FHN SVO CDS; do bash _common/code/train.sh ev_${a}_500; done
cat $R2S2R/outputs/R2S2V_mug/hf25/eval/table.md
```
Do not edit VideoX-Fun beyond the two patches; do not change the trainer flags (the arms must stay identical except the control);
report the table and the per-case `out/<ARM>c500/*.json`. Refer to the backbone only as "W".

## 9. Anonymisation
The backbone is referred to as **W** throughout this folder, the dataset card and the output names (`w-hf25...`). The training
framework and the published weight files keep their own names; do not copy them into labels, tables or write-ups — results must
not be attributable to a specific model.

## 10. Troubleshooting
- `set R2S2R=...`: export the root (every script needs it).
- `resume_lora_weights: ... tactile_head-N.pt missing`: a kill landed between the two saves; `train.sh` deletes such a checkpoint
  on the next start and resumes from the previous pair.
- `expected scalar type Float but found BFloat16` or a shape assert in `hf25_patch.py`: the trainer copy is stale — rerun
  `python _common/code/make_trainer_hf25.py $R2S2R/third_party/VideoX-Fun`.
- `ImportError: cannot import name 'BucketNotFoundError' from 'huggingface_hub.errors'` at the start of training: the environment got
  `datasets >= 5.1` (needs `huggingface_hub >= 1.31`, which `transformers 4.56` forbids). Fixed 2026-10-24: `setup.sh` now installs
  `env/requirements-lock.txt` (the exact reference environment) and pins `datasets==5.0.1`; `git pull` and re-run `setup.sh` — it
  repairs an existing env in place. `pip`'s "dependency conflicts" lines about gradio and "decord 0.6.0 is not supported on this
  platform" are warnings only (gradio is never imported; decord's wheel metadata is wrong but the module works).
- The trainer imports `tensorboard`, `decord`, `av`; `setup.sh` installs them and checks every unconditional import of the trainer at
  the end. pip's cache and build temp go under `$R2S2R/cache/` so a small home-directory quota cannot break wheel builds.
- CUDA OOM on < 80 GB cards: not supported as configured.
- `_common/code/smoke_tiny.sh` (trainer smoke test on a tiny stand-in network) is wired to the authors' cluster; see CLUSTER.md.
- Authors' cluster specifics (job pool, log locations, launchers): [_common/CLUSTER.md](_common/CLUSTER.md).
