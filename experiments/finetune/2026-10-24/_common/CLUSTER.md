# HF25 on the authors' cluster (internal notes)

Workspace root `R = /scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r` (`train.sh` picks it up when it exists; `launch.sh`
exports it). Everything of §2 of the runbook is already in place there: weights `$R/models/Wan2.2-Fun-A14B-Control`, VideoX-Fun
`$R/third_party/VideoX-Fun` (patched, trainer copy generated), envs `$R/envs/videoxfun` + `$R/envs/armA`, raw release
`$R/data/mug-sim2video-pose-tactile`, training set `$R/outputs/R2S2V_mug/hf25/dataset` (+ `hf25_dataset_v2.tar.gz`, the upload
to Hugging Face minus `test/simtouch`).

**Launch (GPU pool, 4 h allocations, card cap in `$R/experiments/R2S2V_mug/pool_cap`):**
```bash
bash $R/experiments/finetune/2026-10-24/FHN/code/launch.sh      # = _common/code/launch.sh FHN_high FHN_low ev_FHN_500
bash $R/experiments/finetune/2026-10-24/SVO/code/launch.sh
bash $R/experiments/finetune/2026-10-24/CDS/code/launch.sh
```
One `pool_retry.sh` loop per training part (relaunches after every preemption / 4 h cut; the trainer resumes), one waiter per
eval part (starts when both experts have step 500). Expect ≥ 1 kill per expert (6.5 h > 4 h): 8-9 h wall-clock each.

**Monitoring:** step stdout `$R/logs/r2s2v/pool_cfix_r2s2r_hf25_<part>.log`; loops `$R/logs/hf25/{loop,wait}_<part>.log`;
done markers `$R/outputs/R2S2V_mug/done_exp_hf25_<part>` (`.gaveup` = the loop stopped: missing script or rc 126 / 127);
live loops `pgrep -af "pool_retry.sh cfix_r2s2r_hf25"`. Relaunch after a `.gaveup` with the same `launch.sh` (it skips parts
that are done or looping). Never start a second loop by hand.

**Eval extras here:** `regr_hf25.py` (R6 regressor, needs `$R/data/TouchScale/metadata/glove_geometry_left.json`) and
`labeladh.py` (SAM3 mug / hand IoU vs `outputs/B_sim2wm/sim/<clip>/seg_simonly.npy`) run automatically in `ev_*` parts.

**Smoke test:** `bash _common/code/launch_smoke.sh` (tiny model `$R/outputs/R2S2V_mug/lora_ge/tiny_model`, one card, ~3 min;
passed 2026-10-24 06:02). **Dataset rebuild:** `rm -rf $R/outputs/R2S2V_mug/hf25/dataset && $R/envs/armA/bin/python
_common/code/build_hf25.py` (CPU, ~10 min; needs the sim renders under `outputs/B_sim2wm/sim/`; the twins' cube controls come
from `mac/twin_cubes.sh`).
