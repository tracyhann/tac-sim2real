# For agents and new collaborators

This repository holds fine-tuning experiments (LoRAs on Wan2.2-Fun-A14B-Control that generate video together with glove touch).
Each experiment is a folder `experiments/finetune/<date>/<title>/` with its own README and code; `experiments/finetune/INDEX.md`
lists them.

**To run the current experiment (HF25, branch `for-amir`)**, follow `experiments/finetune/2026-10-24/README.md` — it is a
complete runbook: public downloads (weights, VideoX-Fun commit, the prepared dataset on Hugging Face), `setup.sh` (creates the
Python env from `env/requirements.txt` and fetches everything), `train.sh` (six independent one-GPU training parts, resumable),
`train.sh ev_*` (evaluation → a metric table). Section 8 of that README is the whole procedure as a command sequence.

Rules of thumb: set `R2S2R` (the data / weights root) before any script; do not modify VideoX-Fun beyond the two patches the
setup applies; keep the training flags as they are (the arms must differ only in the control video); report
`$R2S2R/outputs/R2S2V_mug/hf25/eval/table.md` and the per-case json files.

Older experiment folders are records of runs on the authors' cluster; their scripts contain that cluster's paths.
