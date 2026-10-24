# For agents and new collaborators

This branch holds one fine-tuning experiment: three LoRAs on the video model **W** that generate video together with glove touch.
Follow `experiments/finetune/2026-10-24/README.md` — a complete runbook: public downloads (the W weights and the prepared dataset
on Hugging Face, the training framework at a pinned commit), `setup.sh` (creates the Python env from `env/requirements.txt` and
fetches everything), `train.sh` (six independent one-GPU training parts, resumable), `train.sh ev_*` (evaluation → a metric
table). Section 8 of that README is the whole procedure as a command sequence.

Rules: set `R2S2R` (the data / weights root) before any script; do not modify the training framework beyond the two patches the
setup applies; keep the training flags as they are (the arms must differ only in the control video); refer to the backbone only
as "W" (its identity is withheld for this test); report `$R2S2R/outputs/R2S2V_mug/hf25/eval/table.md` and the per-case json files.
