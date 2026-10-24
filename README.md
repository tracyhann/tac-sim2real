# tac-sim2real · branch `for-amir`

Experiment code for touch-grounded real-to-sim-to-video: a tactile glove records contact during tabletop manipulation, each
clip is rebuilt as a physics replay, and a video diffusion model is fine-tuned (LoRA) to generate head-camera video together
with the glove's touch.

This branch contains one experiment, packaged to be run by someone outside the lab:
**`experiments/finetune/2026-10-24/`** — three video + touch LoRAs on the video model **W** that differ only in the control
video (FHN / SVO / CDS). Start at its [README](experiments/finetune/2026-10-24/README.md) (or [AGENTS.md](AGENTS.md)): every
download is public (W weights and the prepared training set on Hugging Face, the training framework at a pinned commit),
`setup.sh` installs the environment (`env/requirements.txt`) and fetches everything, `train.sh` trains and evaluates.

The backbone is anonymised for this internal test — refer to it as "W" in labels, tables and write-ups.

```
experiments/finetune/
  INDEX.md
  2026-10-24/            README.md (runbook)  setup.sh  env/  _common/{README.md, CLUSTER.md, code/}  FHN/ SVO/ CDS/
```
Model weights, datasets and outputs are not in this repository.
