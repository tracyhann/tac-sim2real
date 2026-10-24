# tac-sim2real

Experiment code for touch-grounded real-to-sim-to-video. A tactile glove records contact during tabletop manipulation (TouchScale). We rebuild each clip as a physics replay, then fine-tune a video model to generate head-camera video together with the glove's touch, and to estimate touch from video.

The real-to-sim harness that produces the simulation replays (cliprep) lives in a separate repository. This repository holds the fine-tuning experiments.

## Layout

```
experiments/
  finetune/
    INDEX.md                       one row per experiment: date, title, what, status / headline
    <date started>/<short title>/
      README.md                    goal, data split, steps, outputs, results
      code/                        the scripts that experiment ran
```

Each experiment folder is self-contained. To find a run, start from [`experiments/finetune/INDEX.md`](experiments/finetune/INDEX.md).

## Experiments so far

All fine-tuning uses LoRAs on Wan2.2-Fun-A14B-Control.

- **Video + touch LoRA** (`tactile_lora_*`, 10-21 to 10-23): one canvas holds the head-camera video and a touch panel drawn as pixels (superseded by the value head of `2026-10-24`). The runs compare training on mixed real + simulation-generated rows with training on real rows only, and use held-out twin clips.
- **Dual-mode LoRA** (`dual_mode_lora`): one LoRA that generates video and touch from simulation, and also estimates touch from a given video. The inpaint mask chooses the task.
- **Diagnostics:** the loss table for the touch regression head, the inference-setting diagnosis, the depth-control ablation, and touch estimation on released videos.
- **HF25 control comparison** (`2026-10-24/`, branch `for-amir`): three video + touch LoRAs on the 25 pairs of the public release
  [tracyhan816/mug-sim2video-pose-tactile](https://huggingface.co/datasets/tracyhan816/mug-sim2video-pose-tactile), differing only
  in the control video (FHN: scene depth + simulated hand / mug; SVO: simulation render only; CDS: cube-scene depth + render). The
  touch is a value array output by a small head on the diffusion transformer's tokens (no pixel panel), normalised per replay.
  The runbook [`experiments/finetune/2026-10-24/README.md`](experiments/finetune/2026-10-24/README.md) lists the downloads
  (weights, VideoX-Fun commit, data), the dataset build, the launch commands and the evaluation.

## Dependencies

The scripts expect:
- [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) (Apache-2.0) and the Wan2.2-Fun-A14B-Control weights;
- the TouchScale data.

Some scripts patch a copy of the VideoX-Fun LoRA trainer at run time (for example `make_trainer.py`); the trainer itself is not included. Paths in the scripts point at the original cluster layout, so set them for your machine. Model weights, datasets and outputs are not in this repository.
