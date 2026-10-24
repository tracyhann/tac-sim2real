# Fine-tuning experiments (Wan2.2-Fun-A14B-Control LoRAs: video + touch)

Layout: `<date started>/<short title>/` with `README.md` (goal, data split, steps, outputs, results), `code/` (the scripts),
`out/` (symlinks to the outputs, which never move: outputs/R2S2V_mug/...), `logs/` (symlinks). Authoritative tree on Vision:
`/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/experiments/finetune/`; Mac mirror (code + READMEs only):
`agentic-real2sim/experiments/finetune/`. New experiments: create `<today>/<title>/` with its own code/ and README.md first.

| date | experiment | what | status / headline |
|---|---|---|---|
| 2026-10-21 | [tactile_lora_e1_tagged](2026-10-21/tactile_lora_e1_tagged/README.md) | video + touch LoRA, 6 canvases, tagged real / sim prompts | stopped at ~150; superseded |
| 2026-10-22 | [tactile_lora_e2e3_mixed_vs_real](2026-10-22/tactile_lora_e2e3_mixed_vs_real/README.md) | e2 mixed vs e3 real-only; first-frame mask + resume fixes | done; real rows misaligned (D68) |
| 2026-10-22 | [tactile_lora_f2f3_aligned_rows](2026-10-22/tactile_lora_f2f3_aligned_rows/README.md) | e2 / e3 retrained on time-aligned rows | done; held-out F1@6mm 0.66 / 0.65 |
| 2026-10-22 | [tactile_lora_h2h3_heldout](2026-10-22/tactile_lora_h2h3_heldout/README.md) | train 2 twins, test 2 held-out twins; mixed vs real-only | evals finishing; F1@6mm 0.59 / 0.66 |
| 2026-10-22 | [dual_mode_lora](2026-10-22/dual_mode_lora/README.md) | ONE LoRA for generate (G) + estimate (E); relative touch v3; runs U / G / E / R | running (LIVE code here) |
| 2026-10-23 | [tactile_lora_h2_depth_control](2026-10-23/tactile_lora_h2_depth_control/README.md) | h2 with the clip's Depth Anything depth as control | done; F1@6mm 0.59 |
| 2026-10-23 | [dual_mode_loss_table](2026-10-23/dual_mode_loss_table/README.md) | R0-R6 losses on the E regression head | done; R6 recommended |
| 2026-10-23 | [dual_mode_infer_diag](2026-10-23/dual_mode_infer_diag/README.md) | why the first U eval broke: CFG | done; E at CFG 1, G CFG 6 + ref |
| 2026-10-23 | [estimate_release_videos](2026-10-23/estimate_release_videos/README.md) | touch estimation on the 25 released videos: dual-mode U epoch 8 (job E) and f2 step 600 (full-video mask); existing checkpoints | done; U contact = always-on (release missed), total r 0.12-0.31; f2 paints scene into the panel |

**Live code that is still at its old path.** `experiments/R2S2V_mug/lora_tactile/` (shared by all tactile_lora runs; the
running h3_ev400 and the after_ck* waiters use it): the tactile_lora folders hold snapshots taken 2026-10-23. Once those jobs
end, lora_tactile can move into a live folder here and become a symlink, as lora_ge already is.
Shared trainer (patched in place, opt-in flags): `third_party/VideoX-Fun/scripts/wan2.2_fun/train_control_lora.py`;
dual-mode copy: `train_control_lora_ge.py` + `ge_mask.py` next to it.
