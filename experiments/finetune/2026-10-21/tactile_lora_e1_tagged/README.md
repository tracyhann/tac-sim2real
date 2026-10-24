# e1 · video + touch LoRA, tagged prompts (2026-10-21)

**Goal.** Overfit Wan2.2-Fun-A14B-Control so it outputs the glove's touch with the video, on one 1088 x 480 canvas
(video 832 x 480 | touch panel 256 x 480, panel v1: 3.0 N per taxel, 30 N bars, absolute newtons).
**Data.** 6 canvases from 2 clips (edf79415, 53780879): real video + real touch, and two zero-shot generated videos + sim
touch per clip; prompt tag "Real recording." / "Simulated touch."; control = frame-0 scene depth + shadowless sim hand / mug.
Built on the Mac by `scene_mug/r2s2v/lora_tactile_data.py` (snapshot in `code/mac/`).
**Training.** Both experts, rank 64 / alpha 32, lr 1e-4, 600 steps planned; VideoX-Fun random inpaint masks (later found to
leak the target). Stopped at step ~150 when the user asked for mixed training without tags (-> e2).
**Code.** `code/` = snapshots (2026-10-23) of the files this run used; live code: `experiments/R2S2V_mug/lora_tactile/`
(shared by every tactile_lora run). Parts `e1_high` / `e1_low` / `e1_eval` of `run_lora.sh`.
**Outputs.** `out/` -> outputs/R2S2V_mug/lora_tactile/{dataset, e1_high, e1_low} (checkpoint-100 kept).
**Status.** Stopped. Superseded by e2 / e3. Also affected by the D68 real-row bug (see e2e3).
