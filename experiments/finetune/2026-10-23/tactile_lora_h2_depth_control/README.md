# h2 with a Depth Anything control (2026-10-23)

**Goal (user 10-23).** Generate video + touch with h2 (step 600) from the REAL clip's Depth Anything V2 depth as control
(no sim), real frame 0 + its touch as start, held-out clips 55494490 / 4397541c, seeds 0 / 1. Tag `...tacth2c600da-real`.
**Code.** `code/` = snapshots of `h2_depthctrl.sbatch`, `h2_depthctrl_score.sbatch`, `predict_tactile_da.py`, `ho_table.py`;
live in `experiments/R2S2V_mug/lora_tactile/`.
**Results (heldout_table.json row "h2 step 600 · real frame 0 · Depth Anything control, no sim").** contact F1 @ 6 mm 0.59
(the same as with the sim control).
**Outputs.** videos `outputs/A_baseline/gen/wan22fun-a14b-tacth2c600da-real/`, scores in labeladh / eval dirs, table row in
outputs/R2S2V_mug/lora_tactile/heldout_table.json.
