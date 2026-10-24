#!/bin/bash
# Video + tactile LoRA on Wan2.2-Fun-A14B-Control (user 2026-10-22), overfit on 6 canvases. e1 = prompts tagged "Real
# recording." / "Simulated touch." (stopped at step ~150: the user wants mixed training); e2 = the same 6, one prompt, no tag (2 clips x {real video + real
# tactile, 2 generated videos + sim tactile}; lora_tactile/README.md); e3 = ablation for e2 (user 2026-10-22): only the 2 real
# pairs (real video + real tactile, 1/3 of the data), same prompt and settings, so e2 vs e3 shows what the sim-generated videos add.
# Parts (one pool card each):
#   e1_high / e1_low  train the high- / low-noise expert's LoRA (rank 64, lr 1e-4, 600 steps, checkpoint every 50 (50 s/step: 100 steps = 83 min);
#                     pool allocations are 4 h, so every retry resumes from the latest checkpoint; done = checkpoint-600)
#   e1_eval           generate with both LoRAs at steps 300 and 600, decode the tactile panel, score vs real and sim
#                     tactile; the video half is scored by evaluate.py / adherence.py / labeladh.py
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug; LT=$R/outputs/R2S2V_mug/lora_tactile; V=$R/third_party/VideoX-Fun; PY=$R/envs/videoxfun/bin/python
A=$X/../A_baseline; B_=$X/../B_sim2wm
export HF_HOME=$R/cache/huggingface TOKENIZERS_PARALLELISM=false; unset HF_TOKEN
part=$1; RUN=${part%%_*}; META=metadata.json; [ "$RUN" = e2 ] && META=metadata_mixed.json   # e2 (user 2026-10-22): mixed training, no real / sim tag
[ "$RUN" = e3 ] && META=metadata_real.json   # e3: real pairs only
DS=$LT/dataset   # h2 / h3 = held-out test (2026-10-22): trained on edf79415 + 53780879 (the e-runs' clips, with time-aligned real rows,
# scene_mug LESSONS D68, and the star control views5/head), scored on 55494490 + 4397541c, never trained on (f8f7ed78 / 9811b8f8 were
# planned as more clips; their automatic twins fail the hold test; dataset_ho/split.json). h2 = mixed (real + sim rows, 6 samples),
# h3 = real rows only (2): h3 step 200 = h2 step 600 in passes per sample (100), h3 step 600 = equal steps.
# Eval parts h2_ev<ck> / h3_ev<ck> (queued by after_ck.sh).
case $RUN in h2) DS=$LT/dataset_ho; META=metadata_mixed.json ;; h3) DS=$LT/dataset_ho; META=metadata_real.json ;; esac
# f2 / f3 = e2 / e3 retrained with ONLY the real-row fix (user 2026-10-22: "fix it and run the fixed method"): dataset_fix = dataset
# rebuilt by r2s2v/lora_tactile_data.py with time-aligned real rows (D68); control, sim rows, tactile and prompts are identical.
case $RUN in f2) DS=$LT/dataset_fix; META=metadata_mixed.json ;; f3) DS=$LT/dataset_fix; META=metadata_real.json ;; esac
CKS="300 600"; [ "$RUN" = e3 ] && CKS="200 300 600"   # e3 step 200 = 100 passes over its 2 samples, like e2 step 600 over 6
STEPS=600; [ "$RUN" = h3 ] && STEPS=400   # h3: stop at the equal-passes point (100 passes = h2 step 600); cut 10-22 for the 4-card cap   # 10-23 (user): h3 continued 200 -> 400 for the equal-step comparison with h2 at 200 and 400
FF=--inpaint_first_frame_only; [ "$RUN" = e1 ] && FF=   # e2/e3 (restarted 10-22 ~02:30): only the first frame is known, as at inference
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T) part $part"
case $part in
  e1_high|e1_low|e2_high|e2_low|e3_high|e3_low|h2_high|h2_low|h3_high|h3_low|f2_high|f2_low|f3_high|f3_low)
    B=${part#*_}; OUT=$LT/${RUN}_$B; mkdir -p $OUT
    if [ "$B" = high ] && [ ! -f $LT/dataset/vae_check.json ]; then $PY $X/lora_tactile/vae_check.py $LT/dataset; fi
    cd $V && $PY -m accelerate.commands.launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no scripts/wan2.2_fun/train_control_lora.py \
      --config_path config/wan2.2/wan_civitai_i2v.yaml --pretrained_model_name_or_path $R/models/Wan2.2-Fun-A14B-Control \
      --train_data_dir $DS/ --train_data_meta $DS/$META \
      --enable_bucket --fix_sample_size 480 1088 --image_sample_size 1088 --video_sample_size 1088 --token_sample_size 1088 \
      --video_sample_stride 1 --video_sample_n_frames 81 --train_batch_size 1 --video_repeat 1 --gradient_accumulation_steps 1 --dataloader_num_workers 2 \
      --max_train_steps $STEPS --checkpointing_steps 50 --learning_rate 1e-04 --lr_scheduler constant --seed 42 --output_dir $OUT \
      --gradient_checkpointing --mixed_precision bf16 --adam_weight_decay 3e-2 --adam_epsilon 1e-10 --vae_mini_batch 1 --max_grad_norm 0.05 --uniform_sampling \
      --train_mode control_ref --control_ref_image first_frame --add_inpaint_info --add_full_ref_image_in_self_attention $FF \
      --boundary_type $B --rank 64 --network_alpha 32 --target_name q,k,v,ffn.0,ffn.2 --use_peft_lora --low_vram --resume_from_checkpoint latest --resume_lora_weights
    ls -la $OUT | tail -5
    [ -f $OUT/checkpoint-$STEPS.safetensors ] || { echo "not finished: will resume"; exit 1; } ;;
  e1_eval|e2_eval|e3_eval)
    cd $X
    EMETA=$META; [ "$RUN" = e3 ] && EMETA=metadata_mixed.json   # e3 trains on the real rows only, but is scored from gen_s0 starts too (same prompt / control)
    for ck in $CKS; do $PY lora_tactile/predict_tactile.py --run $RUN --meta $EMETA --ckpt $ck --clips tsmug-edf79415 tsmug-53780879 --sources gt gen_s0 --seeds 0; done
    M=$(for ck in $CKS; do for s in real sim; do echo -n "wan22fun-a14b-tact${RUN}c$ck-$s "; done; done)
    C="tsmug-edf79415 tsmug-53780879"
    bash $X/run_r2s.sh $A/evaluate.py --clips $C --models $M; bash $X/run_r2s.sh $B_/adherence.py --clips $C --models $M; bash $X/run_r2s.sh labeladh.py --clips $C --models $M ;;
  h2_ev*|h3_ev*|f2_hoev*|f3_hoev*)   # f*_hoev (user 10-23): the f-runs on the same held-out clips, starts, control and seeds as h
    CK=${part##*ev}; cd $X; HT="tsmug-55494490 tsmug-4397541c"; SEEDS=0; case $part in h2_ev600|h3_ev200|f2_hoev600|f3_hoev200|h2_ev200|h2_ev400|h3_ev400) SEEDS="0 1" ;; esac   # the equal-passes pair: 2 seeds
    $PY fun_zeroshot.py --model a14b --clips $HT --seeds 0 --control views5/head/fun_ctrl.mp4 --start views5/head/cond.png --ref views5/head/cond.png --tag wan22fun-a14b-head-c5   # held-out generated starts = baseline (c); skips existing
    $PY lora_tactile/ho_starts.py   # their start canvases + metadata_test rows
    if [ ! -f $R/outputs/R2S2V_mug/done_exp_ho_hob_c ]; then for s in $A/evaluate.py $B_/adherence.py labeladh.py; do bash $X/run_r2s.sh $s --clips $HT --models wan22fun-a14b-head-c5 sim-render; done; touch $R/outputs/R2S2V_mug/done_exp_ho_hob_c; fi   # baseline (c) scored once
    $PY lora_tactile/predict_tactile.py --run $RUN --ds dataset_ho --meta metadata_test.json --ckpt $CK --clips $HT --sources gt gen_s0 --seeds $SEEDS
    M="wan22fun-a14b-tact${RUN}c$CK-real wan22fun-a14b-tact${RUN}c$CK-sim"
    bash $X/run_r2s.sh $A/evaluate.py --clips $HT --models $M; bash $X/run_r2s.sh $B_/adherence.py --clips $HT --models $M; bash $X/run_r2s.sh labeladh.py --clips $HT --models $M ;;
  e2_ev*|e3_ev*|f2_ev*|f3_ev*)   # in-sample eval at one checkpoint (queued by after_ck.sh): the wrong (e) vs fixed (f) comparison, same starts / control
    CK=${part#*_ev}; cd $X; C="tsmug-edf79415 tsmug-53780879"; DSN=dataset; case $RUN in f2|f3) DSN=dataset_fix ;; esac
    $PY lora_tactile/predict_tactile.py --run $RUN --ds $DSN --meta metadata_mixed.json --ckpt $CK --clips $C --sources gt gen_s0 --seeds 0
    M="wan22fun-a14b-tact${RUN}c$CK-real wan22fun-a14b-tact${RUN}c$CK-sim"
    bash $X/run_r2s.sh $A/evaluate.py --clips $C --models $M; bash $X/run_r2s.sh $B_/adherence.py --clips $C --models $M; bash $X/run_r2s.sh labeladh.py --clips $C --models $M ;;
esac
touch $R/outputs/R2S2V_mug/done_exp_lora_$part; echo "part $part done $(date +%T)"
