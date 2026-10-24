#!/bin/bash
# HF25 video + touch LoRAs on Wan2.2-Fun-A14B-Control (user 2026-10-24): three arms, same 25 HF pairs (video 832 x 480 + the
# glove's 880 touch values per frame, normalised F / S), same prompts, same trainer settings as the F / H runs
# (lora_tactile/run_lora.sh: rank 64 / alpha 32, lr 1e-4 constant, first-frame-only inpaint mask, control_ref + first_frame,
# resume from the newest LoRA weights), 500 steps per expert; the arms differ ONLY in the control (build_hf25.py):
#   FHN  own generation control (scene depth + sim pixels, the F / H format)   SVO  sim render only   CDS  cube depth + sim render
# Touch = a value head (hf25_patch.py) on the LOW-noise expert's last-layer tokens, trained with its LoRA (--tactile_head); the
# high-noise expert is a plain video LoRA. Trainer = train_control_lora_hf25.py (make_trainer_hf25.py copy of the shared one).
# Parts (one pool card each; launch.sh):
#   <ARM>_high / <ARM>_low   train one expert's LoRA (low: + the touch head); checkpoint every 25 steps (~20 min; ~45 s/step:
#                            500 steps ~ 6.5 h, so the 4 h pool allocation ends mid-run and the step is killed; pool_retry
#                            relaunches it and the trainer resumes from the newest checkpoint + head with a fresh Adam, as the
#                            F / H runs did; each kill costs < 25 steps + the model load); done = checkpoint-500 (+ tactile_head-500.pt
#                            for low); checkpoints that are not multiples of 100 are pruned once a newer one exists
#   ev_<ARM>_<CK>            predict_hf25.py (video + touch for test_cases.json), regr_hf25.py (external R6 estimator on the
#                            generated videos), labeladh.py (mug / hand IoU vs sim), score_hf25.py (every arm / checkpoint)
R=${R2S2R:-/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r}   # workspace root (see 2026-10-24/README.md for the layout); override R2S2R / PY / PA elsewhere
X=$R/experiments/R2S2V_mug; C=$R/experiments/finetune/2026-10-24/_common/code
O=$R/outputs/R2S2V_mug/hf25; DS=$O/dataset; V=$R/third_party/VideoX-Fun; PY=${PY:-$R/envs/videoxfun/bin/python}; PA=${PA:-$R/envs/armA/bin/python}
export HF_HOME=$R/cache/huggingface TOKENIZERS_PARALLELISM=false; unset HF_TOKEN
STEPS=500; part=$1
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T) part $part"
case $part in
  FHN_high|FHN_low|SVO_high|SVO_low|CDS_high|CDS_low)
    ARM=${part%_*}; B=${part#*_}; OUT=$O/${ARM}_$B; mkdir -p $OUT
    [ -f $DS/metadata_$ARM.json ] || { echo "no $DS/metadata_$ARM.json: run build_hf25.py first"; exit 1; }
    TR=$V/scripts/wan2.2_fun/train_control_lora_hf25.py; [ -f $TR ] && [ -f $V/scripts/wan2.2_fun/hf25_patch.py ] || $PY $C/make_trainer_hf25.py $V || exit 1
    EXTRA=; [ "$B" = low ] && EXTRA="--tactile_head --tactile_lambda 1.0"
    rm -f $OUT/checkpoint-*-compatible_with_comfyui.safetensors
    if [ "$B" = low ]; then                       # a resume needs the head of the newest checkpoint: drop a checkpoint whose head is missing
      for f in $OUT/checkpoint-*.safetensors; do [ -f "$f" ] || continue; n=${f##*checkpoint-}; n=${n%.safetensors}
        [ -f $OUT/tactile_head-$n.pt ] || { echo "checkpoint-$n without its head: removed"; rm -f "$f"; }; done
    fi
    cd $V && $PY -m accelerate.commands.launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no scripts/wan2.2_fun/train_control_lora_hf25.py \
      --config_path config/wan2.2/wan_civitai_i2v.yaml --pretrained_model_name_or_path $R/models/Wan2.2-Fun-A14B-Control \
      --train_data_dir $DS/ --train_data_meta $DS/metadata_$ARM.json \
      --enable_bucket --fix_sample_size 480 832 --image_sample_size 832 --video_sample_size 832 --token_sample_size 832 \
      --video_sample_stride 1 --video_sample_n_frames 81 --train_batch_size 1 --video_repeat 1 --gradient_accumulation_steps 1 --dataloader_num_workers 2 \
      --max_train_steps $STEPS --checkpointing_steps 25 --learning_rate 1e-04 --lr_scheduler constant --seed 42 --output_dir $OUT \
      --gradient_checkpointing --mixed_precision bf16 --adam_weight_decay 3e-2 --adam_epsilon 1e-10 --vae_mini_batch 1 --max_grad_norm 0.05 --uniform_sampling \
      --train_mode control_ref --control_ref_image first_frame --add_inpaint_info --add_full_ref_image_in_self_attention --inpaint_first_frame_only \
      --boundary_type $B --rank 64 --network_alpha 32 --target_name q,k,v,ffn.0,ffn.2 --use_peft_lora --low_vram --resume_from_checkpoint latest --resume_lora_weights $EXTRA
    rm -f $OUT/checkpoint-*-compatible_with_comfyui.safetensors          # the trainer's ComfyUI copies (486 MB each; the lab disk is at 99 %)
    last=$(ls $OUT | sed -n 's/^checkpoint-\([0-9]*\)\.safetensors$/\1/p' | sort -n | tail -1)
    for f in $OUT/checkpoint-*.safetensors $OUT/tactile_head-*.pt; do [ -f "$f" ] || continue; n=${f##*-}; n=${n%.*}
      [ -n "$last" ] && [ $((n % 100)) -ne 0 ] && [ "$n" -lt "$last" ] && rm -f "$f"; done
    ls -la $OUT | tail -5
    [ -f $OUT/checkpoint-$STEPS.safetensors ] || { echo "not finished (newest checkpoint ${last:-none}): will resume"; exit 1; }
    [ "$B" = high ] || [ -f $OUT/tactile_head-$STEPS.pt ] || { echo "checkpoint-$STEPS without its head"; exit 1; } ;;
  ev_*)
    ARM=$(echo $part | cut -d_ -f2); CK=$(echo $part | cut -d_ -f3); arm=$(echo $ARM | tr A-Z a-z)
    for B in low high; do [ -f $O/${ARM}_$B/checkpoint-$CK.safetensors ] || { echo "no ${ARM}_$B checkpoint-$CK"; exit 1; }; done
    [ -f $O/${ARM}_low/tactile_head-$CK.pt ] || { echo "no ${ARM}_low tactile_head-$CK.pt"; exit 1; }
    cd $X && $PY $C/predict_hf25.py --arm $ARM --ckpt $CK || exit 1
    $PY $C/regr_hf25.py --arm $ARM --ckpt $CK || exit 1
    bash $X/run_r2s.sh labeladh.py --clips tsmug-55494490 tsmug-4397541c --models wan22fun-a14b-hf25${arm}c$CK-real wan22fun-a14b-hf25${arm}c$CK-sim
    $PA $C/score_hf25.py ;;
  *) echo "unknown part $part"; exit 2 ;;
esac
touch $R/outputs/R2S2V_mug/done_exp_hf25_$part; echo "part $part done $(date +%T)"
