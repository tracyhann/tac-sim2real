#!/bin/bash
# Trainer smoke test on the tiny stand-in network (lora_ge/tiny_model: 2 blocks, dim 256, real VAE / T5), ONE pool card, ~10 min;
# NOT part of the experiment. Run 1: 3 steps with the touch head, checkpoint at 2; run 2: resumed to step 4 (reloads
# tactile_head-2.pt). Checks: checkpoint-{2,3,4}.safetensors + tactile_head-{2,3,4}.pt exist, the resumed run logged the head
# load, every step's tactile loss is finite, the head file loads into hf25_patch.TactileHead(256).
#   bash launch_smoke.sh   (pool_retry loop, marker done_exp_hf25_smoke) or directly on a card: bash smoke_tiny.sh
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; C=$R/experiments/finetune/2026-10-24/_common/code; O=$R/outputs/R2S2V_mug/hf25
V=$R/third_party/VideoX-Fun; PY=$R/envs/videoxfun/bin/python; T=$R/outputs/R2S2V_mug/lora_ge/tiny_model; DS=$O/dataset; RUN=$O/smoke_tiny
export HF_HOME=$R/cache/huggingface TOKENIZERS_PARALLELISM=false; unset HF_TOKEN
set -eo pipefail
[ -f $T/low_noise_model/config.json ] || $PY $R/experiments/finetune/2026-10-22/dual_mode_lora/code/make_tiny.py $T
$PY $C/make_trainer_hf25.py $V
$PY -c "import json; r=json.load(open('$DS/metadata_SVO.json')); json.dump(r[:2], open('$DS/smoke_meta.json','w'), indent=1); print([x['id'] for x in r[:2]])"
mkdir -p $O/smoke_logs; [ -f $RUN/log.txt ] && cp $RUN/log.txt $O/smoke_logs/log_$(date +%m%d_%H%M%S).txt   # keep the previous attempt's trainer log
rm -rf $RUN; mkdir -p $RUN
train() {  # $1 = max steps
  cd $V && $PY -m accelerate.commands.launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no scripts/wan2.2_fun/train_control_lora_hf25.py \
    --config_path config/wan2.2/wan_civitai_i2v.yaml --pretrained_model_name_or_path $T \
    --train_data_dir $DS/ --train_data_meta $DS/smoke_meta.json \
    --enable_bucket --fix_sample_size 240 416 --image_sample_size 416 --video_sample_size 416 --token_sample_size 416 \
    --video_sample_stride 1 --video_sample_n_frames 81 --train_batch_size 1 --video_repeat 1 --gradient_accumulation_steps 1 --dataloader_num_workers 2 \
    --max_train_steps $1 --checkpointing_steps 2 --learning_rate 1e-04 --lr_scheduler constant --seed 42 --output_dir $RUN \
    --gradient_checkpointing --mixed_precision bf16 --adam_weight_decay 3e-2 --adam_epsilon 1e-10 --vae_mini_batch 1 --max_grad_norm 0.05 --uniform_sampling \
    --train_mode control_ref --control_ref_image first_frame --add_inpaint_info --add_full_ref_image_in_self_attention --inpaint_first_frame_only \
    --boundary_type low --rank 8 --network_alpha 4 --target_name q,k,v,ffn.0,ffn.2 --use_peft_lora --low_vram --resume_from_checkpoint latest --resume_lora_weights \
    --tactile_head --tactile_lambda 1.0 > $RUN/log_$1.txt 2>&1 || { echo "trainer failed (max steps $1); log tail:"; grep -v "^\s*$" $RUN/log_$1.txt | tail -40 | cut -c1-240; exit 1; }
  cat $RUN/log_$1.txt >> $RUN/log.txt; grep -E "tactile head|resume_lora|saved tactile|saving checkpoint" $RUN/log_$1.txt | tail -8
}
train 3
train 4
ls -la $RUN
for n in 2 3 4; do [ -f $RUN/checkpoint-$n.safetensors ] && [ -f $RUN/tactile_head-$n.pt ] || { echo "missing pair $n"; exit 1; }; done
grep -q "tactile head .* loaded" $RUN/log.txt || { echo "resume did not reload the head"; exit 1; }
$PY - <<EOF
import re, torch, sys
sys.path.insert(0, "$C")
from hf25_patch import TactileHead
sd = torch.load("$RUN/tactile_head-4.pt", map_location="cpu"); h = TactileHead(256); h.load_state_dict(sd); print("head loads:", len(sd), "tensors")
log = open("$RUN/log.txt").read(); tl = [float(x) for x in re.findall(r"tactile=([0-9.eE+-]+|nan)", log) if x != "nan"]
print("tactile losses logged:", len(tl), tl[:6]); assert tl and all(t == t and t < 50 for t in tl), "tactile loss missing or not finite"   # v2 loss includes the total-force term (units of S)
EOF
touch $R/outputs/R2S2V_mug/done_exp_hf25_smoke   # the pool_retry loop of launch_smoke.sh stops on this marker
echo "smoke test passed $(date +%T)"
