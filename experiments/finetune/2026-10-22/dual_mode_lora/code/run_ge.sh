#!/bin/bash
# Dual-mode video + touch LoRA on Wan2.2-Fun-A14B-Control (user 2026-10-22: ONE model that generates video + paired touch from
# sim (G) and estimates touch from any given video (E); the job is picked by the inpaint mask). Dataset: lora_ge/ge_data.py
# (real TouchScale clips + zero-shot sim videos with sim touch, split by episode, actor_01 + twins 55494490 / 4397541c held out).
# Trainer: scripts/wan2.2_fun/train_control_lora_ge.py (a copy of train_control_lora.py + --ge_mask; make_trainer.py).
# Runs (same settings as the f2/h2 LoRAs: rank 64, alpha 32, lr 1e-4 constant, 1088 x 480 x 81, both experts; PASSES epochs):
#   U  both jobs (P(G) 0.5), real + sim rows       G  G only, same rows       E  E only, same rows
#   R  E only, real rows only (what sim data adds to estimation)
# Parts (one pool card each): prep (Depth Anything controls + tiny-model trainer test), <RUN>_high / <RUN>_low, ev_<RUN>_<P>
# (predict_ge.py at epoch P), regr (discriminative baseline). Markers: outputs/R2S2V_mug/done_exp_lora_ge_<part>.
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; X=$R/experiments/R2S2V_mug; LG=$R/outputs/R2S2V_mug/lora_ge; DS=$LG/dataset
V=$R/third_party/VideoX-Fun; PY=$R/envs/videoxfun/bin/python; PA=$R/envs/armA/bin/python
export HF_HOME=$R/cache/huggingface TOKENIZERS_PARALLELISM=false; unset HF_TOKEN
part=$1; PASSES=${PASSES:-8}
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T) part $part"
nrows() { $PA -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" $1; }
train() {   # train <out> <meta> <modes> <boundary> <steps> <save every> <epoch steps> [extra...]
  local OUT=$1 META=$2 MODES=$3 B=$4 STEPS=$5 CK=$6 EP=$7; shift 7; mkdir -p $OUT
  rm -f $OUT/*-compatible_with_comfyui.safetensors                          # never used (predict loads the PEFT files)
  local last=$(ls $OUT | sed -n 's/^checkpoint-\([0-9]*\)\.safetensors$/\1/p' | sort -n | tail -1)
  for f in $(ls $OUT | sed -n 's/^checkpoint-\([0-9]*\)\.safetensors$/\1/p'); do   # keep every 2nd epoch and the newest (resume)
    [ $((f % (2 * EP))) -ne 0 ] && [ "$f" != "$last" ] && rm -f $OUT/checkpoint-$f.safetensors
  done
  cd $V && $PY -m accelerate.commands.launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no scripts/wan2.2_fun/train_control_lora_ge.py \
    --config_path config/wan2.2/wan_civitai_i2v.yaml --pretrained_model_name_or_path ${MODEL:-$R/models/Wan2.2-Fun-A14B-Control} \
    --train_data_dir $DS/ --train_data_meta $META --enable_bucket --fix_sample_size ${FIX:-480 1088} --image_sample_size ${SZ:-1088} --video_sample_size ${SZ:-1088} --token_sample_size ${SZ:-1088} \
    --video_sample_stride 1 --video_sample_n_frames 81 --train_batch_size 1 --video_repeat 1 --gradient_accumulation_steps 1 --dataloader_num_workers 2 \
    --max_train_steps $STEPS --checkpointing_steps $CK --learning_rate 1e-04 --lr_scheduler constant --seed 42 --output_dir $OUT \
    --gradient_checkpointing --mixed_precision bf16 --adam_weight_decay 3e-2 --adam_epsilon 1e-10 --vae_mini_batch 1 --max_grad_norm 0.05 --uniform_sampling \
    --train_mode control_ref --control_ref_image first_frame --add_inpaint_info --add_full_ref_image_in_self_attention \
    --boundary_type $B --rank ${RANK:-64} --network_alpha ${ALPHA:-32} --target_name q,k,v,ffn.0,ffn.2 --use_peft_lora --low_vram --resume_from_checkpoint latest --resume_lora_weights \
    --ge_mask --ge_modes $MODES --ge_log $OUT/ge_log.jsonl --ge_keep_every $((2 * EP)) --ge_epoch_steps $EP "$@"
  rm -f $OUT/*-compatible_with_comfyui.safetensors
}
case $part in
  prep)
    cd $X/lora_ge && $PA ge_depth.py || exit 1                                    # Depth Anything controls for every row
    $PA - <<'EOF' || exit 1
import json; from pathlib import Path
DS = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/outputs/R2S2V_mug/lora_ge/dataset"); miss = []
for r in json.loads((DS / "rows.json").read_text()):
    for k in ("file_path", "control_file_path", "control_alt_file_path"):
        if r.get(k) and not (DS / r[k]).exists(): miss.append(r[k])
print("missing files:", miss); assert not miss
EOF
    $PY $X/lora_ge/test_ge_cpu.py 2 || exit 1
    T=$LG/tiny_model; [ -f $T/low_noise_model/config.json ] || $PY $X/lora_ge/make_tiny.py $T || exit 1
    $PA -c "import json,random; r=json.load(open('$DS/metadata_train.json')); a=[x for x in r if x['kind']=='real' and x.get('control_alt_file_path')][:1]+[x for x in r if x['kind']=='real' and not x.get('control_alt_file_path')][:1]+[x for x in r if x['kind']=='sim'][:2]; json.dump(a, open('$LG/tiny_meta.json','w'), indent=1); print([x['id'] for x in a])"
    cp $LG/tiny_meta.json $DS/tiny_meta.json; rm -rf $LG/tiny_run
    MODEL=$T FIX="240 544" SZ=544 RANK=8 ALPHA=4 train $LG/tiny_run $DS/tiny_meta.json GE high 3 2 4 --ge_dump 8 || exit 1
    MODEL=$T FIX="240 544" SZ=544 RANK=8 ALPHA=4 train $LG/tiny_run $DS/tiny_meta.json GE high 6 2 4 --ge_dump 8 || exit 1
    ls -la $LG/tiny_run; $PY $X/lora_ge/verify_tiny.py $LG/tiny_run $DS/tiny_meta.json || exit 1 ;;
  U_high|U_low|G_high|G_low|E_high|E_low|R_high|R_low)
    RUN=${part%%_*}; B=${part#*_}; META=$DS/metadata_train.json; MODES=GE
    case $RUN in G) MODES=G ;; E) MODES=E ;; R) MODES=E; META=$DS/metadata_train_real.json ;; esac
    N=$(nrows $META); STEPS=$((PASSES * N)); OUT=$LG/${RUN}_$B
    CK=15                                                                          # save every 15 steps (~13 min; pool preemption) AND at
    echo "run $RUN expert $B: $N rows x $PASSES passes = $STEPS steps, save every $CK steps + every epoch, keep every 2 epochs, modes $MODES"   # every epoch boundary
    train $OUT $META $MODES $B $STEPS $CK $N
    ls -la $OUT | tail -4; [ -f $OUT/checkpoint-$STEPS.safetensors ] || { echo "not finished: will resume"; exit 1; } ;;
  ev_*)
    cd $X/lora_ge && $PY predict_ge.py --part $part || exit 1
    RUN=$(echo $part | cut -d_ -f2); P=$(echo $part | cut -d_ -f3)
    if [ $RUN = U ] || [ $RUN = G ]; then                                         # G videos: mug / hand IoU vs the sim silhouettes
      cd $X && bash $X/run_r2s.sh labeladh.py --clips tsmug-55494490 tsmug-4397541c --models wan22fun-a14b-ge${RUN}p${P}gt wan22fun-a14b-ge${RUN}p${P}gn || exit 1
    fi ;;
  diag)   # inference A/B (2026-10-23): U p8 E outputs did not keep the given video (known MAE 35-55) and G did not follow the control
    cd $X/lora_ge; E="E_real_6662a9fd_da E_real_55494490_da"; G="G_real_55494490_t"
    $PY predict_ge.py --part ev_U_8 --cases --train_rows real_23e9d00a real_edf79415 --tag dgU8_train || exit 1          # same settings as the eval, TRAINING rows
    $PY predict_ge.py --part ev_U_8 --cases --train_rows real_23e9d00a --guidance 1 --tag dgU8_train_cfg1 || exit 1
    $PY predict_ge.py --part ev_U_8 --cases $E --guidance 1 --tag dgU8_cfg1 || exit 1
    $PY predict_ge.py --part ev_U_8 --cases $E --guidance 6 --negative empty --tag dgU8_negempty || exit 1
    $PY predict_ge.py --part ev_U_8 --cases $E --no_lora --guidance 1 --tag dgbase_cfg1 || exit 1
    $PY predict_ge.py --part ev_U_8 --cases $G --guidance 6 --tag dgU8_ref || exit 1
    $PY predict_ge.py --part ev_U_8 --cases $G --guidance 1 --no_ref_g --tag dgU8_gcfg1 || exit 1 ;;
  regr)   # loss comparison R0-R6 on the regression head + the main table's discriminative baselines (loss_table.py)
    cd $X/lora_ge && $PY loss_table.py || exit 1 ;;
  regrsp) # sensitivity: R0 / R3 on a softplus output (both collapsed with the sigmoid^2 output)
    cd $X/lora_ge && $PY loss_table.py --extra || exit 1 ;;
  *) echo "unknown part $part"; exit 1 ;;
esac
touch $R/outputs/R2S2V_mug/done_exp_lora_ge_$part; echo "part $part done $(date +%T)"
