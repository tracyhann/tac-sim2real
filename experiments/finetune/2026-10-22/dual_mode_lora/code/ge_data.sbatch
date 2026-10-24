#!/bin/bash -l
#SBATCH -J r2s2r-ge-data
#SBATCH -p def
#SBATCH --qos=standard
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/logs/lora_ge/ge_data_%j.out
# CPU job: (re)build the dual-mode LoRA dataset with the RELATIVE touch panel v3 (caps from scale.json measured on the v2
# build), re-measure the scale / clip fractions, marker on success.
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; M=$R/outputs/R2S2V_mug/done_exp_lora_ge_data; rm -f $M
cd $R/experiments/R2S2V_mug/lora_ge && $R/envs/armA/bin/python make_panel_v3.py && $R/envs/armA/bin/python ge_data.py && \
  cp $R/outputs/R2S2V_mug/lora_ge/dataset/scale.json $R/outputs/R2S2V_mug/lora_ge/dataset/scale_v2build.json && $R/envs/armA/bin/python ge_scale.py && touch $M
