#!/bin/bash -l
#SBATCH -J r2s2r-ge-webify
#SBATCH -p def
#SBATCH --qos=standard
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -o /scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/logs/lora_ge/webify_%j.out
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; cd $R/experiments/R2S2V_mug/lora_ge && $R/envs/armA/bin/python webify.py
