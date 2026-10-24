#!/bin/bash -l
#SBATCH -J r2s2r-ge-compose
#SBATCH -p def
#SBATCH --qos=standard
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/logs/lora_ge/compose_%j.out
# CPU job: side-by-side videos of every held-out case (compose_ge.py, epoch 8) + small web copies for the report page.
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r; V=$R/outputs/R2S2V_mug/lora_ge/videos
cd $R/experiments/R2S2V_mug/lora_ge && $R/envs/armA/bin/python compose_ge.py --epoch 8 || exit 1
mkdir -p $V/web && for f in $V/*.mp4; do ffmpeg -loglevel error -y -i $f -vf "scale=1280:-2" -c:v libx264 -crf 30 -preset slow -pix_fmt yuv420p -movflags +faststart $V/web/$(basename $f); done
ls -la $V/web; du -sh $V/web
