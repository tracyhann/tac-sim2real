#!/bin/bash
# The 25 training pairs = the public release https://huggingface.co/datasets/tracyhan816/mug-sim2video-pose-tactile, pinned at
# commit 6cf129c (2026-10-23, 25 videos: r6 head 6, r8b head + top photo 12, r10 user frames 7; per folder video.mp4 (832 x 480,
# 81 frames, 16 fps), tactile.npz (normal_N (81, 880) = the left glove's 880 channels, t_video), hand_pose.npz). Provenance only:
# training uses the prepared set tracyhan816/mug-video-touch-lora-hf25 (setup.sh). Root = env R2S2R.
#   bash fetch_hf.sh   -> data/mug-sim2video-pose-tactile/
R=${R2S2R:-/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r}; D=$R/data/mug-sim2video-pose-tactile
HF=$(ls $R/envs/videoxfun/bin/hf $R/envs/armA/bin/hf 2>/dev/null | head -1); HF=${HF:-hf}
REV=6cf129c769161eec04b5566a903163913b75ca33
export HF_HOME=$R/cache/huggingface; unset HF_TOKEN
$HF download tracyhan816/mug-sim2video-pose-tactile --repo-type dataset --revision $REV --local-dir $D   # the whole repo (42 MB); an --include list dropped video.mp4 on 10-24
echo $REV > $D/REVISION
n=$(ls $D/r*_*/video.mp4 2>/dev/null | wc -l); m=$(ls $D/r*_*/tactile.npz 2>/dev/null | wc -l)
echo "$n videos, $m tactile files in $D"; [ "$n" -eq 25 ] && [ "$m" -eq 25 ] || { echo "expected 25 + 25"; exit 1; }
