#!/bin/bash
# Mac (scene_mug): cube-scene controls for the two held-out twins, so the SVO and CDS arms can be tested on them in their own
# control format. The training rows already have them (round 8b views9_<variant>/head); the twins only had views5 (fused-twin
# depth). Per twin:
#   gt_variants.py prim   G1 primitive scene (desk slab, back wall, one box per object cluster) -> wm/gtvar/prim/prim.json
#   views_prim.py head    -> wm/views9_recorded/head/{fun_ctrl.mp4 (CDS: cube depth + sim hand / mug / coaster pixels),
#                            sim.mp4 (SVO: the sim hand / mug / coaster alone, on black), seg_simonly.npy, cam.json, cond.png}
# then uploads that head folder to Vision outputs/B_sim2wm/sim/tsmug-<ep8>/views9_recorded/head/ (md5-checked).
#   bash twin_cubes.sh            (from anywhere; ~2 min per twin)
set -eo pipefail
SM=/Users/grandpa/phai/agentic-real2sim/experiments/scene_mug; cd $SM
R=/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r
for clip in 55494490_auto_mano 4397541c_auto_mano; do
  ep8=${clip%%_*}; cfg=configs/clips/$clip.yaml; o=runs/clips/$clip/wm/views9_recorded/head
  [ -f runs/clips/$clip/wm/gtvar/prim/prim.json ] || .venv/bin/python r2s2v/gt_variants.py $cfg prim
  [ -f $o/fun_ctrl.mp4 ] && [ -f $o/sim.mp4 ] || .venv/bin/python r2s2v/views_prim.py $cfg --sim runs/clips/$clip/sim --out views9_recorded \
      --photo runs/clips/r2s2v_blocks/photo_top.png head
  D=$R/outputs/B_sim2wm/sim/tsmug-$ep8/views9_recorded/head
  tar -C $o -cf - . | ssh linux3090 "ssh -o BatchMode=yes vision 'mkdir -p $D && tar -C $D -xf -'"
  l=$(cd $o && md5 -q fun_ctrl.mp4 sim.mp4 | tr '\n' ' ')
  r=$(ssh linux3090 "ssh -o BatchMode=yes vision 'cd $D && md5sum fun_ctrl.mp4 sim.mp4'" | awk '{printf "%s ", $1}')
  [ "$l" = "$r" ] && echo "$clip uploaded ($D)" || { echo "$clip upload md5 mismatch: $l vs $r"; exit 1; }
done
