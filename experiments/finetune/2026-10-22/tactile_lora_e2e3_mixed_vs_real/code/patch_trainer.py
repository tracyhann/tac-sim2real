"""Opt-in --inpaint_first_frame_only for VideoX-Fun scripts/wan2.2_fun/train_control_lora.py: the inpaint mask always
keeps only the first frame (as at inference, where the start image is the only known frame), instead of the random
masks (rectangles, random pixels, first/last frames, ...) that can hand the model parts of the target. Default unchanged."""
import sys
from pathlib import Path

p = Path(sys.argv[1]); t = p.read_text()
if "inpaint_first_frame_only" in t:
    print("already patched"); sys.exit(0)
a = '        "--add_full_ref_image_in_self_attention",'
assert t.count(a) == 1
t = t.replace(a, '        "--inpaint_first_frame_only", action="store_true", help="Inpaint mask keeps only the first frame (image-to-video), no random masks."\n    )\n    parser.add_argument(\n' + a, 1)
b = '                        mask = get_random_mask(new_examples["pixel_values"][-1].size())'
assert t.count(b) == 1
t = t.replace(b, '                        mask = get_random_mask(new_examples["pixel_values"][-1].size(), image_start_only=args.inpaint_first_frame_only)')
p.write_text(t); print("patched")
