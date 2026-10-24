"""Write scripts/wan2.2_fun/train_control_lora_ge.py: a COPY of VideoX-Fun's train_control_lora.py (already carrying the
opt-in --inpaint_first_frame_only and --resume_lora_weights patches, lora_tactile/patch_trainer*.py) with the opt-in
dual-mode mask (ge_mask.py, copied next to it). The shared train_control_lora.py is not touched: the f2/f3/h2/h3 pool
loops relaunch it after every preemption.
  --ge_mask                 per sample: job G or E (ge_mask.py), mask from the job, control from the job
  --ge_modes GE|G|E         jobs this run may draw (U = GE, G-only = G, E-only = E)
  --ge_p_gen 0.5            P(G) when both jobs are allowed
  --ge_p_touch0 0.5         in G: P(frame 0's touch panel is known)
  --ge_p_sim_ctrl 0.5       in E: P(sim control) when the row also has a Depth Anything control
  --ge_panel_x0 832 --ge_canvas_w 1088   the panel starts at x0 of a canvas this wide (scaled to the training size)
  --ge_log FILE             one json line per step: the batch's jobs / controls / rows and the loss
  --ge_dump N               save the first N batches' masks, masked inputs and jobs to <output_dir>/ge_dump/ (unit test)
  --ge_epoch_steps N        also save at every epoch boundary (N = steps per epoch)
  --ge_keep_every K         periodic saves keep only checkpoints at multiples of K steps + the newest (no ComfyUI copies)
  --ge_panel_loss_weight w --ge_panel_layout F.npy   optional loss weight on the panel's taxel / bar latent cells (off: w = 1)
With --resume_lora_weights the batches already used in the current epoch are skipped (the epoch order is seeded by
seed + epoch), so a run stopped and resumed by the pool still sees every sample exactly once per epoch.
  python make_trainer.py <VideoX-Fun dir>"""
import shutil
import sys
from pathlib import Path

V = Path(sys.argv[1]); src = V / "scripts/wan2.2_fun/train_control_lora.py"; dst = V / "scripts/wan2.2_fun/train_control_lora_ge.py"
t = src.read_text()
assert "resume_lora_weights" in t and "inpaint_first_frame_only" in t, "the base trainer lacks the earlier patches"


def sub(old, new, n=1):
    global t
    assert t.count(old) == n, (old[:80], t.count(old))
    t = t.replace(old, new)


sub('        "--inpaint_first_frame_only", action="store_true",',
    '''        "--ge_mask", action="store_true", help="Dual-mode structured masks (ge_mask.py): G = generate from frame 0, E = estimate the touch panel from the video."
    )
    parser.add_argument("--ge_modes", type=str, default="GE")
    parser.add_argument("--ge_p_gen", type=float, default=0.5)
    parser.add_argument("--ge_p_touch0", type=float, default=0.5)
    parser.add_argument("--ge_p_sim_ctrl", type=float, default=0.5)
    parser.add_argument("--ge_panel_x0", type=int, default=832)
    parser.add_argument("--ge_canvas_w", type=int, default=1088)
    parser.add_argument("--ge_log", type=str, default=None)
    parser.add_argument("--ge_dump", type=int, default=0)
    parser.add_argument("--ge_keep_every", type=int, default=0)
    parser.add_argument("--ge_epoch_steps", type=int, default=0)
    parser.add_argument("--ge_panel_loss_weight", type=float, default=1.0)
    parser.add_argument("--ge_panel_layout", type=str, default=None)
    parser.add_argument(
        "--inpaint_first_frame_only", action="store_true",''')
# dataset: convert the instance in place, before the sampler is built on it
sub('''        enable_camera_info=args.train_mode == "control_camera_ref"
    )

    def worker_init_fn(_seed):''', '''        enable_camera_info=args.train_mode == "control_camera_ref"
    )
    if args.ge_mask:
        assert args.enable_bucket and args.add_inpaint_info and args.control_ref_image == "first_frame", "--ge_mask needs --enable_bucket --add_inpaint_info --control_ref_image first_frame"
        from ge_mask import GEDataset, make_mask as ge_make_mask
        train_dataset.__class__ = GEDataset; train_dataset.ge_setup(args)
        print(f"ge_mask: modes {args.ge_modes}, P(G) {args.ge_p_gen}, P(touch0 | G) {args.ge_p_touch0}, P(sim ctrl | E, both) {args.ge_p_sim_ctrl}, {len(train_dataset.dataset)} rows")

    def worker_init_fn(_seed):''')
sub('''            if args.add_inpaint_info:
                new_examples["mask_pixel_values"] = []
                new_examples["mask"] = []
                new_examples["clip_pixel_values"] = []''', '''            if args.add_inpaint_info:
                new_examples["mask_pixel_values"] = []
                new_examples["mask"] = []
                new_examples["clip_pixel_values"] = []
            if args.ge_mask:
                new_examples["ge"] = []''')
sub('''                        mask = get_random_mask(new_examples["pixel_values"][-1].size(), image_start_only=args.inpaint_first_frame_only)''',
    '''                        if args.ge_mask:
                            mask = ge_make_mask(new_examples["pixel_values"][-1].size(), example["ge"], args); new_examples["ge"].append(example["ge"])
                        else:
                            mask = get_random_mask(new_examples["pixel_values"][-1].size(), image_start_only=args.inpaint_first_frame_only)''')
# resume: skip the batches the stopped run already used in this epoch
sub('''        for step, batch in enumerate(train_dataloader):
            # Data batch sanity check''', '''        ge_skip = (lw_step - first_epoch * num_update_steps_per_epoch) if (args.ge_mask and lw_step is not None and epoch == first_epoch) else 0
        if ge_skip:
            accelerator.print(f"ge_mask resume: skipping the first {ge_skip} batches of epoch {epoch} (already trained)")
        for step, batch in enumerate(train_dataloader):
            if step < ge_skip:
                continue
            if args.ge_mask and args.ge_dump and accelerator.is_main_process and global_step < args.ge_dump:
                os.makedirs(os.path.join(args.output_dir, "ge_dump"), exist_ok=True)
                torch.save({k: batch[k] for k in ("mask", "mask_pixel_values", "pixel_values", "control_pixel_values", "ge", "text")},
                           os.path.join(args.output_dir, "ge_dump", f"batch_{global_step:04d}.pt"))
            # Data batch sanity check''')
sub('''                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()''',
    '''                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                if args.ge_mask and args.ge_log and accelerator.is_main_process:
                    import json as _gej
                    with open(args.ge_log, "a") as _gef:
                        _gef.write(_gej.dumps(dict(step=global_step + 1, epoch=epoch, ge=[{k: v for k, v in g.items() if k != "control"} for g in batch["ge"]],
                                                   loss=float(avg_loss.item()))) + "\\n")''')
# periodic saves: no ComfyUI copy (never used, 1 GB each) and only the checkpoints at multiples of --ge_keep_every + the newest
sub('''                                    safetensor_kohya_format_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-compatible_with_comfyui.safetensors")
                                    network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                                    save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                                    logger.info(f"Saved safetensor to {safetensor_save_path}")''', '''                                    if not args.ge_mask:
                                        safetensor_kohya_format_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-compatible_with_comfyui.safetensors")
                                        network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                                        save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                                    else:
                                        import re as _ger
                                        for _f in os.listdir(args.output_dir):
                                            _m = _ger.fullmatch(r"checkpoint-(\\d+)\\.safetensors", _f)
                                            if _m and int(_m.group(1)) < global_step and (args.ge_keep_every <= 0 or int(_m.group(1)) % args.ge_keep_every != 0):
                                                os.remove(os.path.join(args.output_dir, _f))
                                    logger.info(f"Saved safetensor to {safetensor_save_path}")''')
# optional (off by default): loss weight w on the latent cells that hold the panel's taxels and bars (dataset/panel_layout_latent.npy)
sub('''                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)''',
    '''                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                if args.ge_mask and args.ge_panel_loss_weight != 1.0:
                    if not hasattr(args, "_ge_w"):
                        import numpy as _gen
                        args._ge_w = (1 + (args.ge_panel_loss_weight - 1) * torch.from_numpy(_gen.load(args.ge_panel_layout)).float())[None, None, None]
                        accelerator.print(f"ge_panel_loss_weight {args.ge_panel_loss_weight} on {int(_gen.load(args.ge_panel_layout).sum())} latent cells")
                    assert args._ge_w.shape[-2:] == noise_pred.shape[-2:], (args._ge_w.shape, noise_pred.shape)
                    weighting = weighting * args._ge_w.to(noise_pred.device)''')
# also save at every epoch boundary (--ge_epoch_steps = steps per epoch), so exact pass points exist whatever the save period
sub('''                if global_step % args.checkpointing_steps == 0:''',
    '''                if global_step % args.checkpointing_steps == 0 or (args.ge_mask and args.ge_epoch_steps > 0 and global_step % args.ge_epoch_steps == 0):''')
dst.write_text(t); shutil.copy(Path(__file__).with_name("ge_mask.py"), dst.with_name("ge_mask.py"))
print("wrote", dst, "and", dst.with_name("ge_mask.py"))
