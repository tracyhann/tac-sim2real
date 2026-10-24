"""Write scripts/wan2.2_fun/train_control_lora_hf25.py: a COPY of VideoX-Fun's train_control_lora.py (already carrying the
opt-in --inpaint_first_frame_only and --resume_lora_weights patches) with the opt-in touch head of hf25_patch.py (copied
next to it). The shared train_control_lora.py is not touched: other runs relaunch it after every preemption.
  --tactile_head          rows carry "tactile" (npz with u (81, 880)); TactileHead on the DiT's last-layer video tokens
                          (+ frame-0 touch) -> u; loss += --tactile_lambda * R1(pred, u); the head is trained with the LoRA
                          (fp32, same optimizer / lr) and saved as tactile_head-<step>.pt next to checkpoint-<step>.safetensors;
                          --resume_lora_weights also reloads it
  --tactile_lambda 1.0    weight of the touch loss (the video loss is the mean flow-matching MSE; the touch loss is a mean too)
  python make_trainer_hf25.py <VideoX-Fun dir>"""
import shutil
import sys
from pathlib import Path

V = Path(sys.argv[1]); src = V / "scripts/wan2.2_fun/train_control_lora.py"; dst = V / "scripts/wan2.2_fun/train_control_lora_hf25.py"
t = src.read_text()
assert "resume_lora_weights" in t and "inpaint_first_frame_only" in t, "the base trainer lacks the earlier patches"


def sub(old, new, n=1):
    global t
    assert t.count(old) == n, (old[:80], t.count(old))
    t = t.replace(old, new)


# ---- flags
sub('        "--inpaint_first_frame_only", action="store_true",',
    '''        "--tactile_head", action="store_true", help="HF25: touch head on the DiT tokens (hf25_patch.py); rows need a tactile npz with u (frames, 880)."
    )
    parser.add_argument("--tactile_lambda", type=float, default=1.0)
    parser.add_argument(
        "--inpaint_first_frame_only", action="store_true",''')
# ---- dataset: samples carry the touch target
sub('''        enable_camera_info=args.train_mode == "control_camera_ref"
    )

    def worker_init_fn(_seed):''', '''        enable_camera_info=args.train_mode == "control_camera_ref"
    )
    if args.tactile_head:
        assert args.enable_bucket and args.add_inpaint_info and args.control_ref_image == "first_frame" and args.use_peft_lora, \\
            "--tactile_head needs --enable_bucket --add_inpaint_info --control_ref_image first_frame --use_peft_lora"
        from hf25_patch import HF25Dataset
        train_dataset.__class__ = HF25Dataset

    def worker_init_fn(_seed):''')
# ---- collate: stack it with the batch
sub('''            new_examples["text"]         = []
''', '''            new_examples["text"]         = []
            new_examples["tactile"]      = []
''')
sub('''                new_examples["text"].append(example["text"])
''', '''                new_examples["text"].append(example["text"])
                if "tactile" in example:
                    new_examples["tactile"].append(torch.from_numpy(example["tactile"]))
''')
sub('''            new_examples["control_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["control_pixel_values"]])
''', '''            new_examples["control_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["control_pixel_values"]])
            if new_examples["tactile"]:
                new_examples["tactile"] = torch.stack(new_examples["tactile"])
            else:
                del new_examples["tactile"]
''')
# ---- the head: created after the LoRA injection, before the trainable parameters are collected (so it joins the optimizer)
sub('''    if args.use_peft_lora:
        logging.info("Add peft parameters")
        trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))''',
    '''    _tcap = None; _tlog = {"tactile": float("nan")}
    if args.tactile_head:
        from hf25_patch import TactileHead, attach_capture
        transformer3d.tactile_head = TactileHead(transformer3d.head.dim)
        _tcap = attach_capture(transformer3d)
        logging.info(f"tactile head: {sum(p.numel() for p in transformer3d.tactile_head.parameters()) / 1e6:.1f} M params, lambda {args.tactile_lambda}")

    if args.use_peft_lora:
        logging.info("Add peft parameters")
        trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))''')
# ---- the trainer casts the whole model to bf16 before / after accelerator.prepare (the F / H LoRAs trained in bf16); the fresh head
# trains in fp32: AdamW steps of lr 1e-4 on bf16 weights round away (same Parameter objects, so the optimizer keeps them)
sub('''
    transformer3d.to(accelerator.device, dtype=weight_dtype)
''', '''
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    if args.tactile_head:
        accelerator.unwrap_model(transformer3d).tactile_head.float()
        accelerator.print("tactile head dtype", next(accelerator.unwrap_model(transformer3d).tactile_head.parameters()).dtype, "(LoRA params", next(p for n, p in transformer3d.named_parameters() if "lora_" in n).dtype, ")")
''')
# ---- save the head with every LoRA checkpoint (head first: a kill in between leaves a head without weights, never the reverse)
sub('''    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)''', '''    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        base_ = os.path.basename(ckpt_file)
        if args.tactile_head and base_.startswith("checkpoint-") and base_.endswith(".safetensors") and "compatible" not in base_:
            step_ = base_[len("checkpoint-"):-len(".safetensors")]
            hsd = {k: v.detach().float().cpu() for k, v in accelerator.unwrap_model(transformer3d).tactile_head.state_dict().items()}
            torch.save(hsd, os.path.join(args.output_dir, f"tactile_head-{step_}.pt"))
            accelerator.print(f"saved tactile_head-{step_}.pt ({len(hsd)} tensors)")''')
# ---- resume: reload the head of the resumed step
sub('''            args.resume_from_checkpoint = None                        # skip the state-dir resume below''',
    '''            if args.tactile_head:
                hp = os.path.join(args.output_dir, f"tactile_head-{lw_step}.pt")
                if not os.path.exists(hp):
                    raise SystemExit(f"resume_lora_weights: {hp} missing (delete {lw_file} to resume from the previous pair)")
                net.tactile_head.load_state_dict(torch.load(hp, map_location="cpu")); accelerator.print(f"resume_lora_weights: tactile head {hp} loaded")
            args.resume_from_checkpoint = None                        # skip the state-dir resume below''')
# ---- loss: video flow matching + lambda * touch (both means), the head read from the captured last-layer tokens
sub('''                loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                loss = loss.mean()
''', '''                loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                loss = loss.mean()
                if args.tactile_head:
                    from hf25_patch import tactile_loss, video_tokens
                    _hw = (noisy_latents.shape[3] // 2) * (noisy_latents.shape[4] // 2); _f = noisy_latents.shape[2]
                    _xt = video_tokens(_tcap[-1], _hw, _f, bool(args.add_full_ref_image_in_self_attention)); _tcap.clear()
                    _tu = batch["tactile"].to(_xt.device).float()
                    assert _tu.shape[1] == 4 * (_f - 1) + 1, (tuple(_tu.shape), _f)
                    _pred = accelerator.unwrap_model(transformer3d).tactile_head(_xt, _hw, _tu[:, 0])
                    _tl = tactile_loss(_pred, _tu); loss = loss + args.tactile_lambda * _tl; _tlog["tactile"] = _tl.detach().item()
''')
sub('''            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}''',
    '''            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "tactile": _tlog["tactile"]}''')

dst.write_text(t); shutil.copy2(Path(__file__).resolve().parent / "hf25_patch.py", dst.parent / "hf25_patch.py")
print("wrote", dst, "and", dst.parent / "hf25_patch.py")
