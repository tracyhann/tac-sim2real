"""Opt-in --resume_lora_weights for VideoX-Fun scripts/wan2.2_fun/train_control_lora.py (2026-10-22): runs saved without
--save_state keep only checkpoint-N.safetensors (the PEFT LoRA weights), and `--resume_from_checkpoint latest` then dies
parsing 'checkpoint-150.safetensors' (and its state path loads into `network`, which is None with --use_peft_lora).
With the flag and no checkpoint-N/ state dirs: load the newest checkpoint-N.safetensors into the injected LoRA
(set_peft_model_state_dict), check every LoRA tensor matched, continue at global_step N with a fresh optimizer (constant
LR, so only Adam's moments restart). Default behaviour unchanged."""
import sys
from pathlib import Path

p = Path(sys.argv[1]); t = p.read_text()
if "resume_lora_weights" in t:
    print("already patched"); sys.exit(0)
a = '        "--inpaint_first_frame_only", action="store_true",'
assert t.count(a) == 1
t = t.replace(a, '        "--resume_lora_weights", action="store_true", help="With --resume_from_checkpoint latest and no checkpoint-N/ state dirs: '
                 'continue from the newest checkpoint-N.safetensors LoRA weights (fresh optimizer)."\n    )\n    parser.add_argument(\n' + a, 1)
b = '''    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:'''
assert t.count(b) == 1
t = t.replace(b, '''    global_step = 0
    first_epoch = 0

    # --resume_lora_weights: no state dirs, only checkpoint-N.safetensors -> load the LoRA weights, continue at step N
    lw_step = None
    if args.resume_from_checkpoint == "latest" and args.resume_lora_weights and args.use_peft_lora and os.path.isdir(args.output_dir):
        ents = os.listdir(args.output_dir)
        state_dirs = [d for d in ents if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))]
        lw = [d for d in ents if d.startswith("checkpoint-") and d.endswith(".safetensors") and d[len("checkpoint-"):-len(".safetensors")].isdigit()]
        if not state_dirs and lw:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            lw_file = max(lw, key=lambda d: int(d[len("checkpoint-"):-len(".safetensors")])); lw_step = int(lw_file[len("checkpoint-"):-len(".safetensors")])
            sd = load_file(os.path.join(args.output_dir, lw_file), device="cpu"); net = accelerator.unwrap_model(transformer3d)
            res = set_peft_model_state_dict(net, sd)
            n_lora = sum(1 for k_, _ in net.named_parameters() if "lora_" in k_)
            unexpected = list(getattr(res, "unexpected_keys", []) or [])
            k0 = sorted(sd)[0]; p0 = dict(net.named_parameters())[k0.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")]
            diff = float((p0.detach().float().cpu() - sd[k0].float()).abs().max())
            accelerator.print(f"resume_lora_weights: {lw_file} -> step {lw_step}; {len(sd)} tensors in the file, {n_lora} LoRA params in the model, "
                              f"{len(unexpected)} unexpected keys, |{k0}| max diff after load {diff:.2e}")
            if unexpected or len(sd) != n_lora or diff > 1e-2:
                raise SystemExit("resume_lora_weights: the LoRA weights did not load cleanly")
            args.resume_from_checkpoint = None                        # skip the state-dir resume below

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:''')
c = '''    else:
        initial_global_step = 0

    # function for saving/removing'''
assert t.count(c) == 1
t = t.replace(c, '''    else:
        initial_global_step = 0
    if lw_step is not None:
        global_step = initial_global_step = lw_step; first_epoch = global_step // num_update_steps_per_epoch
        accelerator.print(f"resume_lora_weights: continuing at global_step {global_step}, epoch {first_epoch}")

    # function for saving/removing''')
p.write_text(t); print("patched")
