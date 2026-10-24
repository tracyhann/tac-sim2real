"""Portable loader for the W video-diffusion pipeline (model "W": the identity of the backbone is withheld for the internal
test; VideoX-Fun provides the pipeline classes) + the video writer and negative prompt the HF25 scripts use. Everything is found under one root (env R2S2R):
  $R2S2R/third_party/VideoX-Fun                 the VideoX-Fun checkout (config/wan2.2/wan_civitai_i2v.yaml, videox_fun/)
  $R2S2R/models/W                                the weights as downloaded from Hugging Face latent-artist/861b8a78d34141a9ad659229049b6423
Default: every component goes to the GPU right after it is loaded (needs ~70 GB of GPU memory for the weights alone; host RAM can
be small). W_GPU_MEMORY_MODE=model_cpu_offload (or sequential_cpu_offload) instead builds the pipeline on the CPU and lets the
framework's own offload helper place it (apply_gpu_memory_mode, as in its examples): fits smaller GPUs, slower, needs host RAM for
the weights; NOT exercised by the authors. Loading takes 2-4 minutes from local disk; on a networked file system copy the weights
to local scratch first (W_STAGE_DIR=/path makes load_pipe do that)."""
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("R2S2R") or ("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r" if Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r").exists() else "."))
V = ROOT / "third_party/VideoX-Fun"; MODEL = ROOT / "models/W"; CONFIG = "config/wan2.2/wan_civitai_i2v.yaml"; WEIGHTS_REPO = "latent-artist/861b8a78d34141a9ad659229049b6423"
DEV, DT = "cuda", torch.bfloat16; GPU_MEMORY_MODE = os.environ.get("W_GPU_MEMORY_MODE", "")   # "" = all on the GPU
# the framework's standard negative prompt for this model family (as in its examples)
NEG_PROMPT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
                "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def stage(src):
    """optional copy of the weights to a local directory (W_STAGE_DIR); returns the directory to load from"""
    d = os.environ.get("W_STAGE_DIR")
    if not d:
        return src
    dst = Path(d) / src.name
    if not (dst / "low_noise_model").exists():
        t = time.time(); shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True); print(f"staged weights to {dst} in {time.time() - t:.0f}s", flush=True)
    return dst


def load_pipe(model="w"):
    """-> (pipe, vae, scheduler, boundary); pipe.transformer = low-noise expert, pipe.transformer_2 = high-noise expert (W has two experts)"""
    assert model == "w", model
    assert (V / CONFIG).exists(), f"VideoX-Fun not found at {V} (set R2S2R)"
    assert (MODEL / "low_noise_model").exists(), f"weights not found at {MODEL} (hf download {WEIGHTS_REPO} --local-dir {MODEL})"
    sys.path.insert(0, str(V)); os.chdir(V)
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from videox_fun.models import AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, Wan2_2Transformer3DModel, WanT5EncoderModel
    from videox_fun.pipeline import Wan2_2FunControlPipeline
    from videox_fun.utils import filter_kwargs
    path = str(stage(MODEL)); cfg = OmegaConf.load(V / CONFIG); tk = OmegaConf.to_container(cfg["transformer_additional_kwargs"]); t = time.time()
    dev = "cpu" if GPU_MEMORY_MODE else DEV                                            # offload mode: build on the CPU, place later (place_pipe)
    tr = Wan2_2Transformer3DModel.from_pretrained(os.path.join(path, tk.get("transformer_low_noise_model_subpath", "transformer")),
                                                  transformer_additional_kwargs=tk, low_cpu_mem_usage=True, torch_dtype=DT).to(dev)
    tr2 = None
    if tk.get("transformer_combination_type", "single") == "moe":
        tr2 = Wan2_2Transformer3DModel.from_pretrained(os.path.join(path, tk.get("transformer_high_noise_model_subpath", "transformer")),
                                                       transformer_additional_kwargs=tk, low_cpu_mem_usage=True, torch_dtype=DT).to(dev)
    VAE = {"AutoencoderKLWan": AutoencoderKLWan, "AutoencoderKLWan3_8": AutoencoderKLWan3_8}[cfg["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = VAE.from_pretrained(os.path.join(path, cfg["vae_kwargs"].get("vae_subpath", "vae")), additional_kwargs=OmegaConf.to_container(cfg["vae_kwargs"])).to(DT).to(dev)
    tok = AutoTokenizer.from_pretrained(os.path.join(path, cfg["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")))
    te = WanT5EncoderModel.from_pretrained(os.path.join(path, cfg["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
                                           additional_kwargs=OmegaConf.to_container(cfg["text_encoder_kwargs"]), low_cpu_mem_usage=True, torch_dtype=DT).eval().to(dev)
    sch = FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(cfg["scheduler_kwargs"])))
    pipe = Wan2_2FunControlPipeline(transformer=tr, transformer_2=tr2, vae=vae, tokenizer=tok, text_encoder=te, scheduler=sch).to(dev)
    boundary = tk.get("boundary", 0.875); print(f"pipeline loaded in {time.time() - t:.0f}s ({'offload mode ' + GPU_MEMORY_MODE if GPU_MEMORY_MODE else 'all on ' + DEV})", flush=True)
    return pipe, vae, sch, boundary


def place_pipe(pipe):
    """offload mode only (call AFTER merging LoRAs): the framework's helper installs the offload hooks and moves what stays on the GPU"""
    if GPU_MEMORY_MODE:
        from videox_fun.utils import apply_gpu_memory_mode
        apply_gpu_memory_mode(pipe, GPU_MEMORY_MODE, DEV, DT)
    return pipe


def lora_device():
    """where merge_lora should put the LoRA tensors: next to the weights (CPU in offload mode)"""
    return "cpu" if GPU_MEMORY_MODE else DEV


def write_video(path, frames, fps, crf=14):
    """uint8 (T, H, W, 3) frames -> H.264 mp4"""
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=None, macro_block_size=1, ffmpeg_params=["-crf", str(crf), "-pix_fmt", "yuv420p", "-preset", "medium"])
    for f in frames:
        w.append_data(np.asarray(f))
    w.close()
