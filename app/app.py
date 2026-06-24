from __future__ import annotations

import gc
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import gradio as gr
import psutil
import torch

# Official Krea-2 repo mounted/cloned in the Docker image.
KREA_REPO = Path(os.getenv("KREA_REPO", "/workspace/krea-2"))
sys.path.insert(0, str(KREA_REPO))

import inference as krea_inference  # type: ignore  # noqa: E402
from sampling import sample  # type: ignore  # noqa: E402

from download_models import download_checkpoint  # noqa: E402
from safety import check_prompt  # noqa: E402

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/workspace/outputs"))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "/workspace/checkpoints"))
DEVICE = os.getenv("KREA2_DEVICE", "cuda")
DEFAULT_DTYPE = os.getenv("KREA2_DTYPE", "bfloat16")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

_PIPELINE_CACHE: dict[tuple[str, str, str], tuple[Any, Any, Any]] = {}


def _dtype_from_name(name: str):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def _resolve_checkpoint_path(checkpoint: str, raw_path: str | None, turbo_path: str | None) -> str:
    if checkpoint == "oss_turbo":
        path = turbo_path or os.getenv("OSS_TURBO") or str(CHECKPOINT_DIR / "turbo.safetensors")
    elif checkpoint == "oss_raw":
        path = raw_path or os.getenv("OSS_RAW") or str(CHECKPOINT_DIR / "raw.safetensors")
    else:
        raise ValueError(f"Unsupported checkpoint: {checkpoint}")

    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        label = "Turbo" if checkpoint == "oss_turbo" else "RAW"
        raise gr.Error(
            f"{label} checkpoint was not found at: {path}\n\n"
            "Use the Model Tools tab to download it, or mount the safetensors file into ./checkpoints."
        )
    return path


def _get_pipeline(checkpoint: str, dtype_name: str, raw_path: str | None, turbo_path: str | None):
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise gr.Error("CUDA is not available inside the container. Check NVIDIA Container Toolkit and docker compose GPU settings.")

    ckpt_path = _resolve_checkpoint_path(checkpoint, raw_path, turbo_path)
    krea_inference.checkpoints[checkpoint] = ckpt_path

    key = (checkpoint, dtype_name, ckpt_path)
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    dtype = _dtype_from_name(dtype_name)
    pipe = krea_inference._pipeline(checkpoint=checkpoint, device=DEVICE, dtype=dtype)
    _PIPELINE_CACHE[key] = pipe
    return pipe


def _zip_paths(paths: list[Path], run_dir: Path) -> Path:
    zip_path = run_dir / "images.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=p.name)
        meta = run_dir / "metadata.json"
        if meta.exists():
            zf.write(meta, arcname=meta.name)
    return zip_path


def _metadata(**kwargs) -> dict[str, Any]:
    data = dict(kwargs)
    data["created_unix"] = time.time()
    data["created_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["device"] = DEVICE
    if torch.cuda.is_available():
        data["gpu"] = torch.cuda.get_device_name(0)
    return data


def generate(
    prompt: str,
    checkpoint: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    y1: float,
    y2: float,
    mu: float | None,
    use_mu: bool,
    num_images: int,
    seed: int,
    dtype_name: str,
    raw_path: str,
    turbo_path: str,
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Prompt is required.")

    allowed, reason = check_prompt(prompt)
    if not allowed:
        raise gr.Error(reason)

    if width % 16 != 0 or height % 16 != 0:
        raise gr.Error("Width and height must be multiples of 16.")

    if checkpoint == "oss_turbo" and cfg != 0:
        gr.Warning("Turbo is designed for CFG 0.0. This will still run, but the recommended setting is 0.0.")

    yield [], None, "Loading model. First load can take several minutes because the checkpoint and Qwen text encoder are initialized."

    dit, ae, encoder = _get_pipeline(checkpoint, dtype_name, raw_path or None, turbo_path or None)

    yield [], None, "Generating images."
    with torch.inference_mode():
        images = sample(
            dit,
            ae,
            encoder,
            [prompt] * int(num_images),
            width=int(width),
            height=int(height),
            steps=int(steps),
            guidance=float(cfg),
            seed=int(seed),
            y1=float(y1),
            y2=float(y2),
            mu=float(mu) if use_mu and mu is not None else None,
        )

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{checkpoint}-{seed}"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for i, image in enumerate(images):
        out_path = run_dir / f"{run_id}_{i:02d}.png"
        image.save(out_path)
        saved.append(out_path)

    meta = _metadata(
        prompt=prompt,
        checkpoint=checkpoint,
        checkpoint_path=_resolve_checkpoint_path(checkpoint, raw_path or None, turbo_path or None),
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        y1=y1,
        y2=y2,
        mu=float(mu) if use_mu and mu is not None else None,
        num_images=num_images,
        seed=seed,
        dtype=dtype_name,
        files=[str(p) for p in saved],
    )
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    zip_path = _zip_paths(saved, run_dir)

    gallery = [(str(p), f"seed {seed + i}") for i, p in enumerate(saved)]
    yield gallery, str(zip_path), f"Saved {len(saved)} image(s) to {run_dir}"


def model_preset(checkpoint: str):
    if checkpoint == "oss_turbo":
        return 1024, 1024, 8, 0.0, 0.5, 1.15, 1.15, True
    return 1024, 1024, 52, 3.5, 0.5, 1.15, 0.0, False


def unload_models() -> str:
    _PIPELINE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return "Model cache cleared."


def system_info() -> str:
    lines = []
    lines.append(f"Python: {sys.version.split()[0]}")
    lines.append(f"Torch: {torch.__version__}")
    lines.append(f"Device target: {DEVICE}")
    lines.append(f"CPU RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
        lines.append(f"GPU VRAM total: {total / (1024**3):.1f} GB")
        lines.append(f"GPU VRAM free: {free / (1024**3):.1f} GB")
        lines.append(f"CUDA capability: {props.major}.{props.minor}")
    else:
        lines.append("CUDA: unavailable")
    lines.append(f"Krea repo: {KREA_REPO}")
    lines.append(f"Checkpoint dir: {CHECKPOINT_DIR}")
    lines.append(f"Output dir: {OUTPUT_DIR}")
    lines.append(f"Cached pipelines: {len(_PIPELINE_CACHE)}")
    return "\n".join(lines)


def history():
    items = []
    for png in sorted(OUTPUT_DIR.glob("**/*.png"), reverse=True)[:100]:
        items.append((str(png), png.parent.name))
    return items


def download_selected(which: str, token: str):
    key = "turbo" if "Turbo" in which else "raw"
    path = download_checkpoint(key, token.strip() or None)
    if key == "turbo":
        os.environ["OSS_TURBO"] = path
    else:
        os.environ["OSS_RAW"] = path
    return f"Downloaded {which} to {path}", path


CSS = """
#status_box textarea {font-family: ui-monospace, Consolas, monospace;}
"""

with gr.Blocks(title="Krea-2 Local UI", css=CSS) as demo:
    gr.Markdown(
        "# Krea-2 Local UI\n"
        "Local Docker GUI for Krea-2 RAW/Turbo text-to-image inference using the official open inference code."
    )

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", lines=6, placeholder="A cinematic concept art frame of...")
                checkpoint = gr.Radio(
                    choices=[("Turbo - fast inference", "oss_turbo"), ("RAW - base model / research", "oss_raw")],
                    value="oss_turbo",
                    label="Checkpoint",
                )
                preset_btn = gr.Button("Apply recommended preset")
                with gr.Row():
                    width = gr.Slider(512, 2048, value=1024, step=16, label="Width")
                    height = gr.Slider(512, 2048, value=1024, step=16, label="Height")
                with gr.Accordion("Advanced sampler controls", open=True):
                    steps = gr.Slider(1, 80, value=8, step=1, label="Steps")
                    cfg = gr.Slider(0, 10, value=0.0, step=0.1, label="CFG / guidance scale")
                    with gr.Row():
                        y1 = gr.Slider(0, 2, value=0.5, step=0.01, label="y1 / min-res timestep shift")
                        y2 = gr.Slider(0, 2, value=1.15, step=0.01, label="y2 / max-res timestep shift")
                    with gr.Row():
                        use_mu = gr.Checkbox(value=True, label="Use fixed mu")
                        mu = gr.Number(value=1.15, label="mu")
                    with gr.Row():
                        num_images = gr.Slider(1, 4, value=1, step=1, label="Images")
                        seed = gr.Number(value=0, precision=0, label="Seed")
                    dtype_name = gr.Dropdown(["bfloat16", "float16", "float32"], value=DEFAULT_DTYPE, label="Runtime dtype")
                with gr.Accordion("Manual checkpoint paths", open=False):
                    turbo_path = gr.Textbox(value=os.getenv("OSS_TURBO", str(CHECKPOINT_DIR / "turbo.safetensors")), label="Turbo safetensors path")
                    raw_path = gr.Textbox(value=os.getenv("OSS_RAW", str(CHECKPOINT_DIR / "raw.safetensors")), label="RAW safetensors path")
                generate_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Output", columns=2, height=720)
                zip_file = gr.File(label="Download run ZIP")
                status = gr.Textbox(label="Status", lines=4, elem_id="status_box")

        preset_btn.click(
            model_preset,
            inputs=[checkpoint],
            outputs=[width, height, steps, cfg, y1, y2, mu, use_mu],
        )
        generate_btn.click(
            generate,
            inputs=[prompt, checkpoint, width, height, steps, cfg, y1, y2, mu, use_mu, num_images, seed, dtype_name, raw_path, turbo_path],
            outputs=[gallery, zip_file, status],
        )

    with gr.Tab("Model Tools"):
        gr.Markdown(
            "Download Krea-2 checkpoints into `./checkpoints`. You may need to accept the model license on Hugging Face and provide `HF_TOKEN`."
        )
        which = gr.Radio(["Turbo", "RAW"], value="Turbo", label="Checkpoint to download")
        token = gr.Textbox(label="HF token", type="password", placeholder="Optional if already authenticated / public access works")
        dl_btn = gr.Button("Download selected checkpoint")
        dl_status = gr.Textbox(label="Download status", lines=3)
        dl_path = gr.Textbox(label="Downloaded path")
        dl_btn.click(download_selected, inputs=[which, token], outputs=[dl_status, dl_path])

        unload_btn = gr.Button("Unload model from VRAM")
        unload_status = gr.Textbox(label="Unload status")
        unload_btn.click(unload_models, outputs=[unload_status])

    with gr.Tab("History"):
        refresh_history = gr.Button("Refresh history")
        hist_gallery = gr.Gallery(label="Recent generated images", columns=4, height=720)
        refresh_history.click(history, outputs=[hist_gallery])
        demo.load(history, outputs=[hist_gallery])

    with gr.Tab("System"):
        sys_btn = gr.Button("Refresh system info")
        sys_out = gr.Textbox(label="System info", lines=14, elem_id="status_box")
        sys_btn.click(system_info, outputs=[sys_out])
        demo.load(system_info, outputs=[sys_out])

    with gr.Tab("Notes"):
        gr.Markdown(
            "## Scope\n"
            "This UI exposes the official open Krea-2 inference controls: prompt, RAW/Turbo checkpoint selection, steps, CFG, y1/y2 timestep shift, fixed mu, width, height, image count, seed, and output prefix/history.\n\n"
            "The open Krea-2 repository does not include a full local clone of every hosted Krea product feature such as video generation, enhancer, realtime canvas, motion transfer, lip sync, or 3D tools. It also does not ship a local LoRA trainer in the official inference repo. For LoRA workflows, train on RAW using a trainer such as Diffusers/Ostris/Kohya, then use Turbo for inference when compatible support is available."
        )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
    )
