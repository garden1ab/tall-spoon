from __future__ import annotations

import gc
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

# Krea-2 uses torch.compile in parts of the model. That usually improves repeated
# generations, but the first generation may spend a long time compiling kernels.
# Set KREA2_DISABLE_TORCH_COMPILE=1 in .env when you want faster first-run latency
# at the expense of slower sustained throughput. This must happen before importing
# the official Krea-2 modules.
if os.getenv("KREA2_DISABLE_TORCH_COMPILE", "0") == "1":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import gradio as gr
import psutil
import torch

DISABLE_TORCH_COMPILE = os.getenv("KREA2_DISABLE_TORCH_COMPILE", "0") == "1"
if DISABLE_TORCH_COMPILE:
    def _no_torch_compile(fn=None, *args, **kwargs):
        if callable(fn):
            return fn
        def _decorator(inner):
            return inner
        return _decorator
    torch.compile = _no_torch_compile  # type: ignore[assignment]

# Official Krea-2 repo mounted/cloned in the Docker image.
KREA_REPO = Path(os.getenv("KREA_REPO", "/workspace/krea-2"))
sys.path.insert(0, str(KREA_REPO))

import inference as krea_inference  # type: ignore  # noqa: E402
from einops import rearrange  # type: ignore  # noqa: E402
from sampling import prepare, roundup, sample, timesteps  # type: ignore  # noqa: E402

from download_models import download_checkpoint  # noqa: E402
from safety import check_prompt  # noqa: E402

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/workspace/outputs"))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "/workspace/checkpoints"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/workspace/models"))
TEXT_ENCODER_REPO = os.getenv("KREA2_TEXT_ENCODER_REPO", "Qwen/Qwen3-VL-4B-Instruct")
TEXT_ENCODER_PATH = Path(os.getenv("KREA2_TEXT_ENCODER_PATH", str(MODEL_DIR / "Qwen-Qwen3-VL-4B-Instruct")))
VAE_REPO = os.getenv("KREA2_VAE_REPO", "Qwen/Qwen-Image")
VAE_PATH = Path(os.getenv("KREA2_VAE_PATH", str(MODEL_DIR / "Qwen-Qwen-Image")))
OFFLINE_MODE = os.getenv("KREA2_OFFLINE_MODE", "1") == "1"
DEVICE = os.getenv("KREA2_DEVICE", "cuda")
DEFAULT_DTYPE = os.getenv("KREA2_DTYPE", "bfloat16")

# Hard-offline runtime mode. This prevents Transformers/Hugging Face from trying
# network fallback when Generate is clicked. Missing model files should fail fast
# with a local-file error instead of opening a URL.
if OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_PIPELINE_CACHE: dict[tuple[str, str, str], tuple[Any, Any, Any]] = {}
MAX_CACHED_PIPELINES = int(os.getenv("KREA2_MAX_CACHED_PIPELINES", "1"))


def _clear_pipeline_cache() -> None:
    """Release cached Krea pipelines and return as much VRAM as possible."""
    _PIPELINE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()



DEFAULT_CONDITIONING_WEIGHTS = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"

CONDITIONING_PRESETS: dict[str, tuple[bool, float, list[float]]] = {
    "Off / untouched": (False, 1.0, [1.0] * 12),
    "Neutral global multiplier only": (True, 1.0, [1.0] * 12),
    "Krea2 ComfyUI default": (True, 4.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0]),
    "Late layer detail boost": (True, 2.0, [1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.4, 2.0, 3.0, 1.2, 2.5, 1.0]),
    "Balanced structure boost": (True, 1.5, [1.2, 1.2, 1.15, 1.15, 1.1, 1.1, 1.0, 1.25, 1.4, 1.0, 1.25, 1.0]),
    "Aggressive prompt adherence": (True, 3.0, [1.0, 1.0, 1.0, 1.1, 1.25, 1.5, 1.75, 2.5, 4.0, 1.3, 3.0, 1.0]),
    "Extreme verification test": (True, 1.0, [0.25, 0.25, 0.25, 0.5, 0.75, 1.0, 1.25, 3.0, 8.0, 0.5, 6.0, 0.25]),
}


def _dtype_from_name(name: str):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def _looks_like_hf_snapshot(path: Path) -> bool:
    """Return True when a local directory appears usable by Transformers.from_pretrained."""
    if not path.exists() or not path.is_dir():
        return False
    required_any = ["config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json"]
    tokenizer_any = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    has_model_config = any((path / item).exists() for item in required_any)
    has_tokenizer = any((path / item).exists() for item in tokenizer_any)
    return has_model_config and has_tokenizer


def _looks_like_vae_snapshot(path: Path) -> bool:
    """Return True when the local Qwen-Image VAE subfolder is present."""
    return path.exists() and path.is_dir() and (path / "vae" / "config.json").exists()


def _resolve_vae_path() -> str:
    path = Path(os.path.expanduser(str(VAE_PATH))).resolve()
    if _looks_like_vae_snapshot(path):
        return str(path)
    if OFFLINE_MODE:
        raise gr.Error(
            "Qwen-Image VAE is not available locally. Runtime is hard-offline, so Generate will not download it.\n\n"
            f"Expected local VAE snapshot at: {path}\n\n"
            "Run the one-time online prep command while internet is available:\n"
            "docker compose --profile prepare run --rm model-prep\n\n"
            "Then restart with docker compose up."
        )
    return VAE_REPO


class LocalQwenAutoencoder(torch.nn.Module):
    """Krea-2 autoencoder loader patched to use a local Qwen-Image VAE snapshot."""

    def __init__(self):
        super().__init__()
        from diffusers import AutoencoderKLQwenImage

        vae_source = _resolve_vae_path()
        kwargs = {"subfolder": "vae"}
        if OFFLINE_MODE:
            kwargs["local_files_only"] = True
        self.ae = AutoencoderKLQwenImage.from_pretrained(vae_source, **kwargs)
        self.compression = 8
        self.channels = 16
        self.register_buffer("latents_mean", torch.tensor(self.ae.config.latents_mean).view(1, -1, 1, 1, 1))
        self.register_buffer("latents_std", torch.tensor(self.ae.config.latents_std).view(1, -1, 1, 1, 1))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c h w -> b c 1 h w")
        x = (x * self.latents_std) + self.latents_mean
        return rearrange(self.ae.decode(x).sample, "b c 1 h w -> b c h w")


def _resolve_text_encoder_path() -> str:
    """Resolve the Qwen text encoder to a local directory in offline mode."""
    path = Path(os.path.expanduser(str(TEXT_ENCODER_PATH))).resolve()
    if _looks_like_hf_snapshot(path):
        return str(path)

    if OFFLINE_MODE:
        raise gr.Error(
            "Krea-2 text encoder is not available locally. Runtime is hard-offline, so Generate will not download it.\n\n"
            f"Expected local text encoder snapshot at: {path}\n\n"
            "Run the one-time online prep command while internet is available:\n"
            "docker compose --profile prepare run --rm model-prep\n\n"
            "Then restart with docker compose up."
        )

    # Dev/online fallback only. Normal packaged use should keep OFFLINE_MODE=1.
    return TEXT_ENCODER_REPO


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

    # Krea-2 is large. Keep only one resident pipeline unless explicitly changed.
    # This prevents accidentally loading RAW + Turbo, or two dtype variants, on 16 GB cards.
    if MAX_CACHED_PIPELINES <= 1 or len(_PIPELINE_CACHE) >= MAX_CACHED_PIPELINES:
        _clear_pipeline_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    dtype = _dtype_from_name(dtype_name)
    # Patch the official pipeline's QwenAutoencoder symbol before it builds the VAE.
    # The upstream file loads AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae").
    # This replacement points it at the local snapshot in offline runtime.
    krea_inference.QwenAutoencoder = LocalQwenAutoencoder
    text_encoder_model_id = _resolve_text_encoder_path()
    base_text_cfg = getattr(krea_inference, "qwen3_vl_4b", None)
    text_encoder_config = krea_inference.TextEncoderConfig(
        model_id=text_encoder_model_id,
        max_length=getattr(base_text_cfg, "max_length", 512),
        select_layers=getattr(
            base_text_cfg,
            "select_layers",
            (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
        ),
    )
    pipe = krea_inference._pipeline(
        checkpoint=checkpoint,
        device=DEVICE,
        dtype=dtype,
        text_encoder_config=text_encoder_config,
    )
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


def _seconds(value: float) -> str:
    return f"{value:.1f}s" if value < 120 else f"{value / 60:.1f}m"



def _format_weights(weights: list[float]) -> str:
    return ",".join(f"{float(w):.4g}" for w in weights)


def _parse_per_layer(s: str | None) -> list[float] | None:
    """Parse a comma/semicolon-separated list of conditioning layer gains."""
    if not s:
        return None
    try:
        vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        return None
    if len(vals) < 2:
        return None
    return vals


def _tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    """Small, cheap stats used to verify conditioning changes in the UI."""
    with torch.no_grad():
        work = t.detach().float()
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "mean_abs": float(work.abs().mean().item()),
            "std": float(work.std().item()),
            "max_abs": float(work.abs().max().item()),
        }


def _scale_cond_tensor(
    t: torch.Tensor,
    multiplier: float,
    per_layer_weights: list[float] | None = None,
) -> tuple[torch.Tensor, str]:
    """Scale Krea-2/Qwen conditioning and report exactly what was applied.

    The official Krea-2 encoder usually returns a 4D tensor shaped
    (B, seq, 12, D). The previous UI only handled flattened ComfyUI-style
    tensors shaped (B, seq, 12*D), so the 12 per-layer weights silently fell
    back to a global multiplier. A global multiplier alone can be muted by
    RMSNorm inside Krea-2's text-fusion path, which is why the control could
    appear to do almost nothing.
    """
    multiplier = float(multiplier)
    if per_layer_weights is None:
        return t * multiplier, "global-only"

    n_layers = len(per_layer_weights)
    orig_dtype = t.dtype

    # Official Krea-2 path: (B, seq, 12, D). Apply gains along the selected
    # Qwen hidden-state/tap axis.
    if n_layers > 1 and t.dim() >= 4 and t.shape[-2] == n_layers:
        work = t.float()
        gains = torch.tensor(per_layer_weights, dtype=work.dtype, device=work.device)
        view_shape = [1] * work.dim()
        view_shape[-2] = n_layers
        work = work * gains.view(*view_shape)
        return work.to(orig_dtype) * multiplier, "4d-layer-axis"

    # ComfyUI-style fallback: (B, seq, 12*D).
    flat = t.shape[-1]
    if n_layers > 1 and flat % n_layers == 0:
        layer_dim = flat // n_layers
        work = t.float().view(*t.shape[:-1], n_layers, layer_dim)
        gains = torch.tensor(per_layer_weights, dtype=work.dtype, device=work.device)
        work = work * gains.view(*([1] * (work.dim() - 2)), n_layers, 1)
        work = work.view(*work.shape[:-2], flat)
        return work.to(orig_dtype) * multiplier, "flattened-layer-axis"

    return t * multiplier, "global-only-shape-mismatch"


def conditioning_preset(name: str):
    enabled, multiplier, weights = CONDITIONING_PRESETS.get(name, CONDITIONING_PRESETS["Krea2 ComfyUI default"])
    return [enabled, multiplier, _format_weights(weights), *weights]


@torch.no_grad()
def sample_with_conditioning_rebalance(
    model,
    ae,
    encoder,
    prompts,
    *,
    negative_prompts=None,
    device="cuda",
    dtype=torch.bfloat16,
    width=1024,
    height=1024,
    steps=28,
    guidance=4.5,
    seed=0,
    minres=256,
    maxres=1280,
    y1=0.5,
    y2=1.15,
    mu=None,
    conditioning_multiplier: float = 1.0,
    per_layer_weights: list[float] | None = None,
):
    """Krea-2 sampler variant that rebalances only positive text conditioning.

    This mirrors the official `sampling.sample` implementation and changes one
    thing: after `encoder(prompts)`, the positive `txt` conditioning tensor is
    globally/per-layer scaled before being passed to the MMDiT. Negative CFG
    conditioning is intentionally left untouched.
    """
    patch = model.config.patch
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")
    n = len(prompts)
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n

    noise = torch.cat(
        [
            torch.randn(
                1,
                ae.channels,
                height // ae.compression,
                width // ae.compression,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    )

    txt, txtmask = encoder(prompts)
    cond_before = _tensor_stats(txt)
    txt, conditioning_applied_mode = _scale_cond_tensor(txt, conditioning_multiplier, per_layer_weights)
    cond_after = _tensor_stats(txt)
    conditioning_debug = {
        "applied_mode": conditioning_applied_mode,
        "multiplier": float(conditioning_multiplier),
        "weights": per_layer_weights,
        "before": cond_before,
        "after": cond_after,
        "mean_abs_ratio": (cond_after["mean_abs"] / cond_before["mean_abs"]) if cond_before["mean_abs"] else None,
        "std_ratio": (cond_after["std"] / cond_before["std"]) if cond_before["std"] else None,
    }
    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)

    if cfg:
        untxt, untxtmask = encoder(negative_prompts)
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    x1 = (minres // (ae.compression * patch)) ** 2
    x2 = (maxres // (ae.compression * patch)) ** 2
    ts = timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    img = x
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
        cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
        if cfg:
            uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
            v = cond + guidance * (cond - uncond)
        else:
            v = cond
        img = img + (tprev - tcurr) * v

    img = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=height // (ae.compression * patch),
        w=width // (ae.compression * patch),
    )
    img = ae.decode(img.to(torch.bfloat16))
    img = img.clamp(-1, 1) * 0.5 + 0.5
    img = rearrange(img * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    return [Image.fromarray(img[i]) for i in range(len(img))], conditioning_debug

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
    cond_rebalance_enabled: bool,
    cond_multiplier: float,
    cond_per_layer_weights: str,
    cond_use_layer_sliders: bool,
    cond_l01: float,
    cond_l02: float,
    cond_l03: float,
    cond_l04: float,
    cond_l05: float,
    cond_l06: float,
    cond_l07: float,
    cond_l08: float,
    cond_l09: float,
    cond_l10: float,
    cond_l11: float,
    cond_l12: float,
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

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    yield [], None, (
        "Loading model. The first run can be slow because Krea-2 initializes the checkpoint, "
        "Qwen text encoder, and optionally Torch/Triton compiled kernels."
    )

    try:
        dit, ae, encoder = _get_pipeline(checkpoint, dtype_name, raw_path or None, turbo_path or None)
    except torch.cuda.OutOfMemoryError as exc:
        _clear_pipeline_cache()
        raise gr.Error(
            "CUDA out of memory while loading Krea-2. Use Turbo, bfloat16, 1 image, and the 16 GB safe preset. "
            "Then restart the container if VRAM is still reserved."
        ) from exc

    load_elapsed = time.perf_counter() - load_start
    gen_start = time.perf_counter()
    yield [], None, (
        f"Model ready in {_seconds(load_elapsed)}. Generating images. "
        "If this is the first generation for this resolution, Torch/Triton may still compile kernels."
    )

    parsed_layer_weights = None
    if cond_rebalance_enabled:
        if cond_use_layer_sliders:
            parsed_layer_weights = [
                float(cond_l01), float(cond_l02), float(cond_l03), float(cond_l04),
                float(cond_l05), float(cond_l06), float(cond_l07), float(cond_l08),
                float(cond_l09), float(cond_l10), float(cond_l11), float(cond_l12),
            ]
        else:
            parsed_layer_weights = _parse_per_layer(cond_per_layer_weights)
            if parsed_layer_weights is None:
                raise gr.Error("Per-layer weights must be a comma-separated list of at least two numbers, or enable the 12 layer sliders.")
        if len(parsed_layer_weights) != 12:
            gr.Warning(f"Expected 12 Krea/Qwen layer weights; got {len(parsed_layer_weights)}. If the tensor shape is not divisible by that count, only the global multiplier will apply.")

    try:
        with torch.inference_mode():
            if cond_rebalance_enabled:
                images, conditioning_debug = sample_with_conditioning_rebalance(
                    dit,
                    ae,
                    encoder,
                    [prompt] * int(num_images),
                    device=DEVICE,
                    dtype=_dtype_from_name(dtype_name),
                    width=int(width),
                    height=int(height),
                    steps=int(steps),
                    guidance=float(cfg),
                    seed=int(seed),
                    y1=float(y1),
                    y2=float(y2),
                    mu=float(mu) if use_mu and mu is not None else None,
                    conditioning_multiplier=float(cond_multiplier),
                    per_layer_weights=parsed_layer_weights,
                )
            else:
                conditioning_debug = {"applied_mode": "disabled"}
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
    except torch.cuda.OutOfMemoryError as exc:
        _clear_pipeline_cache()
        raise gr.Error(
            "CUDA out of memory during generation. Use the 16 GB safe preset, set Images to 1, keep CFG at 0 for Turbo, "
            "and avoid RAW on a 16 GB card unless you add offload/quantization support."
        ) from exc

    gen_elapsed = time.perf_counter() - gen_start
    save_start = time.perf_counter()

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
        conditioning_rebalance_enabled=bool(cond_rebalance_enabled),
        conditioning_multiplier=float(cond_multiplier) if cond_rebalance_enabled else 1.0,
        conditioning_per_layer_weights=parsed_layer_weights if cond_rebalance_enabled else None,
        conditioning_positive_only=True if cond_rebalance_enabled else None,
        conditioning_debug=conditioning_debug,
        files=[str(p) for p in saved],
    )
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    zip_path = _zip_paths(saved, run_dir)

    save_elapsed = time.perf_counter() - save_start
    total_elapsed = time.perf_counter() - total_start
    per_image = gen_elapsed / max(1, len(saved))

    gallery = [(str(p), f"seed {seed + i}") for i, p in enumerate(saved)]
    conditioning_status = "Conditioning rebalance: disabled"
    if cond_rebalance_enabled:
        before = conditioning_debug.get("before", {})
        after = conditioning_debug.get("after", {})
        conditioning_status = (
            f"Conditioning rebalance: {conditioning_debug.get('applied_mode')} | "
            f"shape {before.get('shape')} | "
            f"mean_abs {before.get('mean_abs', 0):.4f} -> {after.get('mean_abs', 0):.4f} | "
            f"std {before.get('std', 0):.4f} -> {after.get('std', 0):.4f}"
        )

    yield gallery, str(zip_path), (
        f"Saved {len(saved)} image(s) to {run_dir}\n"
        f"Timing: load/cache {_seconds(load_elapsed)} | generate {_seconds(gen_elapsed)} "
        f"({_seconds(per_image)} per image) | save {_seconds(save_elapsed)} | total {_seconds(total_elapsed)}\n"
        f"{conditioning_status}\n"
        f"Torch compile disabled: {DISABLE_TORCH_COMPILE}"
    )


def model_preset(checkpoint: str):
    if checkpoint == "oss_turbo":
        return 1024, 1024, 8, 0.0, 0.5, 1.15, 1.15, True
    return 1024, 1024, 52, 3.5, 0.5, 1.15, 0.0, False


def safe_16gb_preset():
    """Conservative settings for 16 GB GPUs / WSL2."""
    return "oss_turbo", 768, 768, 8, 0.0, 0.5, 1.15, 1.15, True, 1, "bfloat16"


def speed_preset(mode: str):
    """Latency-oriented Turbo presets. All keep CFG at 0 to avoid double forward passes."""
    presets = {
        "Ultra-fast test - 512 / 4 steps": ("oss_turbo", 512, 512, 4, 0.0, 0.5, 1.15, 1.15, True, 1, "bfloat16"),
        "Fast draft - 640 / 6 steps": ("oss_turbo", 640, 640, 6, 0.0, 0.5, 1.15, 1.15, True, 1, "bfloat16"),
        "16 GB balanced - 768 / 8 steps": ("oss_turbo", 768, 768, 8, 0.0, 0.5, 1.15, 1.15, True, 1, "bfloat16"),
        "Quality Turbo - 1024 / 8 steps": ("oss_turbo", 1024, 1024, 8, 0.0, 0.5, 1.15, 1.15, True, 1, "bfloat16"),
        "Try float16 draft - 768 / 8 steps": ("oss_turbo", 768, 768, 8, 0.0, 0.5, 1.15, 1.15, True, 1, "float16"),
    }
    return presets.get(mode, presets["16 GB balanced - 768 / 8 steps"])


def warmup_current_settings(
    checkpoint: str,
    width: int,
    height: int,
    cfg: float,
    y1: float,
    y2: float,
    mu: float | None,
    use_mu: bool,
    dtype_name: str,
    raw_path: str,
    turbo_path: str,
) -> str:
    """Compile/load warmup for the selected model/resolution without saving an image."""
    t0 = time.perf_counter()
    try:
        dit, ae, encoder = _get_pipeline(checkpoint, dtype_name, raw_path or None, turbo_path or None)
        with torch.inference_mode():
            images = sample(
                dit,
                ae,
                encoder,
                ["warmup image, simple neutral subject"],
                width=int(width),
                height=int(height),
                steps=1,
                guidance=float(cfg),
                seed=123456,
                y1=float(y1),
                y2=float(y2),
                mu=float(mu) if use_mu and mu is not None else None,
            )
        del images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (
            f"Warmup completed in {_seconds(time.perf_counter() - t0)} for {checkpoint} at {width}x{height}. "
            "Run generation again with the same resolution/settings to reuse the loaded model and compiled kernels."
        )
    except torch.cuda.OutOfMemoryError as exc:
        _clear_pipeline_cache()
        raise gr.Error("CUDA out of memory during warmup. Try the ultra-fast 512 preset first.") from exc


def unload_models() -> str:
    _clear_pipeline_cache()
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
    lines.append(f"Model dir: {MODEL_DIR}")
    lines.append(f"Text encoder path: {TEXT_ENCODER_PATH}")
    lines.append(f"VAE path: {VAE_PATH}")
    lines.append(f"Hard-offline runtime: {OFFLINE_MODE}")
    lines.append(f"HF_HUB_OFFLINE: {os.getenv('HF_HUB_OFFLINE', '')}")
    lines.append(f"TRANSFORMERS_OFFLINE: {os.getenv('TRANSFORMERS_OFFLINE', '')}")
    lines.append(f"Output dir: {OUTPUT_DIR}")
    lines.append(f"Cached pipelines: {len(_PIPELINE_CACHE)}")
    lines.append(f"Torch compile disabled: {DISABLE_TORCH_COMPILE}")
    lines.append(f"TorchInductor cache: {os.getenv('TORCHINDUCTOR_CACHE_DIR', 'default')}")
    lines.append(f"Triton cache: {os.getenv('TRITON_CACHE_DIR', 'default')}")
    return "\n".join(lines)


def history():
    items = []
    for png in sorted(OUTPUT_DIR.glob("**/*.png"), reverse=True)[:100]:
        items.append((str(png), png.parent.name))
    return items


def local_model_status() -> str:
    turbo = Path(os.getenv("OSS_TURBO", str(CHECKPOINT_DIR / "turbo.safetensors")))
    raw = Path(os.getenv("OSS_RAW", str(CHECKPOINT_DIR / "raw.safetensors")))
    text_path = Path(os.path.expanduser(str(TEXT_ENCODER_PATH))).resolve()
    vae_path = Path(os.path.expanduser(str(VAE_PATH))).resolve()

    def fmt_file(path: Path) -> str:
        if path.exists():
            return f"FOUND - {path} ({path.stat().st_size / (1024**3):.2f} GB)"
        return f"MISSING - {path}"

    lines = [
        f"Hard-offline runtime: {OFFLINE_MODE}",
        f"HF_HUB_OFFLINE: {os.getenv('HF_HUB_OFFLINE', '')}",
        f"TRANSFORMERS_OFFLINE: {os.getenv('TRANSFORMERS_OFFLINE', '')}",
        f"Text encoder repo: {TEXT_ENCODER_REPO}",
        f"Text encoder local path: {text_path}",
        f"Text encoder snapshot: {'FOUND' if _looks_like_hf_snapshot(text_path) else 'MISSING/INCOMPLETE'}",
        f"VAE repo: {VAE_REPO}",
        f"VAE local path: {vae_path}",
        f"VAE snapshot: {'FOUND' if _looks_like_vae_snapshot(vae_path) else 'MISSING/INCOMPLETE'}",
        f"Turbo checkpoint: {fmt_file(turbo)}",
        f"RAW checkpoint: {fmt_file(raw)}",
        "",
        "If anything is missing, run while online:",
        "docker compose --profile prepare run --rm model-prep",
        "",
        "Then use normal offline runtime:",
        "docker compose up",
    ]
    return "\n".join(lines)


def download_selected(which: str, token: str):
    if OFFLINE_MODE:
        raise gr.Error(
            "Runtime is hard-offline, so the UI will not download models during Generate or from this tab.\n\n"
            "Use the one-time online prep service instead:\n"
            "docker compose --profile prepare run --rm model-prep"
        )
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
                safe_16gb_btn = gr.Button("Apply 16 GB safe preset")
                with gr.Row():
                    speed_mode = gr.Dropdown(
                        choices=[
                            "Ultra-fast test - 512 / 4 steps",
                            "Fast draft - 640 / 6 steps",
                            "16 GB balanced - 768 / 8 steps",
                            "Quality Turbo - 1024 / 8 steps",
                            "Try float16 draft - 768 / 8 steps",
                        ],
                        value="16 GB balanced - 768 / 8 steps",
                        label="Speed preset",
                    )
                    speed_btn = gr.Button("Apply speed preset")
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
                with gr.Accordion("Conditioning rebalance / experimental weights", open=False):
                    gr.Markdown(
                        "Scales Krea-2's positive Qwen conditioning before denoising. "
                        "This build detects official Krea-2's 4D conditioning shape `(B, seq, 12, D)` and applies the 12 weights on the correct layer/tap axis. "
                        "The status box reports tensor stats so you can verify the conditioning actually changed. Negative CFG conditioning is left untouched."
                    )
                    cond_rebalance_enabled = gr.Checkbox(value=False, label="Enable conditioning rebalance")
                    with gr.Row():
                        cond_preset = gr.Dropdown(
                            choices=list(CONDITIONING_PRESETS.keys()),
                            value="Krea2 ComfyUI default",
                            label="Weight preset",
                        )
                        cond_preset_btn = gr.Button("Apply weight preset")
                    cond_multiplier = gr.Slider(0, 8, value=4.0, step=0.01, label="Global conditioning multiplier")
                    cond_per_layer_weights = gr.Textbox(
                        value=DEFAULT_CONDITIONING_WEIGHTS,
                        label="Manual per-layer weights",
                        placeholder="12 comma-separated values, for example: 1,1,1,1,1,1,1,2.5,5,1.1,4,1",
                    )
                    cond_use_layer_sliders = gr.Checkbox(value=True, label="Use the 12 layer sliders below instead of the manual text field")
                    with gr.Row():
                        cond_l01 = gr.Slider(0, 8, value=1.0, step=0.05, label="Layer 01")
                        cond_l02 = gr.Slider(0, 8, value=1.0, step=0.05, label="Layer 02")
                        cond_l03 = gr.Slider(0, 8, value=1.0, step=0.05, label="Layer 03")
                        cond_l04 = gr.Slider(0, 8, value=1.0, step=0.05, label="Layer 04")
                    with gr.Row():
                        cond_l05 = gr.Slider(0, 8, value=1.0, step=0.05, label="Layer 05")
                        cond_l06 = gr.Slider(0, 8, value=1.0, step=0.05, label="Layer 06")
                        cond_l07 = gr.Slider(0, 8, value=1.0, step=0.05, label="Layer 07")
                        cond_l08 = gr.Slider(0, 8, value=2.5, step=0.05, label="Layer 08")
                    with gr.Row():
                        cond_l09 = gr.Slider(0, 8, value=5.0, step=0.05, label="Layer 09")
                        cond_l10 = gr.Slider(0, 8, value=1.1, step=0.05, label="Layer 10")
                        cond_l11 = gr.Slider(0, 8, value=4.0, step=0.05, label="Layer 11")
                        cond_l12 = gr.Slider(0, 8, value=1.0, step=0.05, label="Layer 12")
                with gr.Accordion("Manual checkpoint paths", open=False):
                    turbo_path = gr.Textbox(value=os.getenv("OSS_TURBO", str(CHECKPOINT_DIR / "turbo.safetensors")), label="Turbo safetensors path")
                    raw_path = gr.Textbox(value=os.getenv("OSS_RAW", str(CHECKPOINT_DIR / "raw.safetensors")), label="RAW safetensors path")
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    warmup_btn = gr.Button("Warm up current settings")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Output", columns=2, height=720)
                zip_file = gr.File(label="Download run ZIP")
                status = gr.Textbox(label="Status", lines=4, elem_id="status_box")

        preset_btn.click(
            model_preset,
            inputs=[checkpoint],
            outputs=[width, height, steps, cfg, y1, y2, mu, use_mu],
        )
        safe_16gb_btn.click(
            safe_16gb_preset,
            inputs=[],
            outputs=[checkpoint, width, height, steps, cfg, y1, y2, mu, use_mu, num_images, dtype_name],
        )
        speed_btn.click(
            speed_preset,
            inputs=[speed_mode],
            outputs=[checkpoint, width, height, steps, cfg, y1, y2, mu, use_mu, num_images, dtype_name],
        )
        cond_layer_sliders = [
            cond_l01, cond_l02, cond_l03, cond_l04, cond_l05, cond_l06,
            cond_l07, cond_l08, cond_l09, cond_l10, cond_l11, cond_l12,
        ]
        cond_preset_btn.click(
            conditioning_preset,
            inputs=[cond_preset],
            outputs=[cond_rebalance_enabled, cond_multiplier, cond_per_layer_weights, *cond_layer_sliders],
        )
        generate_btn.click(
            generate,
            inputs=[
                prompt, checkpoint, width, height, steps, cfg, y1, y2, mu, use_mu,
                num_images, seed, dtype_name, raw_path, turbo_path,
                cond_rebalance_enabled, cond_multiplier, cond_per_layer_weights, cond_use_layer_sliders,
                *cond_layer_sliders,
            ],
            outputs=[gallery, zip_file, status],
        )
        warmup_btn.click(
            warmup_current_settings,
            inputs=[checkpoint, width, height, cfg, y1, y2, mu, use_mu, dtype_name, raw_path, turbo_path],
            outputs=[status],
        )

    with gr.Tab("Model Tools"):
        gr.Markdown(
            "Runtime is hard-offline by default. Generate will only use local files. "
            "Prepare/download models once with `docker compose --profile prepare run --rm model-prep`, "
            "or manually place files into `./checkpoints` and `./models`."
        )
        status_btn = gr.Button("Check local model files")
        model_status = gr.Textbox(label="Local model status", lines=12, elem_id="status_box")
        status_btn.click(local_model_status, outputs=[model_status])
        demo.load(local_model_status, outputs=[model_status])

        gr.Markdown(
            "Optional online/dev download. This is intentionally blocked when `KREA2_OFFLINE_MODE=1`. "
            "Use the prepare service for normal local/offline setup."
        )
        which = gr.Radio(["Turbo", "RAW"], value="Turbo", label="Checkpoint to download")
        token = gr.Textbox(label="HF token", type="password", placeholder="Optional if already authenticated / public access works")
        dl_btn = gr.Button("Download selected checkpoint - online/dev mode only")
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
            "This UI exposes the official open Krea-2 inference controls: prompt, RAW/Turbo checkpoint selection, steps, CFG, y1/y2 timestep shift, fixed mu, width, height, image count, seed, and output prefix/history. It also includes an optional experimental conditioning-rebalance wrapper that scales the positive Qwen conditioning tensor before sampling. The wrapper supports both official Krea-2 4D conditioning tensors and flattened ComfyUI-style tensors, and reports before/after tensor stats in the status box.\n\n"
            "The open Krea-2 repository does not include a full local clone of every hosted Krea product feature such as video generation, enhancer, realtime canvas, motion transfer, lip sync, or 3D tools. It also does not ship a local LoRA trainer in the official inference repo. For LoRA workflows, train on RAW using a trainer such as Diffusers/Ostris/Kohya, then use Turbo for inference when compatible support is available."
        )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
        allowed_paths=[str(OUTPUT_DIR.resolve())],
    )
