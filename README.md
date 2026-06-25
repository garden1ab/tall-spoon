# Krea-2 Local Docker UI

A local Docker/Gradio interface for Krea-2 RAW and Turbo image generation using Krea's official open inference repository.

## What this provides

- Browser UI at `http://localhost:7860`
- Krea-2 Turbo and RAW checkpoint selection
- Prompt, width, height, seed, image count
- Official sampler controls: steps, CFG, y1, y2, fixed `mu`
- Recommended presets for Turbo and RAW
- Model download helper for `turbo.safetensors` and `raw.safetensors`
- Output history and ZIP export
- Experimental conditioning rebalance toggle with preset/manual 12-layer weights
- VRAM unload button
- 16 GB safe preset button for WSL2 / lower-VRAM cards
- Basic local prompt guard

## What it does not provide

The open Krea-2 release is not the entire hosted Krea creative suite. This local UI is for text-to-image inference with the open Krea-2 checkpoints. Hosted Krea features such as video generation, enhancer, realtime canvas, motion transfer, lip sync, and 3D are not included in the open inference repo.

## Hardware notes

Krea-2 uses a large 12B-class diffusion model. The Turbo and RAW safetensor files are each about 26 GB. A high-VRAM NVIDIA GPU is strongly recommended. On 16 GB GPUs, start with the **Apply 16 GB safe preset** button: Turbo, bfloat16, 768x768, 1 image, 8 steps, CFG 0.0. After that works, try 896x896 or 1024x1024.

## Prerequisites

1. Docker Desktop or Docker Engine.
2. NVIDIA GPU driver.
3. NVIDIA Container Toolkit.
4. Enough disk space for Docker layers, Hugging Face cache, the Qwen text encoder, and Krea checkpoints.
5. Optional but recommended: Hugging Face token, after accepting the Krea-2 model license.

## Setup

```bash
cp .env.example .env
docker compose build
docker compose up
```

Open:

```text
http://localhost:7860
```

## Download checkpoints

Option A: Use the UI.

1. Open the `Model Tools` tab.
2. Select `Turbo`.
3. Paste `HF_TOKEN` if needed.
4. Click `Download selected checkpoint`.

Option B: Download manually and place files here:

```text
./checkpoints/turbo.safetensors
./checkpoints/raw.safetensors
```

The container maps these to:

```text
/workspace/checkpoints/turbo.safetensors
/workspace/checkpoints/raw.safetensors
```

## Experimental conditioning rebalance

The Generate tab includes an optional **Conditioning rebalance / experimental weights** section. This ports the intent of the ComfyUI-style Krea-2 conditioning rebalance node into the local Krea-2 sampler.

What it does:

- Adds an **Enable conditioning rebalance** toggle.
- Adds a **Weight preset** selector and **Apply weight preset** button.
- Adds a global conditioning multiplier.
- Adds a manual comma-separated weight field.
- Adds 12 layer sliders for direct tuning of the expected 12 Qwen3-VL conditioning taps.
- Saves the selected conditioning settings into each run's `metadata.json`.

Implementation detail: the official Krea sampler encodes the prompt internally and sends `txt` into `model(..., context=txt, ...)`. This UI adds a local sampler wrapper that scales only the positive text conditioning tensor after `encoder(prompts)` and before denoising. Negative CFG conditioning is left untouched.

The default experimental weights are:

```text
1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0
```

Start with the toggle disabled. Then test the same prompt/seed with `Krea2 ComfyUI default`, `Late layer detail boost`, and `Balanced structure boost`. If outputs become overcooked or unstable, lower the global multiplier first.

## Recommended generation settings

Turbo:

```text
steps: 8
cfg: 0.0
mu: 1.15
resolution: 1024-2048 square or rectangular, multiples of 16
```

RAW:

```text
steps: 52
cfg: 3.5
mu: disabled
resolution: 1024x1024 recommended
```

## WSL2 memory config note

If WSL prints an error like:

```text
wsl: Expected ' ' or '\n' in C:\Users\AI Beast\.wslconfig:1
```

then your Windows-side `.wslconfig` is malformed. Shut down WSL and replace `C:\Users\AI Beast\.wslconfig` with a valid file like this:

```ini
[wsl2]
memory=24GB
processors=8
swap=32GB
localhostForwarding=true
```

Then run this in PowerShell:

```powershell
wsl --shutdown
```

Reopen Ubuntu/WSL and start Docker again. This does not increase GPU VRAM, but it helps prevent WSL memory parsing errors and CPU RAM pressure while loading the text encoder/checkpoints.

## Troubleshooting

### `Python.h: No such file or directory` / Triton or Torch Inductor compile error

This means the Docker image was built without Python development headers. This project now installs `python3.12-dev`, `build-essential`, and `pkg-config` in the image. Rebuild the image, not just `docker compose up`:

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

### `accelerate` warning during model loading

The runtime now includes `accelerate>=0.34` in the `uv run` command. Rebuild the image if the warning persists.

### CUDA is not available inside the container

Check that NVIDIA Container Toolkit is installed and working:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

### Gradio `InvalidPathError` for `/workspace/outputs/...png`

This build launches Gradio with `/workspace/outputs` in `allowed_paths`, so generated PNGs and ZIP files can be displayed/downloaded from the mounted output directory. Rebuild if you still see this error:

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

### Out of memory

Use **Apply 16 GB safe preset** first. Keep Turbo selected, use bfloat16, generate one image at a time, and keep CFG at 0.0 for Turbo. Avoid RAW on a 16 GB card unless you add offload/quantization support. If VRAM remains reserved after an error, stop the container and restart it:

```bash
docker compose down
docker compose up
```

This build also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_MODULE_LOADING=LAZY`, and keeps only one Krea-2 pipeline cached in VRAM.

### Model checkpoint not found

Download from the Model Tools tab or manually place the safetensors files into `./checkpoints`.

### First generation is slow

The first run loads the Krea checkpoint and downloads/initializes the Qwen text encoder. Later generations are faster while the model remains cached in VRAM.

## Updating Krea-2 official code

Rebuild without Docker cache:

```bash
docker compose build --no-cache
```

## License and safety

You are responsible for complying with the Krea-2 Community License and Acceptable Use Policy. This project includes only a minimal local prompt guard and is not production moderation.
