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
- VRAM unload button
- Basic local prompt guard

## What it does not provide

The open Krea-2 release is not the entire hosted Krea creative suite. This local UI is for text-to-image inference with the open Krea-2 checkpoints. Hosted Krea features such as video generation, enhancer, realtime canvas, motion transfer, lip sync, and 3D are not included in the open inference repo.

## Hardware notes

Krea-2 uses a large 12B-class diffusion model. The Turbo and RAW safetensor files are each about 26 GB. A high-VRAM NVIDIA GPU is strongly recommended. Start with Turbo at 1024x1024 before trying 2048x2048.

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

## Troubleshooting

### CUDA is not available inside the container

Check that NVIDIA Container Toolkit is installed and working:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

### Out of memory

Use Turbo, reduce resolution to 1024x1024, generate one image at a time, then click `Unload model from VRAM` before switching checkpoints.

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
