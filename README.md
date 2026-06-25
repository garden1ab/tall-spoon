# Krea-2 Local Docker UI

A local Docker/Gradio interface for Krea-2 RAW and Turbo image generation using Krea's official open inference repository.

## What this provides

- Browser UI at `http://localhost:7860`
- Krea-2 Turbo and RAW checkpoint selection
- Prompt, width, height, seed, image count
- Official sampler controls: steps, CFG, y1, y2, fixed `mu`
- Recommended presets for Turbo and RAW
- One-time online model preparation service for local/offline runtime
- Hard-offline Generate path: local checkpoint files and local Qwen text encoder snapshot only
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

## Setup: local/offline runtime

The normal UI runtime is hard-offline. `Generate` will not download Krea, Qwen, Gradio, Python packages, or Hugging Face files. All online work happens in one of these two explicit phases:

1. Docker image build: installs Linux/Python dependencies and clones the Krea-2 inference repo.
2. One-time model prep: downloads the Qwen text encoder snapshot and Krea checkpoint files into local mounted folders.

Build the image:

```bash
cp .env.example .env
docker compose build
```

While internet is available, prepare the local model cache:

```bash
docker compose --profile prepare run --rm model-prep
```

Default prep downloads:

```text
Qwen/Qwen3-VL-4B-Instruct -> ./models/Qwen-Qwen3-VL-4B-Instruct
Qwen/Qwen-Image vae/* -> ./models/Qwen-Qwen-Image/vae
krea/Krea-2-Turbo turbo.safetensors -> ./checkpoints/turbo.safetensors
```

RAW is off by default because it is also large and is not recommended for a 16 GB GPU. To include RAW, edit `.env`:

```env
KREA2_PREP_RAW=1
```

Then run the prep command again.

Start the local-only runtime:

```bash
docker compose up
```

Open:

```text
http://localhost:7860
```

The UI service is attached to an internal Docker network and binds Gradio only to localhost: `127.0.0.1:7860`. It also sets:

```env
KREA2_OFFLINE_MODE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

This means if the text encoder or checkpoint is missing, generation fails with a local-file error instead of attempting a URL call.

## Manual model placement

You can also download manually and place files here:

```text
./checkpoints/turbo.safetensors
./checkpoints/raw.safetensors
./models/Qwen-Qwen3-VL-4B-Instruct/
./models/Qwen-Qwen-Image/vae/
```

The Qwen text encoder folder must be a full Hugging Face snapshot that can be loaded by `transformers.from_pretrained(local_path)`, including config, tokenizer files, and model shard files. The Qwen-Image VAE folder must include `vae/config.json` and the VAE weights.

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

### Generate still tries to make a URL/model call

Use this build. The app no longer passes `Qwen/Qwen3-VL-4B-Instruct` directly to Krea's pipeline during normal runtime. It resolves the text encoder to this local path instead:

```text
/workspace/models/Qwen-Qwen3-VL-4B-Instruct
```

It also patches the official Qwen VAE loader so it uses this local path instead of `Qwen/Qwen-Image`:

```text
/workspace/models/Qwen-Qwen-Image/vae
```

If either folder is missing or incomplete and `KREA2_OFFLINE_MODE=1`, the UI raises a local-file error and does not call Hugging Face.

Check status in the UI under **Model Tools > Check local model files**, or from the host:

```bash
ls -lah ./checkpoints
ls -lah ./models/Qwen-Qwen3-VL-4B-Instruct
ls -lah ./models/Qwen-Qwen-Image/vae
```

### Python packages download every time `docker compose up` starts

That means the old startup script is still being used. This build installs UI dependencies during `docker compose build` from `requirements-app.txt`, then starts with the already-built virtual environment:

```bash
/workspace/krea-2/.venv/bin/python /workspace/app/app.py
```

Use a no-cache rebuild if startup still shows `Downloading gradio`, `Downloading pandas`, `Downloading ruff`, or similar package logs:

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

### `accelerate` warning during model loading

`accelerate>=0.34` is now installed into the image at build time. Rebuild the image if the warning persists.

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

Run the one-time prep service while online, or manually place the safetensors files into `./checkpoints`. In hard-offline runtime, the Model Tools download button is blocked on purpose.

```bash
docker compose --profile prepare run --rm model-prep
```

### First generation is slow

The first run still loads the Krea checkpoint and initializes the local Qwen text encoder from disk. That is not a network call. Later generations are faster while the model remains cached in VRAM. Use **Warm up current settings** after selecting a resolution if you want to pay the model load/compile cost before a real prompt.

## Updating Krea-2 official code

Rebuild without Docker cache:

```bash
docker compose build --no-cache
```

## License and safety

You are responsible for complying with the Krea-2 Community License and Acceptable Use Policy. This project includes only a minimal local prompt guard and is not production moderation.

## Performance notes

Krea-2 is a large model. On a 16 GB GPU, use Turbo first and avoid RAW until the Turbo path is stable.

Recommended speed settings:

- Checkpoint: `Turbo`
- Resolution: `768x768` for balanced speed, `512x512` for testing
- Steps: `6-8`
- CFG: `0.0` for Turbo
- Images: `1`
- dtype: `bfloat16`, or try `float16` if your GPU is faster with fp16

The first generation for a resolution can be much slower because PyTorch/Triton compiles CUDA kernels. Later generations with the same model, dtype, and resolution should be faster because the model and compile cache are reused.

Use **Warm up current settings** after selecting a resolution if you want to pay the model load/compile cost before a real prompt.

If first-run latency is more important than repeated-generation speed, set this in `.env`:

```env
KREA2_DISABLE_TORCH_COMPILE=1
```

Then rebuild/restart:

```bash
docker compose down
docker compose up --build
```

For Turbo, avoid raising CFG above `0.0`. CFG greater than zero runs an unconditional branch and roughly doubles the denoising model work.
