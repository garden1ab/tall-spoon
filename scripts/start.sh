#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/checkpoints /workspace/outputs /workspace/.cache/huggingface

# Let users mount checkpoints without editing .env.
if [[ -z "${OSS_TURBO:-}" && -f "/workspace/checkpoints/turbo.safetensors" ]]; then
  export OSS_TURBO="/workspace/checkpoints/turbo.safetensors"
fi
if [[ -z "${OSS_RAW:-}" && -f "/workspace/checkpoints/raw.safetensors" ]]; then
  export OSS_RAW="/workspace/checkpoints/raw.safetensors"
fi

cd /workspace/krea-2
uv run \
  --with "gradio>=4.44,<6" \
  --with "huggingface_hub>=0.25" \
  --with "psutil>=5.9" \
  /workspace/app/app.py
