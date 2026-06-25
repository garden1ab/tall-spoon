#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/checkpoints /workspace/models /workspace/outputs /workspace/.cache/huggingface /workspace/.cache/torchinductor /workspace/.cache/triton

# Default runtime is hard-offline. This keeps Generate from contacting Hugging Face
# or any model URL after the model files have been prepared locally.
export KREA2_OFFLINE_MODE="${KREA2_OFFLINE_MODE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
if [[ "$KREA2_OFFLINE_MODE" == "1" ]]; then
  export HF_HUB_OFFLINE="1"
  export TRANSFORMERS_OFFLINE="1"
  export HF_DATASETS_OFFLINE="1"
fi

# Let users mount checkpoints without editing .env.
if [[ -z "${OSS_TURBO:-}" && -f "/workspace/checkpoints/turbo.safetensors" ]]; then
  export OSS_TURBO="/workspace/checkpoints/turbo.safetensors"
fi
if [[ -z "${OSS_RAW:-}" && -f "/workspace/checkpoints/raw.safetensors" ]]; then
  export OSS_RAW="/workspace/checkpoints/raw.safetensors"
fi
if [[ -z "${MODEL_DIR:-}" ]]; then
  export MODEL_DIR="/workspace/models"
fi
if [[ -z "${KREA2_TEXT_ENCODER_PATH:-}" ]]; then
  export KREA2_TEXT_ENCODER_PATH="/workspace/models/Qwen-Qwen3-VL-4B-Instruct"
fi
if [[ -z "${KREA2_VAE_PATH:-}" ]]; then
  export KREA2_VAE_PATH="/workspace/models/Qwen-Qwen-Image"
fi

cd /workspace/krea-2

# Normal startup should not download or resolve Python packages.
# Set KREA2_REPAIR_DEPS=1 only when you intentionally want to repair/reinstall app deps.
if [[ "${KREA2_REPAIR_DEPS:-0}" == "1" ]]; then
  echo "KREA2_REPAIR_DEPS=1 set; reinstalling app dependencies into the existing venv..."
  uv pip install --python /workspace/krea-2/.venv/bin/python -r /workspace/requirements-app.txt
fi

exec /workspace/krea-2/.venv/bin/python /workspace/app/app.py
