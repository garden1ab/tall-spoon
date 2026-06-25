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

# Normal startup should not download or resolve Python packages.
# Set KREA2_REPAIR_DEPS=1 only when you intentionally want to repair/reinstall app deps.
if [[ "${KREA2_REPAIR_DEPS:-0}" == "1" ]]; then
  echo "KREA2_REPAIR_DEPS=1 set; reinstalling app dependencies into the existing venv..."
  uv pip install --python /workspace/krea-2/.venv/bin/python -r /workspace/requirements-app.txt
fi

exec /workspace/krea-2/.venv/bin/python /workspace/app/app.py
