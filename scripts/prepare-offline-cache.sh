#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/checkpoints /workspace/models /workspace/.cache/huggingface
cd /workspace/krea-2

ARGS=()
if [[ "${KREA2_PREP_ALL:-0}" == "1" ]]; then ARGS+=(--all); fi
if [[ "${KREA2_PREP_TEXT_ENCODER:-1}" == "1" ]]; then ARGS+=(--text-encoder); fi
if [[ "${KREA2_PREP_VAE:-1}" == "1" ]]; then ARGS+=(--vae); fi
if [[ "${KREA2_PREP_TURBO:-1}" == "1" ]]; then ARGS+=(--turbo); fi
if [[ "${KREA2_PREP_RAW:-0}" == "1" ]]; then ARGS+=(--raw); fi

exec /workspace/krea-2/.venv/bin/python /workspace/app/prepare_offline_assets.py "${ARGS[@]}"
