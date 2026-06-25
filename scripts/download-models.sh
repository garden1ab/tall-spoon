#!/usr/bin/env bash
set -euo pipefail

echo "Preparing Krea-2 offline assets with the online prep service."
echo "Default downloads: Qwen text encoder + Qwen-Image VAE + Krea-2 Turbo checkpoint."
echo "Set KREA2_PREP_RAW=1 in .env if you also want RAW."

docker compose --profile prepare run --rm model-prep
