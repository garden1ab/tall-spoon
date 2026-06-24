#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints

echo "This helper uses huggingface-cli from the Docker container. Prefer the UI Model Tools tab if you want a simpler flow."
echo "Make sure you accepted the Krea-2 license on Hugging Face."

docker compose run --rm krea2-ui bash -lc '
  uv run --with huggingface_hub python - <<PY
from app.download_models import download_checkpoint
print("Turbo:", download_checkpoint("turbo"))
print("RAW:", download_checkpoint("raw"))
PY
'
