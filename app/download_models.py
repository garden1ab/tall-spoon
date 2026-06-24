from __future__ import annotations

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "/workspace/checkpoints"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "turbo": ("krea/Krea-2-Turbo", "turbo.safetensors"),
    "raw": ("krea/Krea-2-Raw", "raw.safetensors"),
}


def download_checkpoint(which: str, token: str | None = None) -> str:
    if which not in MODEL_FILES:
        raise ValueError(f"Unknown checkpoint: {which}")

    repo_id, filename = MODEL_FILES[which]
    out_path = CHECKPOINT_DIR / filename
    if out_path.exists() and out_path.stat().st_size > 1024 * 1024:
        return str(out_path)

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(CHECKPOINT_DIR),
        token=token or os.getenv("HF_TOKEN") or None,
    )
    return str(out_path)
