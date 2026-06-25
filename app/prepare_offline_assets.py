from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "/workspace/checkpoints"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/workspace/models"))
TEXT_ENCODER_REPO = os.getenv("KREA2_TEXT_ENCODER_REPO", "Qwen/Qwen3-VL-4B-Instruct")
TEXT_ENCODER_PATH = Path(os.getenv("KREA2_TEXT_ENCODER_PATH", str(MODEL_DIR / "Qwen-Qwen3-VL-4B-Instruct")))
VAE_REPO = os.getenv("KREA2_VAE_REPO", "Qwen/Qwen-Image")
VAE_PATH = Path(os.getenv("KREA2_VAE_PATH", str(MODEL_DIR / "Qwen-Qwen-Image")))
HF_TOKEN = os.getenv("HF_TOKEN") or None

CHECKPOINTS = {
    "turbo": ("krea/Krea-2-Turbo", "turbo.safetensors"),
    "raw": ("krea/Krea-2-Raw", "raw.safetensors"),
}


def download_checkpoint(which: str) -> Path:
    repo_id, filename = CHECKPOINTS[which]
    out = CHECKPOINT_DIR / filename
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1024 * 1024:
        print(f"checkpoint {which}: already present at {out}")
        return out
    print(f"checkpoint {which}: downloading {repo_id}/{filename} -> {CHECKPOINT_DIR}")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(CHECKPOINT_DIR),
        token=HF_TOKEN,
    )
    return out


def download_text_encoder() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"text encoder: downloading snapshot {TEXT_ENCODER_REPO} -> {TEXT_ENCODER_PATH}")
    snapshot_download(
        repo_id=TEXT_ENCODER_REPO,
        local_dir=str(TEXT_ENCODER_PATH),
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
    )
    return TEXT_ENCODER_PATH


def download_vae() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"VAE: downloading snapshot {VAE_REPO} vae/* -> {VAE_PATH}")
    snapshot_download(
        repo_id=VAE_REPO,
        local_dir=str(VAE_PATH),
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
        allow_patterns=["vae/*"],
    )
    return VAE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Krea-2 local/offline model files.")
    parser.add_argument("--turbo", action="store_true", help="download Krea-2 Turbo checkpoint")
    parser.add_argument("--raw", action="store_true", help="download Krea-2 RAW checkpoint")
    parser.add_argument("--text-encoder", action="store_true", help="download Qwen text encoder snapshot")
    parser.add_argument("--vae", action="store_true", help="download Qwen-Image VAE snapshot")
    parser.add_argument("--all", action="store_true", help="download Turbo, RAW, text encoder, and VAE")
    args = parser.parse_args()

    any_selected = args.all or args.turbo or args.raw or args.text_encoder or args.vae
    if not any_selected:
        # Practical default for 16 GB GPUs: Turbo + text encoder + VAE. RAW is huge and slow.
        args.turbo = True
        args.text_encoder = True
        args.vae = True

    if args.all or args.text_encoder:
        download_text_encoder()
    if args.all or args.vae:
        download_vae()
    if args.all or args.turbo:
        download_checkpoint("turbo")
    if args.all or args.raw:
        download_checkpoint("raw")

    print("offline asset preparation complete")
    print(f"checkpoint dir: {CHECKPOINT_DIR}")
    print(f"model dir: {MODEL_DIR}")
    print(f"text encoder path: {TEXT_ENCODER_PATH}")
    print(f"VAE path: {VAE_PATH}")


if __name__ == "__main__":
    main()
