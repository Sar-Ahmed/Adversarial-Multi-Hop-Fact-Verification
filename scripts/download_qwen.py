"""Idempotent downloader for the Qwen2.5-3B-Instruct Q4_K_M GGUF.

Used by Phase 03 (decomposer) and Phase 07 (verifier). ~2 GB download.
Uses `huggingface_hub.hf_hub_download` so caching, retries, and progress
are handled by the standard library every contributor already has.

Run: python -m scripts.download_qwen
"""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger

REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
DEST_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def main() -> None:
    from huggingface_hub import hf_hub_download

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    final_path = DEST_DIR / FILENAME

    if final_path.exists() and final_path.stat().st_size > 1_500_000_000:
        logger.info(
            "already present: {} ({:.2f} GB) — skipping download",
            final_path,
            final_path.stat().st_size / 1e9,
        )
        return

    logger.info("downloading {} from {}", FILENAME, REPO_ID)
    cached = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        cache_dir=str(DEST_DIR / ".hf_cache"),
    )
    cached_path = Path(cached)
    logger.info("cached at: {}", cached_path)

    # The HF cache stores under a content-addressable layout; copy/symlink to a
    # stable path that PipelineConfig can reference.
    if not final_path.exists():
        shutil.copy(cached_path, final_path)
    logger.info(
        "ready: {} ({:.2f} GB)",
        final_path,
        final_path.stat().st_size / 1e9,
    )


if __name__ == "__main__":
    from src.utils.logging import setup_logging

    setup_logging(json_sink=False)
    main()
