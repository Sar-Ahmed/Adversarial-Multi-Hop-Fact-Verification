#!/usr/bin/env bash
# Idempotent model downloader.
# Phase 00 ships this as a stub; phases that introduce a model uncomment / add their lines.

set -euo pipefail

CKPT_DIR="$(cd "$(dirname "$0")/.." && pwd)/checkpoints"
mkdir -p "$CKPT_DIR"

echo "[fetch_models] checkpoint dir: $CKPT_DIR"

# === Phase 03 / 07: Qwen 2.5 3B Q4 GGUF (uncomment when Phase 03 lands) ===
# QWEN_GGUF="$CKPT_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"
# if [ ! -f "$QWEN_GGUF" ]; then
#   echo "[fetch_models] downloading Qwen2.5-3B-Instruct Q4_K_M..."
#   curl -L -o "$QWEN_GGUF" \
#     "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
# else
#   echo "[fetch_models] Qwen GGUF already present"
# fi

# === Phase 04: bge-small + bge-reranker-base (downloaded lazily by sentence-transformers) ===
# These auto-download on first use into the HuggingFace cache; no explicit fetch needed
# unless we want to pre-warm the cache or pin to specific revisions.

# === Phase 06 / 07 / 08: cross-encoder/nli-deberta-v3-base ===
# Same: auto-downloaded on first use.

echo "[fetch_models] done. Phase 00: nothing to download yet."
