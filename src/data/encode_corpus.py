"""Encode the corpus with bge-small-en-v1.5 and build a FAISS flat-IP index.

Reads:  artifacts/corpus.parquet
Writes: artifacts/corpus_embeddings.npy  (float32 [N, 384], L2-normalized)
        artifacts/corpus.faiss            (FAISS IndexFlatIP)

Deterministic given the same input. CPU-only path (~30-60 min for 200k passages
on a modern laptop CPU; faster on GPU if `device='cuda'` is set in config).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
CORPUS_IN = ARTIFACTS / "corpus.parquet"
EMB_OUT = ARTIFACTS / "corpus_embeddings.npy"
FAISS_OUT = ARTIFACTS / "corpus.faiss"
META_OUT = ARTIFACTS / "corpus_encoding_meta.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64
EMBED_DIM = 384


def encode() -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    if not CORPUS_IN.exists():
        raise FileNotFoundError(f"{CORPUS_IN} not found — run build_corpus.py first")

    df = pd.read_parquet(CORPUS_IN)
    texts = df["text"].tolist()
    logger.info("loaded {} passages from {}", len(texts), CORPUS_IN)

    logger.info("loading encoder: {}", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = 256  # bge-small was trained at 512 but 256 is enough for sentences

    logger.info("encoding (batch_size={}, normalize=True)...", BATCH_SIZE)
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), 1024), desc="encode"):
        batch_texts = texts[start : start + 1024]
        emb = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        chunks.append(emb.astype(np.float32))

    embeddings = np.vstack(chunks)
    assert embeddings.shape == (
        len(texts),
        EMBED_DIM,
    ), f"unexpected shape {embeddings.shape}, expected ({len(texts)}, {EMBED_DIM})"
    # Sanity: every vector L2-normalized to ~1.0
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        bad = np.sum(np.abs(norms - 1.0) > 1e-3)
        raise RuntimeError(f"{bad} vectors not L2-normalized")

    np.save(EMB_OUT, embeddings)
    logger.info("wrote {} ({:.1f} MB)", EMB_OUT, EMB_OUT.stat().st_size / 1e6)

    logger.info("building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_OUT))
    logger.info("wrote {} (ntotal={})", FAISS_OUT, index.ntotal)

    META_OUT.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "n_passages": int(len(texts)),
                "embed_dim": EMBED_DIM,
                "batch_size": BATCH_SIZE,
                "normalize": True,
                "max_seq_length": model.max_seq_length,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    sanity_check(df, model, index)


def sanity_check(df: pd.DataFrame, model, index) -> None:  # noqa: ANN001
    """Encode an Inception claim, retrieve top-5, log results."""
    query = "Christopher Nolan directed Inception."
    qv = model.encode(
        ["Represent this sentence for searching relevant passages: " + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    scores, ids = index.search(qv, 5)
    logger.info("=== sanity check: {!r} ===", query)
    inception_hit = False
    for s, i in zip(scores[0], ids[0], strict=True):
        row = df.iloc[i]
        text_excerpt = row["text"][:80] + ("..." if len(row["text"]) > 80 else "")
        logger.info(
            "  {:.4f}  {} :: sent {} :: {}",
            float(s),
            row["title"],
            int(row["sent_idx"]),
            text_excerpt,
        )
        if "Inception" in row["title"]:
            inception_hit = True
    if not inception_hit:
        logger.warning("Inception not found in top-5 — investigate before proceeding")
    else:
        logger.info("Inception found in top-5 (sanity check passed)")


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(42)
    setup_logging()
    encode()
