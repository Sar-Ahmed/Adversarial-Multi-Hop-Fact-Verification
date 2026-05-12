"""Sanity-check the cosine distribution for HoVer dev claims.

Phase 06 mining found 0 distractors above cos≥0.85. Likely cause: long
multi-hop HoVer claims (25-40 words) yield lower top-1 cos than the short
QA-style queries bge-small was trained on. This script measures it directly.
"""

from __future__ import annotations


def main() -> None:
    """Print the top-1, top-10, top-50 cosines for 5 HoVer dev claims."""
    import random

    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    from src.config import PipelineConfig
    from src.data.load import load_hover

    cfg = PipelineConfig.load("configs/default.yaml")
    splits = load_hover()
    dev = [ex for ex in splits["validation"] if ex.supporting_facts]
    rng = random.Random(42)
    sample = rng.sample(dev, 5)

    df = pd.read_parquet(cfg.corpus.parquet_path)
    embeddings = np.load(cfg.corpus.embeddings_path)
    encoder = SentenceTransformer(cfg.retriever.encoder, device="cpu")
    encoder.max_seq_length = 256
    prefix = cfg.retriever.query_prefix or ""

    print(f"{'claim_words':>11} | top-1   top-5   top-10  top-50  top-100  top-200")
    for ex in sample:
        n_words = len(ex.claim.split())
        qv = encoder.encode(
            [prefix + ex.claim], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)[0]
        sims = embeddings @ qv
        top200 = np.sort(sims)[::-1][:200]
        print(
            f"{n_words:>11} | "
            f"{top200[0]:.4f}  {top200[4]:.4f}  {top200[9]:.4f}  "
            f"{top200[49]:.4f}  {top200[99]:.4f}  {top200[199]:.4f}    "
            f"{ex.claim[:60]!r}"
        )


if __name__ == "__main__":
    main()
