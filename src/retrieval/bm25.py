"""BM25 retriever over the focused Wikipedia corpus.

Used as the retrieval baseline (Phase 04) and as the hard-negative source
for retriever fine-tuning (Phase 05). Tokenization of 177k passages is the
dominant cost on first run; we cache the tokenized BM25 index to disk so
subsequent runs skip the work.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from src.schema import Passage

if TYPE_CHECKING:
    from src.config import CorpusConfig

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + alnum-token split; what BM25Okapi expects."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Lazy-loaded BM25 retriever; same `retrieve()` signature as DenseRetriever."""

    def __init__(self, corpus_cfg: CorpusConfig, top_k: int = 50) -> None:
        self.corpus_cfg = corpus_cfg
        self.top_k = top_k
        self._bm25 = None
        self._df = None

    def _ensure_loaded(self) -> None:
        if self._bm25 is not None:
            return

        import pandas as pd

        df = pd.read_parquet(self.corpus_cfg.parquet_path)
        cache_path = Path(self.corpus_cfg.parquet_path).parent / "bm25_index.pkl"

        if cache_path.exists():
            logger.info("loading BM25 cache: {}", cache_path)
            with open(cache_path, "rb") as f:
                self._bm25 = pickle.load(f)  # noqa: S301 — trusted local cache
        else:
            from rank_bm25 import BM25Okapi

            logger.info("tokenizing {} passages for BM25 (one-time)", len(df))
            tokens = [_tokenize(t) for t in df["text"].tolist()]
            self._bm25 = BM25Okapi(tokens)
            logger.info("saving BM25 cache: {}", cache_path)
            with open(cache_path, "wb") as f:
                pickle.dump(self._bm25, f)

        self._df = df
        logger.info("BM25 ready: {} passages", len(df))

    def retrieve(self, text: str, top_k: int | None = None) -> list[Passage]:
        """Return the top-K passages by BM25 score, sorted descending."""
        self._ensure_loaded()
        k = top_k or self.top_k
        scores = self._bm25.get_scores(_tokenize(text))  # type: ignore[union-attr]

        # argpartition + argsort gives top-K by score in descending order.
        if k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            partition = np.argpartition(-scores, k - 1)[:k]
            top_idx = partition[np.argsort(-scores[partition])]

        results: list[Passage] = []
        for i in top_idx:
            row = self._df.iloc[int(i)]  # type: ignore[union-attr]
            results.append(
                Passage(
                    doc_id=str(row["doc_id"]),
                    title=str(row["title"]),
                    sent_idx=int(row["sent_idx"]),
                    text=str(row["text"]),
                    score=float(scores[int(i)]),
                )
            )
        return results
