"""Cross-encoder reranker (`BAAI/bge-reranker-base`).

Replaces `StubReranker`. Spec failure-mode #1 (surface keyword overlap) is
addressed here — a cross-encoder reads (claim, passage) jointly so it
catches entity-mismatch where bi-encoder cosine cannot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.schema import Passage

if TYPE_CHECKING:
    from src.config import RerankerConfig


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder; identical `rerank()` signature as StubReranker."""

    def __init__(self, cfg: RerankerConfig, batch_size: int = 16, max_length: int = 512) -> None:
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        logger.info("loading cross-encoder reranker: {}", self.cfg.model)
        self._model = CrossEncoder(self.cfg.model, device="cpu", max_length=self.max_length)

    def rerank(
        self, query: str, candidates: list[Passage], top_k: int | None = None
    ) -> list[Passage]:
        if not candidates:
            return []
        self._ensure_loaded()
        k = top_k or self.cfg.top_k

        pairs = [(query, p.text) for p in candidates]
        scores = self._model.predict(  # type: ignore[union-attr]
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Sort descending; replace each Passage's score with the rerank score.
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
        out: list[Passage] = []
        for i in order[:k]:
            p = candidates[i]
            out.append(
                Passage(
                    doc_id=p.doc_id,
                    title=p.title,
                    sent_idx=p.sent_idx,
                    text=p.text,
                    score=float(scores[i]),
                )
            )
        return out
