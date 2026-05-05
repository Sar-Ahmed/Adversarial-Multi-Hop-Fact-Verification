"""Phase 02 reranker stub: identity (returns the input top-K unchanged).

Replaced in Phase 04 by `BAAI/bge-reranker-base` cross-encoder over the
top-50 dense pool, narrowed to top-10.
"""

from __future__ import annotations

from src.schema import Passage


class StubReranker:
    """No-op reranker: pass-through with optional truncation."""

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    def rerank(
        self,
        query: str,  # noqa: ARG002
        candidates: list[Passage],
        top_k: int | None = None,
    ) -> list[Passage]:
        k = top_k or self.top_k
        return list(candidates)[:k]
