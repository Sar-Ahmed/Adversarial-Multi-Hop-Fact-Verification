"""Phase 04 retrieval / reranker tests.

Most tests are fast unit tests (no model loading). Three integration tests
are marked `slow` — they exercise BM25 (loads tokenized index ~30 s on first
call) and the cross-encoder (~10 s model load).
"""

from __future__ import annotations

import pytest

from src.eval.metrics import bootstrap_ci, recall_at_k
from src.schema import Passage

# === Pure-Python metric helpers ==============================================


def test_recall_at_k_full_recall() -> None:
    retrieved = ["a::1", "b::2", "c::3", "d::4", "e::5"]
    gold = {"a::1", "b::2"}
    assert recall_at_k(retrieved, gold, 5) == 1.0


def test_recall_at_k_partial() -> None:
    retrieved = ["a::1", "b::2", "c::3"]
    gold = {"a::1", "x::1"}
    assert recall_at_k(retrieved, gold, 3) == 0.5


def test_recall_at_k_misses_outside_top_k() -> None:
    retrieved = ["a::1", "b::2", "c::3", "d::4", "e::5"]
    gold = {"e::5"}
    # gold is in top-5 but not top-3
    assert recall_at_k(retrieved, gold, 3) == 0.0
    assert recall_at_k(retrieved, gold, 5) == 1.0


def test_recall_at_k_handles_empty_gold() -> None:
    assert recall_at_k(["a::1"], set(), 5) == 0.0


def test_bootstrap_ci_on_constant_input_gives_zero_width() -> None:
    point, lo, hi = bootstrap_ci([0.5] * 50, n=200, seed=42)
    assert point == 0.5
    assert lo == 0.5
    assert hi == 0.5


def test_bootstrap_ci_returns_sane_bounds() -> None:
    values = [float(i) / 100 for i in range(100)]
    point, lo, hi = bootstrap_ci(values, n=500, seed=42)
    assert 0.45 <= point <= 0.55
    assert lo < point < hi


# === Reranker output shape (no model load needed) ============================


def test_reranker_handles_empty_candidates() -> None:
    """Empty input → empty output, no model load."""
    from src.config import RerankerConfig
    from src.reranker.cross_encoder import CrossEncoderReranker

    rr = CrossEncoderReranker(RerankerConfig(model="dummy", top_k=10))
    assert rr.rerank("any query", []) == []


# === Slow integration tests ===================================================


@pytest.fixture(scope="module")
def cfg():  # noqa: ANN201
    from pathlib import Path

    from src.config import PipelineConfig

    return PipelineConfig.load(Path("configs/default.yaml"))


@pytest.mark.slow
def test_bm25_returns_top_k_descending(cfg) -> None:  # noqa: ANN001
    from src.retrieval.bm25 import BM25Retriever

    bm25 = BM25Retriever(cfg.corpus, top_k=10)
    out = bm25.retrieve("Christopher Nolan directed Inception.", top_k=10)
    assert len(out) == 10
    assert all(isinstance(p, Passage) for p in out)
    scores = [p.score for p in out]
    assert scores == sorted(scores, reverse=True), "BM25 results must be sorted descending"


@pytest.mark.slow
def test_dense_returns_normalized_scores(cfg) -> None:  # noqa: ANN001
    from src.retrieval.dense import DenseRetriever

    dense = DenseRetriever(cfg.retriever, cfg.corpus)
    out = dense.retrieve("Christopher Nolan directed Inception.", top_k=5)
    assert len(out) == 5
    for p in out:
        assert -1.0 <= p.score <= 1.0, f"cosine score out of [-1, 1]: {p.score}"


@pytest.mark.slow
def test_inception_query_top_3_includes_inception_article(cfg) -> None:  # noqa: ANN001
    from src.reranker.cross_encoder import CrossEncoderReranker
    from src.retrieval.dense import DenseRetriever

    dense = DenseRetriever(cfg.retriever, cfg.corpus)
    rerank = CrossEncoderReranker(cfg.reranker)
    candidates = dense.retrieve("Christopher Nolan directed Inception.", top_k=50)
    top10 = rerank.rerank("Christopher Nolan directed Inception.", candidates, top_k=10)
    titles = {p.title for p in top10[:3]}
    assert any("Inception" in t for t in titles), f"Inception not in reranked top-3; got {titles}"
