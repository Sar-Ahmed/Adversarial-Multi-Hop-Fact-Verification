"""Metric helpers — bootstrap CIs and recall@K.

Used by Phase 04 (retrieval eval) and Phase 11 (full pipeline eval). Kept
deliberately small and dependency-light: numpy + Python only.
"""

from __future__ import annotations

import numpy as np


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Per-passage recall: fraction of gold doc_ids found in the first K retrieved.

    For HoVer multi-hop claims (2-4 gold passages), this is the strict metric —
    finding 2 of 3 gold passages = 0.667. Use this for the headline number
    because the verifier needs *all* gold passages to reason correctly, not
    just any one.

    Returns 0.0 when `gold_ids` is empty so callers should filter those out
    *before* averaging.
    """
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & gold_ids) / len(gold_ids)


def hit_rate_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """1.0 if any gold passage is in top-K, else 0.0. The lenient companion to
    recall_at_k. Reported alongside for comparability with retrieval-system
    benchmarks that traditionally use this looser metric."""
    if not gold_ids:
        return 0.0
    return 1.0 if (gold_ids & set(retrieved_ids[:k])) else 0.0


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap mean + 95% CI over `values`.

    Returns (point_estimate, ci_lo, ci_hi) using percentile 2.5 / 97.5.
    """
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs_means = np.empty(n, dtype=np.float64)
    for i in range(n):
        sample = rng.choice(arr, size=arr.size, replace=True)
        bs_means[i] = sample.mean()
    point = float(arr.mean())
    lo = float(np.percentile(bs_means, 2.5))
    hi = float(np.percentile(bs_means, 97.5))
    return point, lo, hi
