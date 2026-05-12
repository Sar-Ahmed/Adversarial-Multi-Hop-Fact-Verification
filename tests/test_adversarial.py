"""Phase 06 unit tests for distractor mining helpers + injection.

The mining pipeline itself is integration-tested by the full run that writes
`artifacts/distractors_v3.json`. Here we test the small pure-Python helpers
(top-k filtering, padding fallback) and the inject_distractors function.
"""

from __future__ import annotations

import pytest

from src.adversarial.inject import inject_distractors
from src.adversarial.mine import _pick_top_k
from src.schema import Passage

# === Mining helper ===========================================================


def test_pick_top_k_filters_below_contra_threshold() -> None:
    candidates = [
        {"doc_id": "a", "cos": 0.9, "contra_prob": 0.95},
        {"doc_id": "b", "cos": 0.92, "contra_prob": 0.3},  # below threshold
        {"doc_id": "c", "cos": 0.86, "contra_prob": 0.9},
    ]
    out = _pick_top_k(candidates, k=5, contra_threshold=0.8)
    assert [c["doc_id"] for c in out] == ["a", "c"]


def test_pick_top_k_ranks_by_cos_times_contra() -> None:
    """Highest cos * contra wins, not highest cos or highest contra alone."""
    candidates = [
        {"doc_id": "high_cos_low_contra", "cos": 0.95, "contra_prob": 0.85},  # 0.808
        {"doc_id": "balanced", "cos": 0.90, "contra_prob": 0.95},  # 0.855
        {"doc_id": "low_cos_high_contra", "cos": 0.86, "contra_prob": 0.99},  # 0.851
    ]
    out = _pick_top_k(candidates, k=3, contra_threshold=0.8)
    assert [c["doc_id"] for c in out] == [
        "balanced",
        "low_cos_high_contra",
        "high_cos_low_contra",
    ]


def test_pick_top_k_handles_empty() -> None:
    assert _pick_top_k([], k=5, contra_threshold=0.8) == []


# === Injection ===============================================================


def _p(doc_id: str, score: float = 0.0) -> Passage:
    return Passage(
        doc_id=doc_id, title=f"T_{doc_id}", sent_idx=0, text=f"text_{doc_id}", score=score
    )


def test_inject_mix_inserts_all_distractors() -> None:
    candidates = [_p(f"c{i}", score=1.0 - i * 0.1) for i in range(5)]
    distractors = [_p(f"d{i}", score=0.9) for i in range(3)]
    out = inject_distractors(candidates, distractors, mode="mix", seed=42)
    out_ids = {p.doc_id for p in out}
    assert {"d0", "d1", "d2"}.issubset(out_ids)
    assert len(out) == 8  # mix does not remove anything


def test_inject_mix_preserves_original_candidates() -> None:
    candidates = [_p(f"c{i}") for i in range(5)]
    distractors = [_p(f"d{i}") for i in range(2)]
    out = inject_distractors(candidates, distractors, mode="mix", seed=42)
    for c in candidates:
        assert any(p.doc_id == c.doc_id for p in out)


def test_inject_replace_bottom_drops_lowest_scored() -> None:
    candidates = [_p(f"c{i}", score=1.0 - i * 0.1) for i in range(5)]  # c0..c4 with scores 1.0..0.6
    distractors = [_p("d0"), _p("d1")]
    out = inject_distractors(candidates, distractors, mode="replace_bottom", seed=42)
    out_ids = [p.doc_id for p in out]
    # The two lowest-scored (c3, c4) should be dropped
    assert "c3" not in out_ids
    assert "c4" not in out_ids
    assert "d0" in out_ids and "d1" in out_ids
    assert len(out) == 5


def test_inject_replace_random_keeps_overall_size() -> None:
    candidates = [_p(f"c{i}") for i in range(10)]
    distractors = [_p(f"d{i}") for i in range(3)]
    out = inject_distractors(candidates, distractors, mode="replace_random", seed=42)
    out_ids = {p.doc_id for p in out}
    assert {"d0", "d1", "d2"}.issubset(out_ids)
    assert len(out) == 10  # replace_random keeps size


def test_inject_dedupes_by_doc_id() -> None:
    """A distractor whose doc_id overlaps a candidate should not be re-injected."""
    candidates = [_p("a"), _p("b")]
    distractors = [_p("a"), _p("c")]  # 'a' overlaps
    out = inject_distractors(candidates, distractors, mode="mix", seed=42)
    out_ids = [p.doc_id for p in out]
    assert out_ids.count("a") == 1
    assert "c" in out_ids


def test_inject_no_op_on_empty_distractors() -> None:
    candidates = [_p("a"), _p("b")]
    out = inject_distractors(candidates, [], mode="mix", seed=42)
    assert [p.doc_id for p in out] == ["a", "b"]


def test_inject_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown inject mode"):
        inject_distractors([_p("a")], [_p("d")], mode="invalid", seed=42)  # type: ignore[arg-type]


def test_inject_returns_new_list_does_not_mutate_input() -> None:
    candidates = [_p("a"), _p("b")]
    distractors = [_p("d")]
    original_ids = [p.doc_id for p in candidates]
    inject_distractors(candidates, distractors, mode="mix", seed=42)
    assert [p.doc_id for p in candidates] == original_ids
