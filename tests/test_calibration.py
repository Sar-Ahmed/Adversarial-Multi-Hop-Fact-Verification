"""Phase 08 calibrator tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.features import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_features,
)
from src.schema import Passage


def _toy_nli() -> dict:
    return {
        "max_contra": 0.91,
        "max_entail": 0.04,
        "max_neutral": 0.05,
        "per_passage": [
            {"doc_id": "a::0", "contra": 0.91, "entail": 0.04, "neutral": 0.05},
            {"doc_id": "b::0", "contra": 0.50, "entail": 0.20, "neutral": 0.30},
        ],
    }


def _toy_passages() -> list[Passage]:
    return [
        Passage(doc_id="a::0", title="A", sent_idx=0, text="The team plays in Paris.", score=0.80),
        Passage(
            doc_id="b::0",
            title="B",
            sent_idx=0,
            text="A completely different passage about cooking dinner.",
            score=0.60,
        ),
    ]


def test_feature_names_match_extracted_vector_length() -> None:
    assert len(FEATURE_NAMES) == N_FEATURES
    v = extract_features("claim about Paris", _toy_passages(), _toy_nli())
    assert v.shape == (N_FEATURES,)


def test_features_are_float32() -> None:
    v = extract_features("claim", _toy_passages(), _toy_nli())
    assert v.dtype == np.float32


def test_features_zero_on_empty_passages() -> None:
    v = extract_features("claim", [], {"max_contra": 0.5, "max_entail": 0.5, "per_passage": []})
    assert np.allclose(v, 0.0)


def test_features_capture_nli_signal() -> None:
    v = extract_features("Paris is in France", _toy_passages(), _toy_nli())
    nli_max_contra = v[FEATURE_NAMES.index("nli_max_contra")]
    nli_max_entail = v[FEATURE_NAMES.index("nli_max_entail")]
    assert nli_max_contra == pytest.approx(0.91, abs=1e-4)
    assert nli_max_entail == pytest.approx(0.04, abs=1e-4)


def test_features_capture_retrieval_score_gap() -> None:
    passages = [
        Passage(doc_id=f"x::{i}", title="X", sent_idx=i, text="t", score=1.0 - i * 0.1)
        for i in range(5)
    ]
    nli = {
        "max_contra": 0,
        "max_entail": 0,
        "max_neutral": 0,
        "per_passage": [
            {"doc_id": f"x::{i}", "contra": 0, "entail": 0, "neutral": 0} for i in range(5)
        ],
    }
    v = extract_features("any", passages, nli)
    gap_idx = FEATURE_NAMES.index("retrieval_score_gap_top1_top5")
    assert v[gap_idx] == pytest.approx(0.4, abs=1e-4)  # 1.0 - 0.6 (top-5)


def test_features_entity_overlap_non_zero_when_shared_tokens() -> None:
    passages = _toy_passages()
    # Claim shares 'Paris' with passage A and not with passage B
    v = extract_features("Paris is the capital of France", passages, _toy_nli())
    overlap_idx = FEATURE_NAMES.index("entity_overlap_top1_jaccard")
    assert v[overlap_idx] > 0.0


def test_features_entity_overlap_zero_when_no_content_tokens_match() -> None:
    passages = [
        Passage(doc_id="z::0", title="Z", sent_idx=0, text="orange grape banana", score=0.9)
    ]
    nli = {
        "max_contra": 0.0,
        "max_entail": 0.0,
        "max_neutral": 0.0,
        "per_passage": [{"doc_id": "z::0", "contra": 0.0, "entail": 0.0, "neutral": 0.0}],
    }
    v = extract_features("airplane motorcycle bicycle", passages, nli)
    overlap_idx = FEATURE_NAMES.index("entity_overlap_top1_jaccard")
    assert v[overlap_idx] == 0.0
