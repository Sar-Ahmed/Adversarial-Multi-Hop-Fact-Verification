"""Phase 07 verifier tests — parser, aggregator, NLI shape.

Fast tests (no model load):
  - JSON parsing of well-formed and malformed LLM outputs
  - Aggregate.aggregate() for both modes + boundary cases

Slow tests (load real models):
  - EnsembleVerifier on a fixed (claim, passages) input
"""

from __future__ import annotations

import pytest

from src.schema import Label, Passage
from src.verifier.aggregate import aggregate
from src.verifier.llm import _extract_json_object, _parse_verdict

# === JSON extraction (pure logic) ============================================


def test_extract_json_object_handles_clean_input() -> None:
    text = '{"verdict": "SUPPORTED", "reason": "ok"}'
    assert _extract_json_object(text) == text


def test_extract_json_object_handles_preamble() -> None:
    text = 'Answer: {"verdict": "REFUTED", "reason": "x"}'
    assert _extract_json_object(text) == '{"verdict": "REFUTED", "reason": "x"}'


def test_extract_json_object_handles_nested_braces() -> None:
    text = '{"verdict": "NEI", "reason": "uses { brace }"}'
    assert _extract_json_object(text) == text


def test_extract_json_object_returns_none_when_no_object() -> None:
    assert _extract_json_object("just some prose with no JSON") is None


# === Verdict parsing =========================================================


def test_parse_verdict_well_formed() -> None:
    raw = '{"verdict": "SUPPORTED", "reason": "Passage 1 says so."}'
    label, reason = _parse_verdict(raw)
    assert label is Label.SUPPORTED
    assert reason == "Passage 1 says so."


def test_parse_verdict_tolerates_lowercase() -> None:
    raw = '{"verdict": "supported", "reason": ""}'
    label, _ = _parse_verdict(raw)
    assert label is Label.SUPPORTED


def test_parse_verdict_tolerates_supports_variant() -> None:
    raw = '{"verdict": "SUPPORTS", "reason": ""}'
    label, _ = _parse_verdict(raw)
    assert label is Label.SUPPORTED


def test_parse_verdict_tolerates_refutes_variant() -> None:
    raw = '{"verdict": "REFUTES", "reason": ""}'
    label, _ = _parse_verdict(raw)
    assert label is Label.REFUTED


def test_parse_verdict_tolerates_not_enough_info() -> None:
    raw = '{"verdict": "NOT ENOUGH INFO", "reason": ""}'
    label, _ = _parse_verdict(raw)
    assert label is Label.NEI


def test_parse_verdict_returns_none_on_malformed_json() -> None:
    assert _parse_verdict("not json") is None
    assert _parse_verdict('{"verdict": missing_quote}') is None


def test_parse_verdict_returns_none_on_unknown_verdict() -> None:
    assert _parse_verdict('{"verdict": "maybe", "reason": ""}') is None


# === Aggregator ==============================================================


def _nli(contra: float, entail: float, neutral: float = 0.0) -> dict:
    return {"max_contra": contra, "max_entail": entail, "max_neutral": neutral}


def test_aggregate_llm_only_passes_through() -> None:
    v, c, r = aggregate(
        mode="llm_only",
        llm_verdict=Label.SUPPORTED,
        llm_confidence=0.6,
        llm_reasoning="x",
        nli_scores=_nli(0.99, 0.01),
    )
    assert v is Label.SUPPORTED
    assert c == 0.6
    assert r == "x"


def test_aggregate_nli_veto_flips_supported_to_refuted() -> None:
    v, c, r = aggregate(
        mode="llm_plus_nli_veto",
        llm_verdict=Label.SUPPORTED,
        llm_confidence=0.6,
        llm_reasoning="orig",
        nli_scores=_nli(0.97, 0.01),
        contra_veto_threshold=0.95,
        entail_threshold=0.7,
    )
    assert v is Label.REFUTED
    assert c >= 0.95
    assert "NLI veto" in r


def test_aggregate_nli_veto_below_threshold_keeps_supported() -> None:
    v, _, _ = aggregate(
        mode="llm_plus_nli_veto",
        llm_verdict=Label.SUPPORTED,
        llm_confidence=0.6,
        llm_reasoning="x",
        nli_scores=_nli(0.94, 0.85),  # contra just below 0.95
        contra_veto_threshold=0.95,
        entail_threshold=0.7,
    )
    assert v is Label.SUPPORTED


def test_aggregate_soft_downgrade_when_entail_weak() -> None:
    v, c, r = aggregate(
        mode="llm_plus_nli_veto",
        llm_verdict=Label.SUPPORTED,
        llm_confidence=0.6,
        llm_reasoning="x",
        nli_scores=_nli(0.1, 0.4),  # below entail threshold, not contra
        contra_veto_threshold=0.95,
        entail_threshold=0.7,
    )
    assert v is Label.SUPPORTED
    assert c == pytest.approx(0.4)  # 0.6 - 0.2
    assert "NLI-weak" in r


def test_aggregate_does_not_override_refuted_or_nei() -> None:
    """Current production policy: NLI doesn't touch REFUTED or NEI verdicts.
    Open follow-up if Phase 11 shows symmetric vetoes would help."""
    for verdict in (Label.REFUTED, Label.NEI):
        v, _, _ = aggregate(
            mode="llm_plus_nli_veto",
            llm_verdict=verdict,
            llm_confidence=0.6,
            llm_reasoning="x",
            nli_scores=_nli(0.99, 0.99),
            contra_veto_threshold=0.95,
            entail_threshold=0.7,
        )
        assert v is verdict


def test_aggregate_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown verifier mode"):
        aggregate(
            mode="bogus",
            llm_verdict=Label.SUPPORTED,
            llm_confidence=0.6,
            llm_reasoning="x",
            nli_scores=_nli(0.0, 0.0),
        )


# === NLI shape (no model load needed for empty case) =========================


def test_nli_score_on_empty_passages_returns_zeros() -> None:
    from src.verifier.nli import NLIVerifier

    nli = NLIVerifier(model_name="dummy")  # won't be loaded
    out = nli.score("claim", [])
    assert out["max_contra"] == 0.0
    assert out["max_entail"] == 0.0
    assert out["n_passages"] == 0
    assert out["per_passage"] == []


# === Slow integration tests ==================================================


@pytest.fixture(scope="module")
def cfg():  # noqa: ANN201
    from pathlib import Path

    from src.config import PipelineConfig

    cfg = PipelineConfig.load(Path("configs/default.yaml"))
    if not cfg.verifier.llm_path or not Path(cfg.verifier.llm_path).exists():
        pytest.skip(f"Qwen GGUF not at {cfg.verifier.llm_path}")
    return cfg


@pytest.mark.slow
def test_ensemble_verifier_on_simple_supported_claim(cfg) -> None:  # noqa: ANN001
    from src.verifier.ensemble import EnsembleVerifier

    ev = EnsembleVerifier(cfg.verifier)
    passages = [
        Passage(
            doc_id="Eiffel::0",
            title="Eiffel_Tower",
            sent_idx=0,
            text="The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
            score=0.9,
        )
    ]
    verdict, conf, reason = ev.verify("The Eiffel Tower is in Paris.", passages)
    assert verdict is Label.SUPPORTED, f"expected SUPPORTED, got {verdict} ({reason!r})"
    assert 0.0 <= conf <= 1.0


@pytest.mark.slow
def test_ensemble_verifier_on_clear_refuted_claim(cfg) -> None:  # noqa: ANN001
    from src.verifier.ensemble import EnsembleVerifier

    ev = EnsembleVerifier(cfg.verifier)
    passages = [
        Passage(
            doc_id="Eiffel::0",
            title="Eiffel_Tower",
            sent_idx=0,
            text="The Eiffel Tower is in Paris, France, not in Berlin.",
            score=0.9,
        )
    ]
    verdict, _, reason = ev.verify("The Eiffel Tower is in Berlin.", passages)
    assert verdict is Label.REFUTED, f"expected REFUTED, got {verdict} ({reason!r})"


@pytest.mark.slow
def test_ensemble_verifier_returns_nei_for_off_topic_passages(cfg) -> None:  # noqa: ANN001
    from src.verifier.ensemble import EnsembleVerifier

    ev = EnsembleVerifier(cfg.verifier)
    passages = [
        Passage(
            doc_id="rand::0",
            title="Photosynthesis",
            sent_idx=0,
            text="Photosynthesis is a process used by plants to convert light into chemical energy.",
            score=0.5,
        )
    ]
    verdict, _, _ = ev.verify("Christopher Nolan directed Inception.", passages)
    assert verdict is Label.NEI, f"expected NEI, got {verdict}"
