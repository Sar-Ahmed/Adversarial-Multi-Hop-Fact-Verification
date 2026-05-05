"""Phase 03 decomposer tests — parser logic + integration.

Most tests exercise the *parsing* path (no LLM call) so they're fast. Only
the integration test marked `slow` actually loads Qwen and runs decomposition.
"""

from __future__ import annotations

import pytest

from src.decomposer.decomposer import Decomposer, _ParseError
from src.schema import ReasoningType, SubClaim

# === JSON-array extraction (pure logic, no model) ============================


def test_extract_json_array_handles_clean_array() -> None:
    text = '[{"id": 0, "text": "x", "depends_on": [], "reasoning_type": "lookup"}]'
    assert Decomposer._extract_json_array(text) == text


def test_extract_json_array_handles_preamble() -> None:
    text = 'Here is the decomposition:\n[{"id": 0, "text": "x"}]'
    out = Decomposer._extract_json_array(text)
    assert out == '[{"id": 0, "text": "x"}]'


def test_extract_json_array_handles_nested_brackets() -> None:
    text = '[{"id": 0, "depends_on": [1, 2]}]'
    assert Decomposer._extract_json_array(text) == text


def test_extract_json_array_returns_none_when_no_array() -> None:
    assert Decomposer._extract_json_array("just some prose, no array") is None


def test_extract_json_array_ignores_brackets_inside_strings() -> None:
    text = '[{"text": "an [example] in a string"}]'
    out = Decomposer._extract_json_array(text)
    assert out == text


# === Subclaim construction from parsed dicts =================================


def test_build_subclaims_from_well_formed_input() -> None:
    parsed = [
        {"id": 0, "text": "claim a", "depends_on": [], "reasoning_type": "lookup"},
        {"id": 1, "text": "claim b", "depends_on": [0], "reasoning_type": "composition"},
    ]
    out = Decomposer._build_subclaims(parsed)
    assert [s.id for s in out] == [0, 1]
    assert out[1].depends_on == (0,)
    assert out[1].reasoning_type is ReasoningType.COMPOSITION


def test_build_subclaims_drops_forward_depends_on() -> None:
    """If model emits depends_on pointing to a later id, drop the dep silently."""
    parsed = [
        {"id": 0, "text": "a", "depends_on": [1], "reasoning_type": "lookup"},
    ]
    out = Decomposer._build_subclaims(parsed)
    assert out[0].depends_on == ()


def test_build_subclaims_renumbers_duplicate_ids() -> None:
    parsed = [
        {"id": 0, "text": "a", "depends_on": []},
        {"id": 0, "text": "b", "depends_on": []},
    ]
    out = Decomposer._build_subclaims(parsed)
    assert {s.id for s in out} == {0, 1}


def test_build_subclaims_falls_back_to_OTHER_for_unknown_reasoning_type() -> None:
    parsed = [{"id": 0, "text": "a", "depends_on": [], "reasoning_type": "made_up"}]
    out = Decomposer._build_subclaims(parsed)
    assert out[0].reasoning_type is ReasoningType.OTHER


def test_build_subclaims_skips_empty_text_entries() -> None:
    parsed = [
        {"id": 0, "text": "real", "depends_on": []},
        {"id": 1, "text": "  ", "depends_on": []},
    ]
    out = Decomposer._build_subclaims(parsed)
    assert len(out) == 1
    assert out[0].text == "real"


def test_build_subclaims_raises_on_empty_after_filtering() -> None:
    with pytest.raises(_ParseError, match="empty after parsing"):
        Decomposer._build_subclaims([{"id": 0, "text": "  "}])


def test_build_subclaims_raises_on_non_dict_entry() -> None:
    with pytest.raises(_ParseError, match="not a dict"):
        Decomposer._build_subclaims(["not a dict"])  # type: ignore[list-item]


def test_built_subclaims_are_real_subclaim_instances() -> None:
    parsed = [{"id": 0, "text": "a", "depends_on": []}]
    out = Decomposer._build_subclaims(parsed)
    assert isinstance(out[0], SubClaim)


# === Integration with real Qwen model ========================================


@pytest.fixture(scope="module")
def real_decomposer():  # noqa: ANN201
    """Real Qwen-backed decomposer. Skips the test if the GGUF isn't downloaded."""
    from pathlib import Path

    from src.config import PipelineConfig

    cfg = PipelineConfig.load("configs/default.yaml")
    if not cfg.decomposer.llm_path or not Path(cfg.decomposer.llm_path).exists():
        pytest.skip(f"Qwen GGUF not at {cfg.decomposer.llm_path} — run scripts/download_qwen.py")

    return Decomposer(
        llm_path=cfg.decomposer.llm_path,
        n_ctx=cfg.decomposer.n_ctx,
        max_tokens=cfg.decomposer.max_tokens,
        temperature=cfg.decomposer.temperature,
    )


@pytest.mark.slow
def test_decomposer_on_simple_lookup_claim(real_decomposer) -> None:
    out = real_decomposer.decompose("The Eiffel Tower is located in Paris.")
    assert 1 <= len(out) <= 3
    assert all(isinstance(s, SubClaim) for s in out)


@pytest.mark.slow
def test_decomposer_emits_multiple_subclaims_for_compound(real_decomposer) -> None:
    out = real_decomposer.decompose(
        "Christopher Nolan directed Inception and won an Oscar for Best Director."
    )
    assert len(out) >= 2, f"expected ≥2 sub-claims for compound claim, got {len(out)}"


@pytest.mark.slow
def test_decomposer_handles_dependent_pattern(real_decomposer) -> None:
    out = real_decomposer.decompose("The director of Inception also directed The Dark Knight.")
    assert len(out) >= 2
    # At least one sub-claim should depend on an earlier one
    assert any(
        s.depends_on for s in out
    ), f"expected dependent sub-claim for composition pattern; got {[s.depends_on for s in out]}"
