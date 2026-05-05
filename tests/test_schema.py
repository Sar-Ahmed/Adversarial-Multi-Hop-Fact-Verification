"""Phase 02 schema unit tests.

Pure-Python tests — no model loading, no network. Fast.
"""

from __future__ import annotations

import pytest

from src.schema import (
    EvidenceChain,
    Label,
    Passage,
    ReasoningType,
    SubClaim,
    SubClaimVerification,
)

# === SubClaim regression test for the V2 bug ==================================


def test_subclaim_uses_id_field_name_not_sub_claim_id() -> None:
    """V2 crashed everywhere because its decomposer wrote SubClaim(sub_claim_id=...)
    against this exact dataclass. The field is `id`. This test exists so we can
    never silently re-introduce that bug."""
    sc = SubClaim(id=0, text="Inception was directed by Christopher Nolan.")
    assert sc.id == 0
    assert sc.text.startswith("Inception")
    with pytest.raises(TypeError):
        SubClaim(sub_claim_id=0, text="x")  # type: ignore[call-arg]


# === SubClaim invariants ======================================================


def test_subclaim_depends_on_must_reference_earlier_claims() -> None:
    SubClaim(id=2, text="ok", depends_on=[0, 1])  # earlier ids ok
    with pytest.raises(ValueError, match="depends on"):
        SubClaim(id=0, text="ok", depends_on=[0])  # self-dep fails
    with pytest.raises(ValueError, match="depends on"):
        SubClaim(id=1, text="ok", depends_on=[2])  # forward dep fails


def test_subclaim_normalizes_depends_on_list_to_tuple() -> None:
    sc = SubClaim(id=3, text="ok", depends_on=[0, 1, 2])
    assert isinstance(sc.depends_on, tuple)
    assert sc.depends_on == (0, 1, 2)


def test_subclaim_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="empty text"):
        SubClaim(id=0, text="   ")


def test_subclaim_default_reasoning_type_is_other() -> None:
    sc = SubClaim(id=0, text="ok")
    assert sc.reasoning_type is ReasoningType.OTHER


# === Passage invariants =======================================================


def test_passage_rejects_negative_sent_idx() -> None:
    with pytest.raises(ValueError, match="sent_idx"):
        Passage(doc_id="d", title="t", sent_idx=-1, text="x")


def test_passage_requires_doc_id() -> None:
    with pytest.raises(ValueError, match="doc_id"):
        Passage(doc_id="", title="t", sent_idx=0, text="x")


# === SubClaimVerification invariants ==========================================


def test_verification_confidence_must_be_in_unit_interval() -> None:
    SubClaimVerification(sub_claim_id=0, verdict=Label.SUPPORTED, confidence=0.0)
    SubClaimVerification(sub_claim_id=0, verdict=Label.SUPPORTED, confidence=1.0)
    with pytest.raises(ValueError, match="confidence"):
        SubClaimVerification(sub_claim_id=0, verdict=Label.SUPPORTED, confidence=1.01)
    with pytest.raises(ValueError, match="confidence"):
        SubClaimVerification(sub_claim_id=0, verdict=Label.SUPPORTED, confidence=-0.01)


# === EvidenceChain invariants =================================================


def _toy_chain() -> EvidenceChain:
    sc = SubClaim(id=0, text="The Eiffel Tower is in Paris.")
    v = SubClaimVerification(sub_claim_id=0, verdict=Label.SUPPORTED, confidence=0.9)
    return EvidenceChain(
        claim="The Eiffel Tower is in Paris.",
        sub_claims=(sc,),
        verifications=(v,),
        final_verdict=Label.SUPPORTED,
        final_confidence=0.9,
    )


def test_evidence_chain_requires_verifications_to_reference_real_subclaims() -> None:
    sc = SubClaim(id=0, text="ok")
    bad_v = SubClaimVerification(sub_claim_id=99, verdict=Label.NEI, confidence=0.5)
    with pytest.raises(ValueError, match="unknown sub_claim_id"):
        EvidenceChain(
            claim="x",
            sub_claims=(sc,),
            verifications=(bad_v,),
            final_verdict=Label.NEI,
            final_confidence=0.5,
        )


def test_evidence_chain_render_text_includes_claim_and_verdict() -> None:
    chain = _toy_chain()
    out = chain.render_text()
    assert "The Eiffel Tower" in out
    assert "SUPPORTED" in out


def test_evidence_chain_normalizes_lists_to_tuples() -> None:
    sc = SubClaim(id=0, text="x")
    v = SubClaimVerification(sub_claim_id=0, verdict=Label.NEI, confidence=0.5)
    chain = EvidenceChain(
        claim="x",
        sub_claims=[sc],  # type: ignore[arg-type]
        verifications=[v],  # type: ignore[arg-type]
        final_verdict=Label.NEI,
        final_confidence=0.5,
    )
    assert isinstance(chain.sub_claims, tuple)
    assert isinstance(chain.verifications, tuple)


# === Enum identity ============================================================


def test_label_values_are_canonical_strings() -> None:
    assert Label.SUPPORTED.value == "SUPPORTED"
    assert Label.REFUTED.value == "REFUTED"
    assert Label.NEI.value == "NEI"


def test_reasoning_type_has_six_categories() -> None:
    assert {rt.value for rt in ReasoningType} == {
        "lookup",
        "comparison",
        "temporal",
        "composition",
        "negation",
        "other",
    }
