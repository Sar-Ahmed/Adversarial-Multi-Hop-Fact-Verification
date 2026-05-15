"""Phase 10 — evidence chain validator + JSON round-trip tests.

All fast (no model load, no network). Construct toy chains, validate,
serialize, deserialize, check equality.
"""

from __future__ import annotations

from src.evidence.chain import (
    dependency_path,
    from_jsonable,
    to_jsonable,
    validate,
)
from src.schema import (
    EvidenceChain,
    Label,
    Passage,
    SubClaim,
    SubClaimVerification,
)


def _toy_passage(doc_id: str = "Inception::0") -> Passage:
    return Passage(
        doc_id=doc_id,
        title="Inception",
        sent_idx=0,
        text="Inception is a 2010 film directed by Christopher Nolan.",
        score=0.85,
    )


def _toy_chain() -> EvidenceChain:
    sc = SubClaim(id=0, text="Inception was directed by Christopher Nolan.")
    p = _toy_passage()
    v = SubClaimVerification(
        sub_claim_id=0,
        verdict=Label.SUPPORTED,
        confidence=0.92,
        cited_passage_ids=(p.doc_id,),
        reasoning="Passage 1 says Nolan directed it.",
    )
    return EvidenceChain(
        claim="Inception was directed by Christopher Nolan.",
        sub_claims=(sc,),
        verifications=(v,),
        final_verdict=Label.SUPPORTED,
        final_confidence=0.92,
        passages_by_id={p.doc_id: p},
    )


# === Validator ===============================================================


def test_validate_passes_on_well_formed_chain() -> None:
    result = validate(_toy_chain())
    assert result.is_valid, result.errors


def test_validate_catches_unknown_citation() -> None:
    sc = SubClaim(id=0, text="x")
    v = SubClaimVerification(
        sub_claim_id=0,
        verdict=Label.SUPPORTED,
        confidence=0.5,
        cited_passage_ids=("nonexistent::0",),
        reasoning="r",
    )
    chain = EvidenceChain(
        claim="x",
        sub_claims=(sc,),
        verifications=(v,),
        final_verdict=Label.SUPPORTED,
        final_confidence=0.5,
        passages_by_id={},
    )
    result = validate(chain)
    assert not result.is_valid
    assert any("cites unknown passage_id" in e for e in result.errors)


def test_validate_catches_missing_citation_on_non_nei_verdict() -> None:
    sc = SubClaim(id=0, text="x")
    v = SubClaimVerification(
        sub_claim_id=0,
        verdict=Label.SUPPORTED,
        confidence=0.5,
        cited_passage_ids=(),
        reasoning="r",
    )
    chain = EvidenceChain(
        claim="x",
        sub_claims=(sc,),
        verifications=(v,),
        final_verdict=Label.SUPPORTED,
        final_confidence=0.5,
    )
    result = validate(chain)
    assert not result.is_valid
    assert any("no citations" in e for e in result.errors)


def test_validate_allows_nei_with_reasoning_and_no_citations() -> None:
    sc = SubClaim(id=0, text="x")
    v = SubClaimVerification(
        sub_claim_id=0,
        verdict=Label.NEI,
        confidence=0.4,
        cited_passage_ids=(),
        reasoning="no evidence found",
    )
    chain = EvidenceChain(
        claim="x",
        sub_claims=(sc,),
        verifications=(v,),
        final_verdict=Label.NEI,
        final_confidence=0.4,
    )
    result = validate(chain)
    assert result.is_valid, result.errors


# === JSON round-trip =========================================================


def test_to_and_from_jsonable_roundtrip() -> None:
    chain = _toy_chain()
    blob = to_jsonable(chain)
    rebuilt = from_jsonable(blob)
    assert rebuilt.claim == chain.claim
    assert rebuilt.final_verdict is chain.final_verdict
    assert rebuilt.final_confidence == chain.final_confidence
    assert len(rebuilt.sub_claims) == len(chain.sub_claims)
    assert rebuilt.sub_claims[0].text == chain.sub_claims[0].text
    assert rebuilt.passages_by_id == chain.passages_by_id


def test_jsonable_dict_is_plain_python_types() -> None:
    blob = to_jsonable(_toy_chain())
    # final_verdict should be a string, not an enum
    assert isinstance(blob["final_verdict"], str)
    assert isinstance(blob["sub_claims"][0]["depends_on"], list)


# === Dependency path =========================================================


def test_dependency_path_on_linear_chain() -> None:
    chain = EvidenceChain(
        claim="x",
        sub_claims=(
            SubClaim(id=0, text="a"),
            SubClaim(id=1, text="b", depends_on=(0,)),
            SubClaim(id=2, text="c", depends_on=(1,)),
        ),
        verifications=(
            SubClaimVerification(sub_claim_id=0, verdict=Label.NEI, confidence=0.5, reasoning="r"),
            SubClaimVerification(sub_claim_id=1, verdict=Label.NEI, confidence=0.5, reasoning="r"),
            SubClaimVerification(sub_claim_id=2, verdict=Label.NEI, confidence=0.5, reasoning="r"),
        ),
        final_verdict=Label.NEI,
        final_confidence=0.5,
    )
    assert dependency_path(chain) == [0, 1, 2]


def test_dependency_path_with_branching() -> None:
    chain = EvidenceChain(
        claim="x",
        sub_claims=(
            SubClaim(id=0, text="a"),
            SubClaim(id=1, text="b"),
            SubClaim(id=2, text="c", depends_on=(0, 1)),
        ),
        verifications=(
            SubClaimVerification(sub_claim_id=0, verdict=Label.NEI, confidence=0.5, reasoning="r"),
            SubClaimVerification(sub_claim_id=1, verdict=Label.NEI, confidence=0.5, reasoning="r"),
            SubClaimVerification(sub_claim_id=2, verdict=Label.NEI, confidence=0.5, reasoning="r"),
        ),
        final_verdict=Label.NEI,
        final_confidence=0.5,
    )
    order = dependency_path(chain)
    # Both 0 and 1 must come before 2; their relative order is up to Khan's
    # algorithm but the constraint is what matters.
    assert order.index(2) > order.index(0)
    assert order.index(2) > order.index(1)
