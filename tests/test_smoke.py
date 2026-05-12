"""Phase 02 integration smoke test — gates every PR from now on.

Five fixed claims, run end-to-end through the real Phase 02 pipeline (real
dense retriever from Phase 01, stub decomposer + reranker + verifier).
Asserts shape and types only — values come from stubs, so nothing is asserted
about correctness. Phase 03+ swap stubs for real components and the same
test continues to pass.

Marked `slow` because it loads bge-small + 260 MB FAISS index. Marked `smoke`
so `make smoke` picks it up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.schema import EvidenceChain, Label, SubClaim, SubClaimVerification

# Five intentionally-varied fixed claims.
SMOKE_CLAIMS: list[str] = [
    "Christopher Nolan directed Inception.",
    "The Eiffel Tower is located in Berlin.",
    "The director of Inception also directed The Dark Knight.",
    "Inception was released in 2010.",
    "Christopher Nolan owns a yacht in Monaco.",
]


@pytest.fixture(scope="module")
def pipeline():  # noqa: ANN201 — pytest fixture
    """Build the pipeline once per module to amortize bge-small + FAISS load."""
    from src.config import PipelineConfig
    from src.pipeline import build_pipeline
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(42)
    setup_logging(json_sink=False)

    cfg = PipelineConfig.load(Path("configs/default.yaml"))
    return build_pipeline(cfg)


@pytest.mark.smoke
@pytest.mark.slow
@pytest.mark.parametrize("claim", SMOKE_CLAIMS)
def test_pipeline_returns_well_formed_evidence_chain(pipeline, claim: str) -> None:
    chain = pipeline.verify(claim)

    # Shape
    assert isinstance(chain, EvidenceChain)
    assert chain.claim == claim
    assert len(chain.sub_claims) >= 1
    assert all(isinstance(sc, SubClaim) for sc in chain.sub_claims)
    assert len(chain.verifications) == len(chain.sub_claims)
    assert all(isinstance(v, SubClaimVerification) for v in chain.verifications)

    # Verdicts and confidences are valid types in valid ranges
    assert isinstance(chain.final_verdict, Label)
    assert 0.0 <= chain.final_confidence <= 1.0
    for v in chain.verifications:
        assert isinstance(v.verdict, Label)
        assert 0.0 <= v.confidence <= 1.0

    # Every cited passage id resolves
    for v in chain.verifications:
        for pid in v.cited_passage_ids:
            assert pid in chain.passages_by_id, f"orphan citation {pid}"


@pytest.mark.smoke
@pytest.mark.slow
def test_pipeline_real_retriever_returns_relevant_passages(pipeline) -> None:
    """The retriever in Phase 02 is real (Phase 01 corpus). Inception query
    must surface the Inception article in the top retrieved set — a regression
    check for Phase 01 corpus quality."""
    chain = pipeline.verify("Christopher Nolan directed Inception.")
    titles = {p.title for p in chain.passages_by_id.values()}
    assert any(
        "Inception" in t for t in titles
    ), f"Inception article missing from retrieved passages — got titles: {titles}"


@pytest.mark.smoke
def test_pipeline_render_text_for_smoke_claim_does_not_crash(pipeline) -> None:
    chain = pipeline.verify("Christopher Nolan directed Inception.")
    text = chain.render_text()
    assert "Christopher Nolan" in text
    assert "verdict" in text.lower()


@pytest.mark.smoke
@pytest.mark.slow
def test_pipeline_adversarial_mode_runs_end_to_end(pipeline) -> None:
    """Phase 06: pipeline must produce a valid EvidenceChain when adversarial
    distractors are injected. We use synthetic distractors here (not the
    mined HoVer set) so the test doesn't depend on a phase-specific artifact."""
    from src.schema import Passage

    synthetic_distractors = [
        Passage(
            doc_id=f"distractor::{i}",
            title=f"FakeDistractor_{i}",
            sent_idx=0,
            text=(
                f"This passage {i} mentions Christopher Nolan but asserts he "
                "did not direct Inception, which is false."
            ),
            score=0.9,
        )
        for i in range(5)
    ]
    chain = pipeline.verify(
        "Christopher Nolan directed Inception.",
        adversarial_distractors=synthetic_distractors,
    )
    assert chain.final_verdict is not None
    assert 0.0 <= chain.final_confidence <= 1.0
    # Distractor doc_ids should appear in passages_by_id ONLY if they survived
    # the reranker's top_k cut. We don't assert presence; we just assert the
    # pipeline ran without crashing and produced a well-formed chain.
