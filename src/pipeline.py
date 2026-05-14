"""End-to-end pipeline orchestration.

`Pipeline.verify(claim)` runs:
    decomposer  →  for each sub-claim:
                       retriever (top-K) → reranker (top-N) → verifier
                   aggregate verifications → final verdict
                → return EvidenceChain

Aggregation rule (Phase 02 baseline; Phase 07 owns the real version):
- any sub-claim REFUTED  → final REFUTED
- all sub-claims SUPPORTED → final SUPPORTED
- otherwise              → final NEI
- final_confidence = mean of contributing sub-claim confidences

Components are passed in by the caller (`build_pipeline` is the standard
factory). This keeps `Pipeline` itself zero-import — easy to test with mocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from loguru import logger

from src.schema import EvidenceChain, Label, Passage, SubClaim, SubClaimVerification

if TYPE_CHECKING:
    from src.config import PipelineConfig


class _Decomposer(Protocol):
    def decompose(self, claim: str) -> list[SubClaim]: ...


class _Retriever(Protocol):
    def retrieve(self, text: str, top_k: int | None = None) -> list[Passage]: ...


class _Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[Passage], top_k: int | None = None
    ) -> list[Passage]: ...


class _Verifier(Protocol):
    def verify(self, claim: str, passages: list[Passage]) -> tuple[Label, float, str]: ...


class Pipeline:
    """Linear orchestration. No silent fallbacks."""

    def __init__(
        self,
        cfg: PipelineConfig,
        decomposer: _Decomposer,
        retriever: _Retriever,
        reranker: _Reranker,
        verifier: _Verifier,
    ) -> None:
        self.cfg = cfg
        self.decomposer = decomposer
        self.retriever = retriever
        self.reranker = reranker
        self.verifier = verifier

    def verify(
        self,
        claim: str,
        *,
        adversarial_distractors: list[Passage] | None = None,
    ) -> EvidenceChain:
        """Verify a single claim and return its EvidenceChain.

        `adversarial_distractors` — optional pre-mined distractors (Phase 06)
        injected into the dense retrieval pool before reranking. The eval
        harness toggles this on/off to measure adversarial robustness.
        """
        if not claim or not claim.strip():
            raise ValueError("Pipeline.verify called with empty claim")

        logger.info(
            "verify: {!r} (adversarial={})",
            claim[:120],
            bool(adversarial_distractors),
        )

        sub_claims = self.decomposer.decompose(claim)
        logger.info("  decomposed into {} sub-claim(s)", len(sub_claims))

        verifications: list[SubClaimVerification] = []
        passages_by_id: dict[str, Passage] = {}

        for sc in sub_claims:
            candidates = self.retriever.retrieve(sc.text, top_k=self.cfg.retriever.top_k)
            if adversarial_distractors:
                from src.adversarial.inject import inject_distractors

                candidates = inject_distractors(
                    candidates,
                    adversarial_distractors,
                    mode=self.cfg.adversarial.inject_mode,
                    seed=self.cfg.eval.seed,
                )
            top = self.reranker.rerank(sc.text, candidates, top_k=self.cfg.reranker.top_k)
            for p in top:
                passages_by_id[p.doc_id] = p

            verdict, confidence, reasoning = self.verifier.verify(sc.text, top)
            cited = tuple(p.doc_id for p in top[:3])
            verifications.append(
                SubClaimVerification(
                    sub_claim_id=sc.id,
                    verdict=verdict,
                    confidence=confidence,
                    cited_passage_ids=cited,
                    reasoning=reasoning,
                )
            )
            logger.info(
                "  sub-claim {} → {} (conf={:.2f}, {} citations)",
                sc.id,
                verdict.value,
                confidence,
                len(cited),
            )

        final_verdict, final_confidence = self._aggregate(verifications)
        logger.info(
            "  final: {} (conf={:.2f})",
            final_verdict.value,
            final_confidence,
        )

        return EvidenceChain(
            claim=claim,
            sub_claims=tuple(sub_claims),
            verifications=tuple(verifications),
            final_verdict=final_verdict,
            final_confidence=final_confidence,
            passages_by_id=passages_by_id,
        )

    @staticmethod
    def _aggregate(verifications: list[SubClaimVerification]) -> tuple[Label, float]:
        """Phase 02 placeholder aggregator. Phase 07 owns the real version."""
        if not verifications:
            return Label.NEI, 0.0

        verdicts = [v.verdict for v in verifications]
        avg_conf = sum(v.confidence for v in verifications) / len(verifications)

        if Label.REFUTED in verdicts:
            return Label.REFUTED, avg_conf
        if all(v == Label.SUPPORTED for v in verdicts):
            return Label.SUPPORTED, avg_conf
        return Label.NEI, avg_conf


def build_pipeline(cfg: PipelineConfig) -> Pipeline:
    """Default factory.

    Decomposer: real Qwen-backed Decomposer if `cfg.decomposer.llm_path` is set,
    else `StubDecomposer`. The choice is config-driven, not silent — null in
    YAML means "use the stub".

    Retriever: always real (Phase 01 corpus).
    Reranker / Verifier: stubs until Phase 04 / Phase 07 land.
    """
    from src.reranker.cross_encoder import CrossEncoderReranker
    from src.retrieval.dense import DenseRetriever

    decomposer: _Decomposer
    if cfg.decomposer.llm_path:
        from src.decomposer.decomposer import Decomposer

        decomposer = Decomposer(
            llm_path=cfg.decomposer.llm_path,
            n_ctx=cfg.decomposer.n_ctx,
            max_tokens=cfg.decomposer.max_tokens,
            temperature=cfg.decomposer.temperature,
            seed=cfg.eval.seed,
        )
    else:
        from src.decomposer.stub import StubDecomposer

        decomposer = StubDecomposer()

    verifier: _Verifier
    if cfg.verifier.llm_path:
        from src.verifier.ensemble import EnsembleVerifier

        verifier = EnsembleVerifier(cfg.verifier)
    else:
        from src.verifier.stub import StubVerifier

        verifier = StubVerifier()

    return Pipeline(
        cfg=cfg,
        decomposer=decomposer,
        retriever=DenseRetriever(cfg.retriever, cfg.corpus),
        reranker=CrossEncoderReranker(cfg.reranker),
        verifier=verifier,
    )
