"""Ensemble verifier: LLM + NLI + aggregator with one verify() entry point.

This is the production component that swaps in for `StubVerifier`. It owns
exactly two model handles (Qwen 3B Q4 GGUF and DeBERTa-v3 NLI cross-encoder),
keeps them loaded across calls, and exposes the same `verify(claim, passages)`
signature that `Pipeline` expects.

Operator can pick the aggregation mode via `cfg.verifier.mode`:

  - `llm_only`              — diagnostic / ablation
  - `llm_plus_nli_veto`     — production default
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.schema import Label, Passage
from src.verifier.aggregate import aggregate
from src.verifier.llm import LLMVerifier
from src.verifier.nli import NLIVerifier

if TYPE_CHECKING:
    from src.config import VerifierConfig


class EnsembleVerifier:
    """LLM verdict + NLI-derived veto / soft-downgrade rule, optionally
    re-aggregated through the Phase 08 NEI calibrator.

    If `calibrator` is supplied, it overrides the rule-based final verdict
    whenever its max-class probability exceeds the calibrator's own
    `decision_threshold` (see `NEICalibrator.predict`). Otherwise the rule
    output is returned unchanged.
    """

    def __init__(self, cfg: VerifierConfig, calibrator=None) -> None:  # noqa: ANN001
        if not cfg.llm_path:
            raise ValueError(
                "EnsembleVerifier needs cfg.verifier.llm_path set. "
                "If you want a stub verifier, use StubVerifier instead."
            )
        self.cfg = cfg
        self.llm = LLMVerifier(llm_path=cfg.llm_path)
        self.nli = NLIVerifier(model_name=cfg.nli_model)
        self.calibrator = calibrator  # Phase 08 — optional NEICalibrator

    def verify(
        self,
        claim: str,
        passages: list[Passage],
    ) -> tuple[Label, float, str]:
        """Run LLM + NLI in parallel-by-thought, then aggregate per mode."""
        llm_verdict, llm_conf, llm_reason = self.llm.verify(claim, passages)
        if self.cfg.mode == "llm_only":
            return aggregate(
                mode="llm_only",
                llm_verdict=llm_verdict,
                llm_confidence=llm_conf,
                llm_reasoning=llm_reason,
                nli_scores={},  # unused in llm_only
            )

        nli_scores = self.nli.score(claim, passages)
        verdict, confidence, reasoning = aggregate(
            mode=self.cfg.mode,
            llm_verdict=llm_verdict,
            llm_confidence=llm_conf,
            llm_reasoning=llm_reason,
            nli_scores=nli_scores,
            contra_veto_threshold=self.cfg.contra_veto_threshold,
            entail_threshold=self.cfg.entail_threshold,
        )
        if verdict is not llm_verdict:
            logger.info(
                "NLI veto: LLM={} → {} (max_contra={:.3f}, max_entail={:.3f})",
                llm_verdict.value,
                verdict.value,
                nli_scores["max_contra"],
                nli_scores["max_entail"],
            )

        # Optional Phase 08 calibrator: re-aggregate (verdict, NLI, retrieval,
        # lexical) features through the trained classifier and use its verdict
        # if it's more confident than the rule.
        if self.calibrator is not None:
            from src.calibration.features import extract_features

            features = extract_features(claim, passages, nli_scores)
            cal_verdict, cal_conf, _cal_probs = self.calibrator.predict(features)
            if cal_verdict is not verdict:
                logger.info(
                    "calibrator override: {} → {} (cal_conf={:.3f})",
                    verdict.value,
                    cal_verdict.value,
                    cal_conf,
                )
            return cal_verdict, cal_conf, f"{reasoning} [calibrator: p={cal_conf:.2f}]"

        return verdict, confidence, reasoning

    def verify_with_trace(
        self,
        claim: str,
        passages: list[Passage],
    ) -> dict:
        """Like `verify`, but returns a full trace dict for the eval harness."""
        llm_verdict, llm_conf, llm_reason = self.llm.verify(claim, passages)
        nli_scores = (
            self.nli.score(claim, passages)
            if self.cfg.mode != "llm_only"
            else {"max_contra": 0.0, "max_entail": 0.0, "max_neutral": 0.0, "per_passage": []}
        )
        verdict, confidence, reasoning = aggregate(
            mode=self.cfg.mode,
            llm_verdict=llm_verdict,
            llm_confidence=llm_conf,
            llm_reasoning=llm_reason,
            nli_scores=nli_scores,
            contra_veto_threshold=self.cfg.contra_veto_threshold,
            entail_threshold=self.cfg.entail_threshold,
        )
        return {
            "claim": claim,
            "passage_doc_ids": [p.doc_id for p in passages],
            "llm": {
                "verdict": llm_verdict.value,
                "confidence": llm_conf,
                "reasoning": llm_reason,
            },
            "nli": {
                "max_contra": nli_scores.get("max_contra", 0.0),
                "max_entail": nli_scores.get("max_entail", 0.0),
                "max_neutral": nli_scores.get("max_neutral", 0.0),
                "per_passage": nli_scores.get("per_passage", []),
            },
            "mode": self.cfg.mode,
            "final": {
                "verdict": verdict.value,
                "confidence": confidence,
                "reasoning": reasoning,
            },
        }
