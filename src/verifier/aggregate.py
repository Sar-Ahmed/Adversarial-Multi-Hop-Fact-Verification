"""Pure-function aggregators that combine LLM verdict + NLI scores.

Two production modes (verifier-level):

- `llm_only`              — return the LLM verdict and confidence unchanged.
                            Baseline; mirrors V1 Phase 4. Useful for ablation.

- `llm_plus_nli_veto`     — production mode. If the LLM says SUPPORTED but
                            max NLI contradiction is high (≥ contra_veto_threshold),
                            flip to REFUTED. If the LLM says SUPPORTED but
                            max entailment is below entail_threshold, downgrade
                            confidence by 0.2 (signal that the evidence is
                            present but weak).

V1's finding was that LLM-only achieves ~0.54 accuracy on HoVer dev (the 3B
model has a ~74% SUPPORTED bias), and the NLI veto lifts it to ~0.575-0.60.
V3 ships `llm_plus_nli_veto` as the default and exposes both modes for the
Phase 07 ablation.
"""

from __future__ import annotations

from src.schema import Label


def aggregate(
    *,
    mode: str,
    llm_verdict: Label,
    llm_confidence: float,
    llm_reasoning: str,
    nli_scores: dict,
    contra_veto_threshold: float = 0.95,
    entail_threshold: float = 0.7,
) -> tuple[Label, float, str]:
    """Combine LLM verdict + NLI signals per `mode`. Returns (verdict, confidence, reasoning)."""

    if mode == "llm_only":
        return llm_verdict, llm_confidence, llm_reasoning

    if mode not in ("llm_plus_nli_veto", "llm_plus_nli_bidir"):
        raise ValueError(f"unknown verifier mode: {mode!r}")

    max_contra = float(nli_scores.get("max_contra", 0.0))
    max_entail = float(nli_scores.get("max_entail", 0.0))

    # === Veto 1: SUPPORTED → REFUTED (legacy rule, both modes share it) ===
    # LLM said SUPPORTED but NLI thinks the evidence contradicts → flip.
    if llm_verdict is Label.SUPPORTED and max_contra >= contra_veto_threshold:
        reasoning = (
            f"{llm_reasoning} [NLI veto: max_contra={max_contra:.2f} "
            f">= {contra_veto_threshold}]"
        )
        return Label.REFUTED, min(max_contra, 1.0), reasoning

    # === Veto 2 & 3: bidirectional rules (only in llm_plus_nli_bidir) ===
    # Phase 07 found that the 3B model defaults to NEI on multi-hop claims
    # ~95% of the time even when NLI sees strong contradiction or entailment.
    # The bidirectional mode flips NEI when NLI gives a confident signal.
    if mode == "llm_plus_nli_bidir":
        if llm_verdict is Label.NEI and max_contra >= contra_veto_threshold:
            reasoning = (
                f"{llm_reasoning} [NLI-bidir: NEI→REFUTED on max_contra=" f"{max_contra:.2f}]"
            )
            return Label.REFUTED, min(max_contra, 1.0), reasoning
        if llm_verdict is Label.NEI and max_entail >= contra_veto_threshold:
            reasoning = (
                f"{llm_reasoning} [NLI-bidir: NEI→SUPPORTED on max_entail=" f"{max_entail:.2f}]"
            )
            return Label.SUPPORTED, min(max_entail, 1.0), reasoning

    # Soft downgrade: LLM said SUPPORTED but entailment is weak.
    if llm_verdict is Label.SUPPORTED and max_entail < entail_threshold:
        return (
            llm_verdict,
            max(llm_confidence - 0.2, 0.0),
            f"{llm_reasoning} [NLI-weak: max_entail={max_entail:.2f} < {entail_threshold}]",
        )

    return llm_verdict, llm_confidence, llm_reasoning
