"""Unified pipeline schema (Phase 02).

These dataclasses are the contract between every component. Phase 01's
`src/data/types.py` had dataset-loader-specific types; this module is the
canonical project schema used by decomposer / retriever / reranker / verifier
/ evidence-chain builder.

Design rules (per docs/DECISIONS.md):

- All structures are immutable after construction (`frozen=True` + `tuple`
  for sequence fields). Consumers receive read-only views.
- `__post_init__` validates invariants up-front so downstream code can trust
  what it gets.
- Field name `id` (NOT `sub_claim_id`) — explicit fix for the V2 bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Label(str, Enum):
    """3-class verdict taxonomy."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NEI = "NEI"


class ReasoningType(str, Enum):
    """Categorical hint about how a sub-claim must be verified.

    Set by the decomposer (Phase 03) and read by aggregator / temporal module
    (Phase 09). `OTHER` is the safe default when the decomposer is unsure.
    """

    LOOKUP = "lookup"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    COMPOSITION = "composition"
    NEGATION = "negation"
    OTHER = "other"


@dataclass(frozen=True)
class Passage:
    """A retrieved Wikipedia passage."""

    doc_id: str
    title: str
    sent_idx: int
    text: str
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.sent_idx < 0:
            raise ValueError(f"Passage.sent_idx must be >= 0, got {self.sent_idx}")
        if not self.doc_id:
            raise ValueError("Passage.doc_id must be non-empty")


@dataclass(frozen=True)
class SubClaim:
    """An atomic sub-claim emitted by the decomposer.

    Field name is `id` (not `sub_claim_id`). This is the single most explicit
    regression check in the project — V2 crashed on every claim because the
    decomposer used `sub_claim_id=...` against this dataclass.
    """

    id: int
    text: str
    depends_on: tuple[int, ...] = ()
    reasoning_type: ReasoningType = ReasoningType.OTHER

    def __post_init__(self) -> None:
        # Accept list inputs (decomposer JSON yields lists) but store as tuple.
        if isinstance(self.depends_on, list):
            object.__setattr__(self, "depends_on", tuple(self.depends_on))
        for dep in self.depends_on:
            if dep >= self.id:
                raise ValueError(
                    f"SubClaim {self.id} depends on {dep}, which is not an earlier sub-claim"
                )
        if not self.text.strip():
            raise ValueError(f"SubClaim {self.id} has empty text")


@dataclass(frozen=True)
class SubClaimVerification:
    """The verifier's verdict for a single sub-claim.

    `cited_passage_ids` reference passages stored by id in the parent
    EvidenceChain so the chain stays compact (no inline passage text).
    """

    sub_claim_id: int
    verdict: Label
    confidence: float
    cited_passage_ids: tuple[str, ...] = ()
    reasoning: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.cited_passage_ids, list):
            object.__setattr__(self, "cited_passage_ids", tuple(self.cited_passage_ids))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"SubClaimVerification.confidence must be in [0, 1], got {self.confidence}"
            )


@dataclass(frozen=True)
class EvidenceChain:
    """The end-to-end output of `Pipeline.verify(claim)`.

    Holds the original claim, the decomposition, per-sub-claim verifications,
    and the final aggregated verdict. `passages_by_id` is the lookup table for
    rendering — keeps citations as IDs, not duplicated text blocks.
    """

    claim: str
    sub_claims: tuple[SubClaim, ...]
    verifications: tuple[SubClaimVerification, ...]
    final_verdict: Label
    final_confidence: float
    passages_by_id: dict[str, Passage] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize sequence types.
        if isinstance(self.sub_claims, list):
            object.__setattr__(self, "sub_claims", tuple(self.sub_claims))
        if isinstance(self.verifications, list):
            object.__setattr__(self, "verifications", tuple(self.verifications))

        if not 0.0 <= self.final_confidence <= 1.0:
            raise ValueError(
                f"EvidenceChain.final_confidence must be in [0, 1], got {self.final_confidence}"
            )

        # Every verification must reference a real sub-claim.
        sc_ids = {sc.id for sc in self.sub_claims}
        for v in self.verifications:
            if v.sub_claim_id not in sc_ids:
                raise ValueError(
                    f"Verification references unknown sub_claim_id={v.sub_claim_id}; "
                    f"sub-claims are {sorted(sc_ids)}"
                )

    def render_text(self, max_passage_chars: int = 200) -> str:
        """Human-readable rendering for CLI output. Phase 10 extends this."""
        lines = [
            f"Claim: {self.claim}",
            f"Final verdict: {self.final_verdict.value} (confidence {self.final_confidence:.2f})",
            f"Sub-claims: {len(self.sub_claims)}",
            "",
        ]
        verifications_by_id = {v.sub_claim_id: v for v in self.verifications}
        for sc in self.sub_claims:
            lines.append(f"Sub-claim {sc.id} [{sc.reasoning_type.value}]: {sc.text}")
            if sc.depends_on:
                lines.append(f"  depends_on: {list(sc.depends_on)}")
            v = verifications_by_id.get(sc.id)
            if v is None:
                lines.append("  (no verification)")
                continue
            lines.append(f"  Verdict: {v.verdict.value} (confidence {v.confidence:.2f})")
            if v.reasoning:
                lines.append(f"  Reasoning: {v.reasoning}")
            for pid in v.cited_passage_ids:
                p = self.passages_by_id.get(pid)
                if p is None:
                    lines.append(f"  [{pid}] (passage not in lookup)")
                    continue
                excerpt = p.text[:max_passage_chars]
                if len(p.text) > max_passage_chars:
                    excerpt = excerpt.rstrip() + "..."
                lines.append(f"  [{pid}] {p.title}: {excerpt}")
            lines.append("")
        return "\n".join(lines).rstrip()
