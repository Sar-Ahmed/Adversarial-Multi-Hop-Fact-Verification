"""Phase 02 decomposer stub: returns the original claim as a single sub-claim.

Replaced in Phase 03 by the Qwen2.5-3B-Q4 few-shot decomposer with
dependency-DAG output and JSON parsing + retry + fallback.
"""

from __future__ import annotations

from src.schema import ReasoningType, SubClaim


class StubDecomposer:
    """Single-sub-claim wrapper. The interface every Phase 03+ decomposer must satisfy."""

    def decompose(self, claim: str) -> list[SubClaim]:
        return [SubClaim(id=0, text=claim, depends_on=(), reasoning_type=ReasoningType.OTHER)]
