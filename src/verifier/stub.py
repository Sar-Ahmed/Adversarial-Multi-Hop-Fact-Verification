"""Phase 02 verifier stub: always returns NEI with confidence 0.5.

Replaced in Phase 07 by the LLM + NLI veto ensemble (Qwen 3B Q4 LLM verdict
plus DeBERTa NLI cross-encoder veto rule).
"""

from __future__ import annotations

from src.schema import Label, Passage


class StubVerifier:
    """Returns (NEI, 0.5) for every claim. Kept honest about its uncertainty."""

    def verify(
        self,
        claim: str,  # noqa: ARG002
        passages: list[Passage],  # noqa: ARG002
    ) -> tuple[Label, float, str]:
        return Label.NEI, 0.5, "stub verifier — no real reasoning yet (Phase 07 replaces this)"
