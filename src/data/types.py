"""Phase 01 data-loading types.

These are dataset-specific dataclasses. The unified pipeline schema
(`src/schema.py`, `Label` enum, `SubClaim`, etc.) lands in Phase 02.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HoverExample:
    """A HoVer claim with its gold supporting facts.

    HoVer has only two labels: SUPPORTED and NOT_SUPPORTED. We normalize the
    NOT_SUPPORTED label to "REFUTED" to align with FEVER and the project's
    Label enum (Phase 02).
    """

    uid: str
    claim: str
    label: str  # "SUPPORTED" or "REFUTED"
    num_hops: int
    # FEVER-encoded titles touched by the gold supporting facts (deduped).
    supporting_titles: tuple[str, ...]
    # (title, sent_idx) pairs verbatim from the dataset.
    supporting_facts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class FeverExample:
    """A FEVER claim with its gold evidence.

    FEVER has three labels: SUPPORTS, REFUTES, NOT ENOUGH INFO. We map them to
    SUPPORTED / REFUTED / NEI to align with the project's Label enum.
    """

    id: int
    claim: str
    label: str  # "SUPPORTED" | "REFUTED" | "NEI"
    # Unique gold titles across all evidence sets (FEVER-encoded).
    evidence_titles: tuple[str, ...]
    # All (title, sent_idx) pairs across all evidence sets, deduped.
    evidence_facts: tuple[tuple[str, int], ...]
