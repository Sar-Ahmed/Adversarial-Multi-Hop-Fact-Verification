"""Phase 10 — EvidenceChain serializer + structural validator.

The `EvidenceChain` dataclass already exists in `src/schema.py` with a
`render_text()` method. This module adds two things the spec calls out:

  1. `validate(chain)` — structural checks:
       - every cited `passage_id` resolves to `chain.passages_by_id`
       - every `depends_on` is an acyclic DAG pointing at *earlier* sub-claims
       - every sub-claim has either citations OR a NEI verdict with reasoning
  2. `to_jsonable(chain)` / `from_jsonable(data)` — JSON-safe round-trip

Why not extend the schema directly? Keeping the schema dataclass small (no
JSON dependency, no validator method) lets future consumers import it
cheaply. Phase 10's responsibilities are the *output* contract.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from src.schema import EvidenceChain, Label, Passage, ReasoningType, SubClaim, SubClaimVerification


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of `validate(chain)`. `errors` is empty iff `is_valid` is True."""

    is_valid: bool
    errors: tuple[str, ...]


def validate(chain: EvidenceChain) -> ValidationResult:
    """Run all structural checks. Returns (is_valid, errors)."""
    errors: list[str] = []
    seen_sc_ids: set[int] = set()
    sc_by_id: dict[int, SubClaim] = {}
    for sc in chain.sub_claims:
        if sc.id in seen_sc_ids:
            errors.append(f"duplicate sub_claim_id={sc.id}")
        seen_sc_ids.add(sc.id)
        sc_by_id[sc.id] = sc

    # 1. depends_on must reference earlier ids that exist
    for sc in chain.sub_claims:
        for dep in sc.depends_on:
            if dep not in seen_sc_ids:
                errors.append(f"sub-claim {sc.id} depends_on {dep} (not in chain)")
            if dep >= sc.id:
                errors.append(
                    f"sub-claim {sc.id} depends_on {dep} (not earlier — violates topological order)"
                )

    # 2. Cycle check via BFS over the depends_on graph (should be impossible
    #    given the forward-only constraint above, but defend-in-depth).
    in_progress: set[int] = set()
    done: set[int] = set()

    def dfs(node: int) -> bool:
        if node in in_progress:
            return True  # cycle
        if node in done:
            return False
        in_progress.add(node)
        sc = sc_by_id.get(node)
        if sc is None:
            return False
        for dep in sc.depends_on:
            if dfs(dep):
                return True
        in_progress.discard(node)
        done.add(node)
        return False

    for sc in chain.sub_claims:
        if dfs(sc.id):
            errors.append(f"cycle detected involving sub_claim_id={sc.id}")

    # 3. Verifications must reference real sub-claims
    sc_ids = {sc.id for sc in chain.sub_claims}
    for v in chain.verifications:
        if v.sub_claim_id not in sc_ids:
            errors.append(f"verification references unknown sub_claim_id={v.sub_claim_id}")

    # 4. Every cited passage_id must resolve to chain.passages_by_id
    for v in chain.verifications:
        for pid in v.cited_passage_ids:
            if pid not in chain.passages_by_id:
                errors.append(f"sub-claim {v.sub_claim_id} cites unknown passage_id={pid}")

    # 5. Either citations OR explicit NEI with non-empty reasoning
    for v in chain.verifications:
        if not v.cited_passage_ids:
            if v.verdict is not Label.NEI:
                errors.append(
                    f"sub-claim {v.sub_claim_id} has no citations but verdict={v.verdict.value}"
                )
            elif not v.reasoning.strip():
                errors.append(f"sub-claim {v.sub_claim_id} is NEI with no reasoning text")

    return ValidationResult(is_valid=(len(errors) == 0), errors=tuple(errors))


# === JSON serialization ======================================================


def to_jsonable(chain: EvidenceChain) -> dict:
    """Serialize an EvidenceChain to a JSON-safe dict.

    `passages_by_id` is inlined as a dict-of-dicts; nothing has to be looked
    up elsewhere to re-render the chain from JSON.
    """
    return {
        "claim": chain.claim,
        "final_verdict": chain.final_verdict.value,
        "final_confidence": float(chain.final_confidence),
        "sub_claims": [
            {
                "id": sc.id,
                "text": sc.text,
                "depends_on": list(sc.depends_on),
                "reasoning_type": sc.reasoning_type.value,
            }
            for sc in chain.sub_claims
        ],
        "verifications": [
            {
                "sub_claim_id": v.sub_claim_id,
                "verdict": v.verdict.value,
                "confidence": float(v.confidence),
                "cited_passage_ids": list(v.cited_passage_ids),
                "reasoning": v.reasoning,
            }
            for v in chain.verifications
        ],
        "passages_by_id": {
            pid: {
                "doc_id": p.doc_id,
                "title": p.title,
                "sent_idx": p.sent_idx,
                "text": p.text,
                "score": float(p.score),
            }
            for pid, p in chain.passages_by_id.items()
        },
    }


def from_jsonable(data: dict) -> EvidenceChain:
    """Reconstruct an EvidenceChain from a previously-serialized dict."""
    sub_claims = tuple(
        SubClaim(
            id=int(sc["id"]),
            text=sc["text"],
            depends_on=tuple(int(d) for d in sc["depends_on"]),
            reasoning_type=ReasoningType(sc["reasoning_type"]),
        )
        for sc in data["sub_claims"]
    )
    verifications = tuple(
        SubClaimVerification(
            sub_claim_id=int(v["sub_claim_id"]),
            verdict=Label(v["verdict"]),
            confidence=float(v["confidence"]),
            cited_passage_ids=tuple(v["cited_passage_ids"]),
            reasoning=v.get("reasoning", ""),
        )
        for v in data["verifications"]
    )
    passages_by_id = {
        pid: Passage(
            doc_id=p["doc_id"],
            title=p["title"],
            sent_idx=int(p["sent_idx"]),
            text=p["text"],
            score=float(p.get("score", 0.0)),
        )
        for pid, p in data.get("passages_by_id", {}).items()
    }
    return EvidenceChain(
        claim=data["claim"],
        sub_claims=sub_claims,
        verifications=verifications,
        final_verdict=Label(data["final_verdict"]),
        final_confidence=float(data["final_confidence"]),
        passages_by_id=passages_by_id,
    )


def dependency_path(chain: EvidenceChain) -> list[int]:
    """Topological order over the sub-claim depends_on DAG.

    Returns sub_claim ids in an order where each sub-claim's deps come first.
    Helpful for rendering a `1 → 2 → 3` chain in a human-readable layout.
    """
    indeg: dict[int, int] = {sc.id: 0 for sc in chain.sub_claims}
    children: dict[int, list[int]] = {sc.id: [] for sc in chain.sub_claims}
    for sc in chain.sub_claims:
        for dep in sc.depends_on:
            if dep in indeg:
                indeg[sc.id] += 1
                children[dep].append(sc.id)

    # Khan's algorithm
    queue = deque([sid for sid, d in indeg.items() if d == 0])
    out: list[int] = []
    while queue:
        sid = queue.popleft()
        out.append(sid)
        for child in children[sid]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    return out
