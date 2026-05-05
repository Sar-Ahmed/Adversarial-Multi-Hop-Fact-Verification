"""System prompt + few-shot examples for the claim decomposer.

Six examples cover the patterns the decomposer must handle:
 - lookup        — single atomic claim (no decomposition needed)
 - compound      — "and"-joined independent facts
 - negation      — preserve polarity
 - composition   — "X who did Y also did Z" (depends_on)
 - comparison    — "A more than B" (3 sub-claims, depends_on)
 - temporal      — explicit dates / years

Lifted in shape from V1's decomposer/examples.py and adapted for the V3
schema (id / depends_on / reasoning_type fields). Negation is the third
example to avoid the V1 bug where putting it first causes downstream
over-negation in the model's outputs.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """You decompose compound claims into atomic sub-claims for fact-checking.

Output ONLY a JSON array. Each sub-claim is an object with these fields:
  - id: integer, 0-indexed
  - text: a single atomic claim that can be verified against one passage
  - depends_on: list of earlier sub-claim ids this builds on (or [])
  - reasoning_type: one of "lookup", "comparison", "temporal", "composition", "negation", "other"

Rules:
- Each sub-claim must be independently verifiable.
- Preserve the original claim's polarity. Negations stay negative.
- For "X who did Y also did Z" patterns, emit two sub-claims with depends_on linking them.
- For comparisons, emit one sub-claim per side plus a third comparison sub-claim depending on both.
- Keep the total sub-claim count between 1 and 5.
- Output ONLY the JSON array. No preamble, no explanation, no trailing text."""


_EXAMPLES: list[tuple[str, list[dict]]] = [
    # 1. Simple lookup — no decomposition needed
    (
        "The Eiffel Tower is located in Paris.",
        [
            {
                "id": 0,
                "text": "The Eiffel Tower is located in Paris.",
                "depends_on": [],
                "reasoning_type": "lookup",
            }
        ],
    ),
    # 2. Compound: two independent facts joined by "and"
    (
        "Christopher Nolan directed Inception and won an Oscar for Best Director.",
        [
            {
                "id": 0,
                "text": "Christopher Nolan directed Inception.",
                "depends_on": [],
                "reasoning_type": "lookup",
            },
            {
                "id": 1,
                "text": "Christopher Nolan won an Oscar for Best Director.",
                "depends_on": [],
                "reasoning_type": "lookup",
            },
        ],
    ),
    # 3. Negation — preserve polarity (placed third so model doesn't over-negate later examples)
    (
        "Tom Cruise has never won an Academy Award.",
        [
            {
                "id": 0,
                "text": "Tom Cruise has never won an Academy Award.",
                "depends_on": [],
                "reasoning_type": "negation",
            }
        ],
    ),
    # 4. Composition: "the X who did Y also did Z"
    (
        "The director of Inception also directed The Dark Knight.",
        [
            {
                "id": 0,
                "text": "Inception was directed by a specific person.",
                "depends_on": [],
                "reasoning_type": "lookup",
            },
            {
                "id": 1,
                "text": "That same person directed The Dark Knight.",
                "depends_on": [0],
                "reasoning_type": "composition",
            },
        ],
    ),
    # 5. Comparison: needs both sides plus a comparison step
    (
        "The Dark Knight grossed more than Inception worldwide.",
        [
            {
                "id": 0,
                "text": "The Dark Knight has a known worldwide gross.",
                "depends_on": [],
                "reasoning_type": "lookup",
            },
            {
                "id": 1,
                "text": "Inception has a known worldwide gross.",
                "depends_on": [],
                "reasoning_type": "lookup",
            },
            {
                "id": 2,
                "text": "The Dark Knight worldwide gross exceeds Inception worldwide gross.",
                "depends_on": [0, 1],
                "reasoning_type": "comparison",
            },
        ],
    ),
    # 6. Temporal: explicit year + factual claim
    (
        "Inception was released in 2010 and grossed over $800 million.",
        [
            {
                "id": 0,
                "text": "Inception was released in 2010.",
                "depends_on": [],
                "reasoning_type": "temporal",
            },
            {
                "id": 1,
                "text": "Inception grossed over $800 million.",
                "depends_on": [],
                "reasoning_type": "lookup",
            },
        ],
    ),
]


def build_messages(claim: str, *, retry: bool = False) -> list[dict[str, str]]:
    """Build the chat-completion messages for one decomposition call."""
    sys_text = SYSTEM_PROMPT
    if retry:
        sys_text += (
            "\n\nIMPORTANT: your previous response could not be parsed. "
            "Output the JSON array and nothing else. No prose, no markdown fences."
        )

    messages: list[dict[str, str]] = [{"role": "system", "content": sys_text}]
    for in_claim, decomp in _EXAMPLES:
        messages.append({"role": "user", "content": f"Claim: {in_claim}"})
        messages.append({"role": "assistant", "content": json.dumps(decomp)})
    messages.append({"role": "user", "content": f"Claim: {claim}"})
    return messages


def prompt_hash() -> str:
    """Stable hash of the prompt + few-shot. Logged in eval JSONs for traceability."""
    import hashlib

    blob = SYSTEM_PROMPT + "\n".join(f"{c}|{json.dumps(d, sort_keys=True)}" for c, d in _EXAMPLES)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
