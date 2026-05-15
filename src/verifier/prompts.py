"""Verifier prompt template + JSON parsing for the Qwen 3B LLM.

The LLM sees the claim and a numbered list of evidence passages and decides
SUPPORTED / REFUTED / NEI. Output is a JSON object so we can parse it
mechanically; if parsing fails, the verifier falls back to NEI rather than
guessing.

Two design choices worth flagging:

- We *don't* ask the LLM for a numeric confidence. The 3B model's
  self-reported confidence is unreliable, and Phase 08's calibrator builds
  proper confidence from NLI features instead. The LLM's job is just verdict +
  one-sentence reason.
- Three short few-shot examples (one per class) are baked into the system
  message. This is the cheapest format-anchor available and avoids the
  multi-turn few-shot expansion the decomposer uses.

## Variants

- **v1** (`SYSTEM_PROMPT`): the v3.0 production prompt. Strict NEI rule:
  "if on-topic but doesn't address the specific assertion, return NEI".
- **v2** (`SYSTEM_PROMPT_V2`): the v3.1 soft-prompt candidate (Phase 16).
  Replaces the strict rule with a "lean toward SUPPORTED/REFUTED when evidence
  partially addresses the claim" rule. NEI is reserved for off-topic evidence.

`build_messages(claim, passages, variant=...)` picks the prompt; the default
is `"v1"` to preserve v3.0 behavior for any caller that doesn't specify.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a fact-checker. Given a CLAIM and a numbered list of EVIDENCE passages, output a JSON object that decides one of three verdicts:

- SUPPORTED: the evidence clearly supports the claim.
- REFUTED: the evidence clearly contradicts the claim.
- NEI: the evidence does not contain enough information to decide.

Rules:
- Reason ONLY from the evidence shown. Do not use outside knowledge.
- If the evidence is on-topic but doesn't address the specific assertion, return NEI.
- If even part of the claim is contradicted, return REFUTED.

Output ONLY a JSON object with exactly these fields:
- "verdict": one of "SUPPORTED", "REFUTED", "NEI"
- "reason": one short sentence citing which passage(s) you used

Examples:

CLAIM: The Eiffel Tower is in Paris.
EVIDENCE:
[1] The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.

Output:
{"verdict": "SUPPORTED", "reason": "Passage 1 directly states the Eiffel Tower is in Paris."}

CLAIM: The Eiffel Tower is in Berlin.
EVIDENCE:
[1] The Eiffel Tower is in Paris, France.

Output:
{"verdict": "REFUTED", "reason": "Passage 1 states the Eiffel Tower is in Paris, not Berlin."}

CLAIM: Christopher Nolan owns a yacht in Monaco.
EVIDENCE:
[1] Christopher Nolan is a British-American filmmaker known for his nonlinear storylines.

Output:
{"verdict": "NEI", "reason": "Passage 1 describes Nolan's filmmaking but says nothing about yacht ownership."}"""


SYSTEM_PROMPT_V2 = """You are a fact-checker. Given a CLAIM and a numbered list of EVIDENCE passages, output a JSON object that decides one of three verdicts:

- SUPPORTED: the evidence supports the claim.
- REFUTED: the evidence contradicts the claim.
- NEI: the evidence does not address the claim's key entities or assertions.

Rules:
- Reason ONLY from the evidence shown. Do not use outside knowledge.
- If any part of the claim is directly contradicted by the evidence, return REFUTED.
- If every key part of the claim is directly supported by the evidence, return SUPPORTED.
- If the claim has multiple parts and the evidence taken together clearly leans one way (even if not every part has a perfect cite), prefer SUPPORTED or REFUTED over NEI.
- Return NEI only when the evidence is off-topic — it does not mention the claim's entities or assertions at all.

Output ONLY a JSON object with exactly these fields:
- "verdict": one of "SUPPORTED", "REFUTED", "NEI"
- "reason": one short sentence citing which passage(s) you used

Examples:

CLAIM: The Eiffel Tower is in Paris.
EVIDENCE:
[1] The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.

Output:
{"verdict": "SUPPORTED", "reason": "Passage 1 directly states the Eiffel Tower is in Paris."}

CLAIM: The director of Inception was born in London.
EVIDENCE:
[1] Christopher Nolan is a British-American filmmaker born in London on 30 July 1970.
[2] Inception is a 2010 science fiction action film written and directed by Christopher Nolan.

Output:
{"verdict": "SUPPORTED", "reason": "Passages 1-2 together establish Nolan directed Inception and was born in London."}

CLAIM: The Eiffel Tower is in Berlin.
EVIDENCE:
[1] The Eiffel Tower is in Paris, France.

Output:
{"verdict": "REFUTED", "reason": "Passage 1 states the Eiffel Tower is in Paris, not Berlin."}

CLAIM: Christopher Nolan owns a yacht in Monaco.
EVIDENCE:
[1] The Sahara is the largest hot desert in the world, covering most of North Africa.
[2] Bananas are an important commercial crop in tropical regions.

Output:
{"verdict": "NEI", "reason": "Neither passage mentions Christopher Nolan, yachts, or Monaco."}"""


_PROMPTS = {
    "v1": SYSTEM_PROMPT,
    "v2": SYSTEM_PROMPT_V2,
}


def get_prompt(variant: str = "v1") -> str:
    """Return the system prompt text for the named variant."""
    if variant not in _PROMPTS:
        raise ValueError(f"unknown prompt variant {variant!r}; expected one of {list(_PROMPTS)}")
    return _PROMPTS[variant]


def build_messages(
    claim: str,
    passages_text: list[str],
    variant: str = "v1",
) -> list[dict[str, str]]:
    """Build the chat-completion messages for one verifier call."""
    numbered = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(passages_text))
    user = f"CLAIM: {claim}\nEVIDENCE:\n{numbered}\n\nOutput:"
    return [
        {"role": "system", "content": get_prompt(variant)},
        {"role": "user", "content": user},
    ]


def prompt_hash(variant: str = "v1") -> str:
    """Stable short hash of the named prompt for traceability in eval JSONs."""
    import hashlib

    return hashlib.sha256(get_prompt(variant).encode("utf-8")).hexdigest()[:12]
