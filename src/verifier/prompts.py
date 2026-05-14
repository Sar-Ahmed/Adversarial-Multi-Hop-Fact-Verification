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


def build_messages(claim: str, passages_text: list[str]) -> list[dict[str, str]]:
    """Build the chat-completion messages for one verifier call."""
    numbered = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(passages_text))
    user = f"CLAIM: {claim}\nEVIDENCE:\n{numbered}\n\nOutput:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def prompt_hash() -> str:
    """Stable short hash of the prompt for traceability in eval JSONs."""
    import hashlib

    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]
