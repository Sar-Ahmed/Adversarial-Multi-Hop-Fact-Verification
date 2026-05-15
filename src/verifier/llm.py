"""LLM-only verifier: Qwen 3B Q4 with JSON output + safe NEI fallback.

Reuses `LocalLLM` from the decomposer module so the model loads once per
process and the KV cache is shared across decomposer + verifier calls
(huge throughput win on CPU).

Output: (verdict, confidence, reasoning).

`confidence` is intentionally a placeholder 0.6 (verdict != NEI) / 0.4 (NEI)
in Phase 07 — the 3B model's self-reported confidence is unreliable, and
Phase 08's calibrator builds the real confidence from NLI features.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from src.decomposer.llm import LocalLLM
from src.schema import Label, Passage
from src.verifier.prompts import build_messages

_VALID_VERDICTS = {"SUPPORTED": Label.SUPPORTED, "REFUTED": Label.REFUTED, "NEI": Label.NEI}


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced { ... } JSON object substring, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_verdict(raw: str) -> tuple[Label, str] | None:
    """Parse the LLM's JSON output into (Label, reasoning). Returns None on failure."""
    blob = _extract_json_object(raw)
    if blob is None:
        return None
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    verdict_raw = parsed.get("verdict")
    if not isinstance(verdict_raw, str):
        return None
    # Tolerate uppercase / case-insensitive variants
    verdict_key = verdict_raw.strip().upper()
    label = _VALID_VERDICTS.get(verdict_key)
    if label is None:
        # Common LLM mistake: "SUPPORTS" / "REFUTES" / "NOT ENOUGH INFO"
        norm = re.sub(r"[^A-Z]", "", verdict_key)
        if norm.startswith("SUPP"):
            label = Label.SUPPORTED
        elif norm.startswith("REF"):
            label = Label.REFUTED
        elif "NEI" in norm or "NOTENOUGH" in norm or "INSUFF" in norm:
            label = Label.NEI
        else:
            return None
    reason = str(parsed.get("reason", "")).strip()
    return label, reason


class LLMVerifier:
    """Qwen 3B Q4 verifier. Three-class verdict from claim + passages."""

    def __init__(
        self,
        llm_path: str,
        n_ctx: int = 4096,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int = 42,
        prompt_variant: str = "v1",
    ) -> None:
        self.llm = LocalLLM(model_path=llm_path, n_ctx=n_ctx, seed=seed)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_variant = prompt_variant

    def verify(self, claim: str, passages: list[Passage]) -> tuple[Label, float, str]:
        """Return (verdict, confidence, reasoning)."""
        if not claim.strip():
            raise ValueError("LLMVerifier.verify called with empty claim")

        if not passages:
            return Label.NEI, 0.4, "no passages retrieved"

        passages_text = [p.text for p in passages]
        messages = build_messages(claim, passages_text, variant=self.prompt_variant)
        raw = self.llm.chat(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["\n\n\n", "CLAIM:"],
        )
        parsed = _parse_verdict(raw)
        if parsed is None:
            logger.warning("verifier parse failure; raw output: {!r}", raw[:200])
            return Label.NEI, 0.3, f"parse failure: {raw[:120]!r}"

        verdict, reasoning = parsed
        confidence = 0.6 if verdict is not Label.NEI else 0.4
        return verdict, confidence, reasoning
