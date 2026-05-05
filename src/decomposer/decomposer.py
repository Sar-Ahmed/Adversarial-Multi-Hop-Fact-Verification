"""Real claim decomposer: Qwen2.5-3B-Q4 few-shot with retry + safe fallback.

Replaces `StubDecomposer`. Implements the spec's requirement #1 (atomic
sub-claims with dependency edges, including the "X who did Y also did Z"
nested-logic case).

Robustness layers (order matters):
 1. Greedy generation, regex extraction of the first balanced [ ... ] array.
 2. On parse failure, retry once with a stricter "JSON only, no preamble" nudge.
 3. On second failure, fall back to a single sub-claim wrapping the whole input.

V1 reported 0% fallback rate on n=200 with this exact strategy. We aim to
match.
"""

from __future__ import annotations

import json

from loguru import logger

from src.decomposer.llm import LocalLLM
from src.decomposer.prompts import build_messages
from src.schema import ReasoningType, SubClaim


class _ParseError(Exception):
    """Internal — signals a retry / fallback to the outer decompose() loop."""


class Decomposer:
    """Few-shot decomposer over a local Qwen 3B Q4 model."""

    def __init__(
        self,
        llm_path: str,
        n_ctx: int = 4096,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.llm = LocalLLM(model_path=llm_path, n_ctx=n_ctx, seed=seed)
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Per-call signal — read by eval_decomposer.py to compute fallback rate
        # without conflating "parser actually failed" with "model correctly emitted
        # one sub-claim because the input was atomic".
        self.last_call_used_fallback: bool = False

    def decompose(self, claim: str) -> list[SubClaim]:
        """Return atomic sub-claims for `claim`. Sets `last_call_used_fallback`."""
        if not claim.strip():
            raise ValueError("Decomposer.decompose called with empty claim")
        self.last_call_used_fallback = False
        try:
            return self._try_decompose(claim, retry=False)
        except _ParseError as e1:
            logger.warning("decompose attempt 1 failed: {} — retrying", e1)
            try:
                return self._try_decompose(claim, retry=True)
            except _ParseError as e2:
                logger.warning(
                    "decompose attempt 2 failed: {} — using single-sub-claim fallback", e2
                )
                self.last_call_used_fallback = True
                return [
                    SubClaim(
                        id=0,
                        text=claim,
                        depends_on=(),
                        reasoning_type=ReasoningType.OTHER,
                    )
                ]

    def _try_decompose(self, claim: str, *, retry: bool) -> list[SubClaim]:
        messages = build_messages(claim, retry=retry)
        raw = self.llm.chat(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["\n\n\n", "Claim:"],
        )
        json_text = self._extract_json_array(raw)
        if json_text is None:
            raise _ParseError(f"could not extract JSON array from output: {raw[:200]!r}")
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise _ParseError(
                f"JSON decode failed: {e}; tried to parse: {json_text[:200]!r}"
            ) from e
        if not isinstance(parsed, list):
            raise _ParseError(f"parsed JSON is not a list: type={type(parsed).__name__}")
        return self._build_subclaims(parsed)

    @staticmethod
    def _extract_json_array(text: str) -> str | None:
        """Scan for the first balanced JSON array in `text`. Returns the substring."""
        start = text.find("[")
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
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @staticmethod
    def _build_subclaims(parsed: list) -> list[SubClaim]:
        result: list[SubClaim] = []
        seen_ids: set[int] = set()
        for i, entry in enumerate(parsed):
            if not isinstance(entry, dict):
                raise _ParseError(f"entry {i} is not a dict: {type(entry).__name__}")

            # Resolve id, renumbering on duplicates.
            try:
                sid = int(entry.get("id", i))
            except (TypeError, ValueError):
                sid = i
            if sid in seen_ids:
                sid = (max(seen_ids) + 1) if seen_ids else i
            seen_ids.add(sid)

            text = str(entry.get("text", "")).strip()
            if not text:
                continue  # skip empty entries silently

            # depends_on: keep only ints that point to earlier ids in our renumbered set.
            deps_raw = entry.get("depends_on") or []
            deps: list[int] = []
            if isinstance(deps_raw, list):
                for d in deps_raw:
                    try:
                        di = int(d)
                    except (TypeError, ValueError):
                        continue
                    if di < sid and di in seen_ids:
                        deps.append(di)

            rt_raw = str(entry.get("reasoning_type", "other")).lower()
            try:
                rt = ReasoningType(rt_raw)
            except ValueError:
                rt = ReasoningType.OTHER

            result.append(
                SubClaim(
                    id=sid,
                    text=text,
                    depends_on=tuple(deps),
                    reasoning_type=rt,
                )
            )

        if not result:
            raise _ParseError("decomposition is empty after parsing")
        return result
