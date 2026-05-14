"""Feature extractor for the NEI calibrator.

Eleven cheap-to-compute features that don't require the LLM. The Phase 08
spec called for 12 features including LLM verdict one-hot, but on this
compute budget the LLM call per FEVER training example would cost ~5 h.
We drop the LLM feature and rely on NLI + retrieval + lexical signals.

Documented gap: dropping LLM verdict from the feature set probably leaves
~3-5 points of accuracy on the table. Phase 13 error analysis will tell us
whether it matters. If it does, retrain with the LLM one-hot column (cached
in artifacts/per_subclaim_traces.jsonl for HoVer dev) once we have FEVER
LLM traces too.
"""

from __future__ import annotations

import re

import numpy as np

from src.schema import Passage

# Minimal stoplist — keeps the entity-overlap feature focused on content tokens.
_STOPLIST: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "their",
        "his",
        "her",
        "him",
        "i",
        "you",
        "we",
        "us",
        "our",
        "your",
        "my",
        "me",
        "not",
        "no",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

FEATURE_NAMES: tuple[str, ...] = (
    "nli_max_contra",
    "nli_max_entail",
    "nli_max_neutral",
    "nli_mean_contra",
    "nli_mean_entail",
    "nli_contra_minus_entail_top1",
    "retrieval_top1_score",
    "retrieval_score_gap_top1_top5",
    "claim_n_words",
    "mean_passage_length_chars",
    "entity_overlap_top1_jaccard",
)
N_FEATURES = len(FEATURE_NAMES)


def _content_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return {t for t in (m.lower() for m in _TOKEN_RE.findall(text)) if t not in _STOPLIST}


def extract_features(
    claim: str,
    top_passages: list[Passage],
    nli_scores: dict,
) -> np.ndarray:
    """Return a `(N_FEATURES,)` float32 vector for the calibrator.

    `top_passages` is the top-K reranked list passed to the verifier.
    `nli_scores` is the dict returned by `NLIVerifier.score(claim, top_passages)`
    — must contain `max_contra`, `max_entail`, `max_neutral`, and `per_passage`.

    Returns zeros if `top_passages` is empty (downstream calibrator should
    handle this as a feature-of-its-own).
    """
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    if not top_passages:
        return vec

    n_pass = len(top_passages)

    # NLI aggregates (5 features)
    vec[0] = float(nli_scores.get("max_contra", 0.0))
    vec[1] = float(nli_scores.get("max_entail", 0.0))
    vec[2] = float(nli_scores.get("max_neutral", 0.0))
    per = nli_scores.get("per_passage") or []
    if per:
        vec[3] = float(np.mean([p["contra"] for p in per]))
        vec[4] = float(np.mean([p["entail"] for p in per]))
        vec[5] = float(per[0]["contra"] - per[0]["entail"])

    # Retrieval (2)
    scores = [float(p.score) for p in top_passages]
    vec[6] = scores[0]
    if n_pass >= 5:
        vec[7] = scores[0] - scores[4]
    else:
        vec[7] = scores[0] - scores[-1]

    # Lexical / shape (3)
    vec[8] = float(len(_TOKEN_RE.findall(claim)))
    vec[9] = float(np.mean([len(p.text) for p in top_passages]))

    claim_toks = _content_tokens(claim)
    top1_toks = _content_tokens(top_passages[0].text)
    if claim_toks and top1_toks:
        intersection = len(claim_toks & top1_toks)
        union = len(claim_toks | top1_toks)
        vec[10] = float(intersection / union) if union > 0 else 0.0

    return vec
