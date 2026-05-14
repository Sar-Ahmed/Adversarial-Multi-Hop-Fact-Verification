"""NLI scoring of (claim, passage) pairs via `cross-encoder/nli-deberta-v3-base`.

For each passage we get softmax probabilities (contradiction, entailment,
neutral). The aggregator (Phase 07 / aggregate.py) reads `max_contra` and
`max_entail` across passages to decide whether to veto the LLM's verdict.

The contradiction label is at index 0; entailment at 1; neutral at 2. This
matches the model card for `cross-encoder/nli-deberta-v3-base` and is the
same encoding Phase 06 mining uses.
"""

from __future__ import annotations

from loguru import logger

from src.schema import Passage

NLI_CONTRA_IDX = 0
NLI_ENTAIL_IDX = 1
NLI_NEUTRAL_IDX = 2


class NLIVerifier:
    """Lazy-loaded NLI cross-encoder; returns per-passage scores + aggregates."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        max_length: int = 256,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        logger.info("loading NLI model: {}", self.model_name)
        self._model = CrossEncoder(self.model_name, device="cpu", max_length=self.max_length)

    def score(self, claim: str, passages: list[Passage]) -> dict:
        """Return per-passage NLI scores + aggregate max_contra / max_entail.

        Output shape::

            {
                "per_passage": [
                    {"doc_id": "...", "contra": 0.91, "entail": 0.03, "neutral": 0.06},
                    ...
                ],
                "max_contra": float,
                "max_entail": float,
                "max_neutral": float,
                "n_passages": int,
            }

        Empty `passages` returns all zeros (NLI cannot fire a veto when there
        is no evidence to read).
        """
        if not passages:
            return {
                "per_passage": [],
                "max_contra": 0.0,
                "max_entail": 0.0,
                "max_neutral": 0.0,
                "n_passages": 0,
            }
        self._ensure_loaded()
        pairs = [(claim, p.text) for p in passages]
        scores = self._model.predict(  # type: ignore[union-attr]
            pairs,
            batch_size=self.batch_size,
            apply_softmax=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )  # shape (N, 3)

        per_passage: list[dict] = []
        for i, p in enumerate(passages):
            per_passage.append(
                {
                    "doc_id": p.doc_id,
                    "contra": float(scores[i, NLI_CONTRA_IDX]),
                    "entail": float(scores[i, NLI_ENTAIL_IDX]),
                    "neutral": float(scores[i, NLI_NEUTRAL_IDX]),
                }
            )
        max_contra = float(scores[:, NLI_CONTRA_IDX].max())
        max_entail = float(scores[:, NLI_ENTAIL_IDX].max())
        max_neutral = float(scores[:, NLI_NEUTRAL_IDX].max())
        return {
            "per_passage": per_passage,
            "max_contra": max_contra,
            "max_entail": max_entail,
            "max_neutral": max_neutral,
            "n_passages": len(passages),
        }
