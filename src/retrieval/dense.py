"""Dense retriever — bge-small over a FAISS IndexFlatIP corpus.

Phase 02 ships the real retriever (not a stub) because the corpus + index
were built in Phase 01 and there's no benefit to mocking. Phase 04 / 05 will
extend this module (BM25 baseline alongside, optional fine-tuned encoder).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.schema import Passage

if TYPE_CHECKING:
    from src.config import CorpusConfig, RetrieverConfig


class DenseRetriever:
    """Encoder-and-FAISS retriever. Lazy-loads to keep test_schema.py fast."""

    def __init__(
        self,
        retriever_cfg: RetrieverConfig,
        corpus_cfg: CorpusConfig,
        *,
        model_override: str | None = None,
        index_override: str | None = None,
    ) -> None:
        """`model_override` / `index_override` let Phase 05 evaluate base vs
        fine-tune side-by-side without mutating `PipelineConfig`."""
        self.cfg = retriever_cfg
        self.corpus_cfg = corpus_cfg
        self._model_override = model_override
        self._index_override = index_override
        self._model = None
        self._index = None
        self._df = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import faiss
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        encoder_path = self._model_override or self.cfg.finetune_path or self.cfg.encoder
        index_path = self._index_override or self.corpus_cfg.faiss_path

        logger.info("loading dense encoder: {}", encoder_path)
        self._model = SentenceTransformer(encoder_path, device="cpu")
        self._model.max_seq_length = 256

        logger.info("loading FAISS index: {}", index_path)
        self._index = faiss.read_index(index_path)

        logger.info("loading corpus parquet: {}", self.corpus_cfg.parquet_path)
        self._df = pd.read_parquet(self.corpus_cfg.parquet_path)

        if self._index.ntotal != len(self._df):
            raise RuntimeError(
                f"Index ntotal={self._index.ntotal} disagrees with corpus rows={len(self._df)}"
            )

    def retrieve(self, text: str, top_k: int | None = None) -> list[Passage]:
        self._ensure_loaded()
        k = top_k or self.cfg.top_k
        query = (self.cfg.query_prefix or "") + text
        qv = self._model.encode(  # type: ignore[union-attr]
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores, ids = self._index.search(qv.astype("float32"), k)  # type: ignore[union-attr]

        results: list[Passage] = []
        for s, i in zip(scores[0], ids[0], strict=True):
            if i < 0:
                continue
            row = self._df.iloc[int(i)]  # type: ignore[union-attr]
            results.append(
                Passage(
                    doc_id=str(row["doc_id"]),
                    title=str(row["title"]),
                    sent_idx=int(row["sent_idx"]),
                    text=str(row["text"]),
                    score=float(s),
                )
            )
        return results
