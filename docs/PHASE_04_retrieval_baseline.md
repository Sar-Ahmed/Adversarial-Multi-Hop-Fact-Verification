# Phase 04 — Retrieval Baseline (BM25 / Dense / Reranker)

**Goal.** Replace `RetrieverStub` and `RerankerStub` with real BM25 + dense + cross-encoder rerank components. Establish the baseline R@5 / R@10 / R@20 numbers before any fine-tuning. Spec requirement #2 (first half).

**Effort.** 1–2 days.
**Compute.** CPU. Reranker is the slowest component here; ~1–2 s per (query, 50-passage) batch on a modern laptop CPU.
**Depends on.** Phase 01 (corpus + index), Phase 02 (config + smoke test), Phase 03 (real decomposer feeding sub-claims to retrieve).

## Why this exists

We need the baseline numbers before Phase 05 fine-tuning so we can prove that fine-tuning helps (or honestly admit it doesn't — V1's case). We also need the reranker because the spec's failure mode #1 is that surface-level cosine similarity destroys retrieval; a cross-encoder is the standard fix.

## Inputs

- `artifacts/corpus.parquet`, `artifacts/corpus.faiss`, `artifacts/corpus_embeddings.npy` from Phase 01.
- HoVer dev split from Phase 01 loader.
- `BAAI/bge-small-en-v1.5` (already used in encoding) and `BAAI/bge-reranker-base` (downloaded by `scripts/fetch_models.sh`).

## Deliverables

- `src/retrieval/bm25.py` — BM25 retriever over the same corpus (uses `rank_bm25.BM25Okapi`).
- `src/retrieval/dense.py` — dense retriever with FAISS, query prefix `"Represent this sentence for searching relevant passages: "`, top-50 default.
- `src/reranker/cross_encoder.py` — `Reranker.rerank(query, candidates, top_k)` using `sentence_transformers.CrossEncoder`.
- `src/eval/retrieval_eval.py` — runs each retriever on the HoVer dev gold set, reports R@5, R@10, R@20 with bootstrap 95% CIs.
- `artifacts/retrieval_eval_baseline.json` — committed result with model names, top_k, query prefix, seed.
- `tests/test_retrieval.py` — unit tests verifying:
  - dense retriever returns L2-normalized scores in [-1, 1] and `len(results) == top_k`.
  - reranker output is sorted descending by score.
  - "Inception" claim retrieves Inception-titled passage in top-5 (sanity).

## Technical approach

- **BM25.** Tokenize corpus once at startup (cache on disk via pickle). Score over the full 200k passages — millisecond latency. Same interface as dense.
- **Dense.** Reuse the FAISS IndexFlatIP from Phase 01. Encode query with bge-small + the BGE query prefix; faiss search top-50.
- **Reranker.** `CrossEncoder("BAAI/bge-reranker-base").predict([(query, passage_text), ...])`. Batch size 16 on CPU, 64 on GPU. Returns scores; sort descending, return top-10.
- **Common interface.** Each retriever exposes `retrieve(text: str, top_k: int) -> list[Passage]` so they're drop-in swappable.

## Implementation steps

1. Implement `bm25.py` — load parquet, tokenize, build `BM25Okapi`, save tokenized corpus to disk to avoid re-tokenizing each run.
2. Implement `dense.py` — load FAISS, load corpus parquet (for doc_id/title/text lookup by row index), implement `retrieve()`.
3. Implement `cross_encoder.py` — load `BAAI/bge-reranker-base` via `sentence_transformers.CrossEncoder`, score (query, text) pairs, return top-k.
4. Wire all three into `pipeline.py`. Default config: dense + reranker. BM25 available via `retriever.kind: bm25`.
5. Write `retrieval_eval.py`:
   - Load HoVer dev (train→eval not used here).
   - For each example: retrieve top-K with each retriever, check if gold passages are in the result.
   - Recall@K = fraction of gold passages found in top-K.
   - Bootstrap 1000-sample CI over the example axis.
6. Run all three (`bm25`, `dense`, `dense+reranker`) on full HoVer dev (~4k examples) — ~30–60 min total.
7. Save `artifacts/retrieval_eval_baseline.json`.

## Exit criteria

- [ ] `make smoke` still passes (now using real dense + reranker).
- [ ] BM25 R@10 ≥ 0.80 on HoVer dev.
- [ ] Dense R@10 ≥ 0.85 on HoVer dev.
- [ ] Dense + reranker R@10 ≥ 0.88 on HoVer dev.
- [ ] All three numbers reported with 95% CI in `artifacts/retrieval_eval_baseline.json`.
- [ ] Sanity check (Inception claim) returns the correct passage in top-3 with each retriever.
- [ ] Reranker latency ≤ 2 s per query on top-50 batch (CPU).

## Risks and gotchas

- BM25 over 200k passages is fast but tokenization is the hidden cost — cache aggressively.
- bge-small embeddings need the query prefix for retrieval but **not** for indexing. This is asymmetric — easy to get wrong. Verify by encoding "Inception" with and without the prefix and confirming the asymmetric variant retrieves the Inception article.
- HoVer's gold passages are identified by `(title, sentence_id)` — make sure your row→passage map handles this exactly. V1's `fever_encode`/`fever_decode` helpers are needed for FEVER-encoded titles.
- Don't include the query in the corpus by accident (search-the-database-for-itself bug).

## What NOT to do

- Do not try to swap `IndexFlatIP` for `IndexHNSW` here. We're at 200k passages — flat is fine. HNSW is a Phase 05+ optimization that is likely unneeded.
- Do not fine-tune the reranker yet (a fine-tuned reranker on (claim, passage, label) is a possible Phase 12 ablation, not core).
- Do not ship a top_k that varies between train and eval. Pick top-50 → top-10 and stick to it.

## Outcome (filled at end of phase)

> Append: BM25 / dense / dense+rerank R@5/10/20 with CIs, total wall time, sanity check log.
