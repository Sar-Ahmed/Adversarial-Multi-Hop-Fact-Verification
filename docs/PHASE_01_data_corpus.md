# Phase 01 — Data and Corpus

**Goal.** Load HoVer + FEVER (train + dev), build a focused ~200k-passage Wikipedia corpus seeded from gold titles in both datasets, encode it with `bge-small-en-v1.5`, and save a FAISS flat-IP index.

**Effort.** 1–2 days.
**Compute.** CPU. (Encoding 200k passages with bge-small ≈ 30–60 min on a modern laptop CPU; ~5 min on Colab T4 if you prefer.)
**Depends on.** Phase 00.

## Why this exists

Spec requires retrieval over 2–4 Wikipedia passages per claim. We do not need full Wikipedia (5M+ articles) — V1 demonstrated that a focused 223k-passage corpus seeded from HoVer + FEVER gold titles gives R@10 = 0.92, which is enough headroom for the reasoner to be the bottleneck (which it is). This phase produces the corpus + index that every later retrieval phase reads.

## Inputs

- HoVer dataset (`hover` config in the HuggingFace `datasets` registry — the legacy `hover_fever` namespace from the spec is now just `hover`).
- FEVER dataset (`fever` config from `datasets`).
- Wikipedia dump source — we use FEVER's bundled `wiki-pages` (cached and immutable; same source V1 used).

## Deliverables

- `src/data/load.py` — `load_hover()`, `load_fever()` returning typed splits.
- `src/data/build_corpus.py` — produces `artifacts/corpus.parquet` (columns: `doc_id`, `title`, `sent_idx`, `text`).
- `src/data/encode_corpus.py` — produces `artifacts/corpus_embeddings.npy` (`float32 [N, 384]`, L2-normalized) and `artifacts/corpus.faiss` (FAISS IndexFlatIP).
- `artifacts/corpus_stats.json` — passage count, unique title count, avg passage length, dataset coverage.
- `tests/test_data_loaders.py` — verifies HoVer dev has the expected size, schema fields are present, no nulls in `claim` or `label` columns.

## Technical approach

- **Corpus construction.**
  1. Collect gold titles from HoVer train + dev and FEVER train + dev. Deduplicate.
  2. Walk FEVER `wiki-pages` and emit one row per `(title, sent_idx)` for every title in the gold set.
  3. Drop empty sentences and stub articles (<10 chars).
  4. Save as parquet (zstd compression) — typically 20–30 MB.
- **Encoding.**
  - Model: `BAAI/bge-small-en-v1.5` (base, no fine-tune yet).
  - Batch size 64 on CPU, 256 on Colab T4.
  - L2-normalize after encoding so dot-product == cosine similarity.
- **Index.**
  - FAISS `IndexFlatIP` over the normalized vectors. No training needed for flat indexes.
  - Save to disk; load via mmap in Phase 04 retriever.

## Implementation steps

1. `pip install datasets==2.20.0` (already pinned in Phase 00).
2. Write `load_hover()`/`load_fever()` returning `dict[split_name, list[Example]]`. Smoke-load 5 examples from each in pytest.
3. Write `build_corpus.py`:
   - Pull gold titles into a set.
   - Stream-iterate FEVER wiki-pages with `datasets`'s streaming mode to avoid loading 5 GB at once.
   - Filter, dedupe, write parquet.
4. Write `encode_corpus.py` using `sentence_transformers.SentenceTransformer.encode(..., normalize_embeddings=True, batch_size=64)`.
5. Build FAISS IndexFlatIP, `index.add(vectors)`, save with `faiss.write_index`.
6. Compute `corpus_stats.json` (counts + a histogram of passages-per-title).
7. Quick sanity check: encode a known claim ("Christopher Nolan directed Inception"), retrieve top-5, verify the Inception article shows up.

## Exit criteria

- [ ] `artifacts/corpus.parquet` exists, has between 150k and 280k rows, schema matches spec.
- [ ] `artifacts/corpus_embeddings.npy` shape is `[N, 384]`, dtype float32, every row has L2 norm ≈ 1.0 (within 1e-5).
- [ ] `artifacts/corpus.faiss` loads via `faiss.read_index` and `index.ntotal == N`.
- [ ] `artifacts/corpus_stats.json` written and committed.
- [ ] Sanity-check retrieval (Inception claim) returns at least one passage from the actual Inception article in top-5.
- [ ] `tests/test_data_loaders.py` passes.

## Risks and gotchas

- HuggingFace's `fever` config has *several* sub-configs (`v1.0`, `wiki_pages`, `paper_test`); load the one matching the FEVER paper's official splits.
- The `hover` dataset's dev split is the only labeled eval set we have — keep it untouched, never mix with train.
- FEVER `wiki-pages` titles use a custom encoding (e.g. `Inception_-LRB-2010_film-RRB-`); V1 has helper functions in `src/data/fetch_wiki.py` we can port. Reuse them.
- Memory: encoding 200k passages with bge-small in float32 = ~330 MB. Trivial. But float16 storage on disk is fine if we want half the size — just decode-on-load.

## What NOT to do

- Do not download the full English Wikipedia. Focused corpus only.
- Do not encode-as-you-retrieve. Pre-encode once.
- Do not include the dev gold passages in the corpus *only* — the corpus must include distractor candidates (other titles) too, otherwise retrieval is trivially perfect.

## Outcome (filled at end of phase)

> Append: final passage count, encoding wall time, sanity-check retrieval result, any title-encoding quirks.
