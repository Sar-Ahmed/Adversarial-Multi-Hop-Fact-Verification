# Phase 02 — Schema and Pipeline Scaffolding

**Goal.** Define the dataclass schema, the frozen `PipelineConfig`, and an end-to-end `pipeline.verify(claim)` that runs the *shape* of the system on real corpus data using stub components for the not-yet-built pieces. Wire the 5-example integration smoke test that gates every PR from now on.

**Effort.** 1 day.
**Compute.** CPU.
**Depends on.** Phase 00, Phase 01.

## Why this exists

V2's biggest failure was that the pipeline crashed on its first call because the schema and the call site disagreed (`SubClaim(sub_claim_id=...)` vs `SubClaim(id=...)`). That bug existed because nothing exercised the real call path. V3 makes the integration smoke test the gate from Phase 02 onward, so this can never recur.

We also commit to "every component takes `PipelineConfig` as a constructor arg" here. By the time real models arrive in Phase 03+, the contract is already enforced.

## Inputs

- `artifacts/corpus.parquet`, `artifacts/corpus.faiss` from Phase 01.
- The schema designs from V2 (good shape, broken instantiation) and V1 (correct but lighter).

## Deliverables

- `src/schema.py` — final schema:
  - `Label` enum: `SUPPORTED`, `REFUTED`, `NEI`.
  - `ReasoningType` enum: `lookup`, `comparison`, `temporal`, `composition`, `negation`, `other`.
  - `@dataclass(frozen=True) Passage(doc_id, title, sent_idx, text, score)`.
  - `@dataclass SubClaim(id: int, text: str, depends_on: list[int], reasoning_type: ReasoningType)`.
  - `@dataclass SubClaimVerification(sub_claim_id: int, verdict: Label, confidence: float, cited_passage_ids: list[str], reasoning: str)`.
  - `@dataclass EvidenceChain(claim: str, sub_claims: list[SubClaim], verifications: list[SubClaimVerification], final_verdict: Label, final_confidence: float)`.
- `src/config.py` — `PipelineConfig` (pydantic BaseModel; immutable after load) with sub-models for each component.
- `src/pipeline.py` — `Pipeline.verify(claim: str) -> EvidenceChain` wired with stub components.
- `src/cli.py` — `python -m src.cli verify "<claim>"` entry point.
- `src/decomposer/stub.py`, `src/retrieval/stub.py`, `src/reranker/stub.py`, `src/verifier/stub.py` — minimal stubs that satisfy the interface and are replaced in later phases. Stubs **may** return canned data; they **must not** silently mask missing models.
- `tests/test_smoke.py` — 5 fixed claims, end-to-end run, asserts shape and types (not values). Marked `@pytest.mark.smoke`.
- `tests/test_schema.py` — round-trip serialization, enum validation, depends_on integrity.

## Technical approach

- **Schema:** dataclasses in `src/schema.py`. Use `__post_init__` to validate: `depends_on` ids reference existing sub-claims; `confidence` ∈ [0, 1].
- **Config:** pydantic v2 `BaseModel` with `model_config = ConfigDict(frozen=True)`. One `PipelineConfig.load(path)` classmethod reads YAML and instantiates the full nested tree.
- **Stubs:**
  - `DecomposerStub.decompose(claim)` returns one SubClaim wrapping the claim.
  - `RetrieverStub.retrieve(text)` returns the top-50 from the real FAISS index using bge-small base — this is real; "stub" only means we don't have fine-tune yet.
  - `RerankerStub.rerank(query, candidates)` returns the input unchanged (identity).
  - `VerifierStub.verify(claim, passages)` returns `Label.NEI` with confidence 0.5.
- **Pipeline:** linear orchestration; each step logged with `loguru` at INFO; per-claim trace dump optional via `--trace` flag.

## Implementation steps

1. Write `src/schema.py`. Run `pytest tests/test_schema.py`.
2. Write `src/config.py`. Verify `PipelineConfig.load("configs/default.yaml")` returns a frozen object.
3. Write the four stubs.
4. Write `src/pipeline.py` that takes a `PipelineConfig` and constructs each component.
5. Write `src/cli.py` (typer): `verify`, `eval`, `smoke` commands.
6. Write `tests/test_smoke.py` with 5 fixed claims (one each of: simple SUPPORTED, simple REFUTED, multi-hop, temporal, NEI-likely).
7. Replace `tests/test_smoke_placeholder.py` from Phase 00. `make smoke` now runs the real test.
8. Update CI to require `pytest -m smoke` to pass on every PR.

## Exit criteria

- [ ] `python -m src.cli verify "Christopher Nolan directed Inception."` runs end-to-end and prints an `EvidenceChain` (with stub verifier returning NEI).
- [ ] `pytest -m smoke` passes in <5 min on CPU.
- [ ] `pytest tests/test_schema.py` passes including the SubClaim `id` (NOT `sub_claim_id`) sanity check — explicit regression test for the V2 bug.
- [ ] `PipelineConfig` cannot be mutated after load (assignment raises).
- [ ] No module imports another module's *concrete* class directly except in `pipeline.py`. Modules consume interfaces.
- [ ] `loguru` JSON sink contains structured per-step entries for the smoke run.

## Risks and gotchas

- pydantic v2's `frozen=True` interacts oddly with nested dataclasses; verify with a `pytest.raises(ValidationError)` on assignment.
- Stubs must not lie. If a real retriever is supposed to be there but the index is missing, raise — don't return `[]`.
- Don't over-design `EvidenceChain` here. Phase 10 finalizes the renderer; this phase only ensures the data structure exists.

## What NOT to do

- Do not skip the explicit regression test for the V2 SubClaim bug. It's a one-line test and it documents intent.
- Do not introduce `MagicMock` in tests/test_smoke.py — it must run real components (real corpus encoder, real FAISS read). Phase 03+ swap stubs for real models without changing the test.
- Do not commit `configs/default.yaml` with real model paths yet — those land in their owning phases.

## Outcome (Phase 02 closed 2026-05-06)

**Wall time.** ~3 hours design + write + lint + verify (no model training; just schema + glue).

**Test results.**

- `pytest tests/test_schema.py`: **13 passed in 0.06 s** (no model loading). Includes the explicit V2 `SubClaim(sub_claim_id=...)` regression test that asserts the field is `id` and that the V2 keyword raises `TypeError`.
- `pytest -m smoke` on `tests/test_smoke.py`: **7 passed in 79.7 s** (target <5 min). 5 parametrized end-to-end claims + 1 retriever-quality regression (Inception article must appear in retrieved set) + 1 render-doesn't-crash check.
- `python -m src.cli verify "Christopher Nolan directed Inception."`: works in both human-readable and `--json` modes. Top-3 retrieved are Inception article, Tom Hardy (Nolan-film actor), The Dark Knight — illustrating the keyword-overlap pattern Phase 04's reranker is designed to fix.

**Schema module size.** `src/schema.py` is 159 LOC including docstrings (target was <300).

**Deviations from plan.**

- Phase doc described `src/retrieval/stub.py`. Renamed to `src/retrieval/dense.py` because Phase 02's dense retriever is the *real* Phase 04 component (using bge-small base + Phase 01 FAISS index). It is not a stub. Same logic for the file location — Phase 04 will iterate this file in place.
- `tests/test_smoke.py` is marked both `smoke` and `slow` (loads bge-small + 260 MB FAISS index). `make smoke` runs both unconditionally; `make test` (which excludes `slow`) skips it. This matches the docs/PHASE_00 contract.
- typer's `Argument`/`Option` pattern requires function calls in argument defaults; ruff `B008` warnings are silenced with explicit `# noqa: B008` rather than refactored. Standard practice in the typer ecosystem.

**Verified pipeline architecture.**

- `Pipeline` takes Protocol-typed components (`_Decomposer`, `_Retriever`, `_Reranker`, `_Verifier`) — no concrete-class imports. `build_pipeline(cfg)` does the wiring; future phases swap implementations without touching `Pipeline` itself.
- `PipelineConfig` is a frozen pydantic v2 BaseModel; mutation attempts raise `ValidationError`. Confirmed via spot test.
- All structures in `src/schema.py` are frozen dataclasses; sequence fields normalize list inputs to tuples in `__post_init__` for true immutability.

**Open follow-ups.**

- `tests/test_smoke_placeholder.py` was deleted — replaced by `tests/test_smoke.py`.
- Phase 03 will replace `StubDecomposer` with the Qwen2.5-3B-Q4 few-shot decomposer; the smoke test should continue to pass without modification.
- Phase 04 will replace `StubReranker` with the cross-encoder; same.
- Phase 07 will replace `StubVerifier` with LLM + NLI veto; same.
- Phase 10 will extend `EvidenceChain.render_text()` with the structured-chain renderer (citation indices, dependency paths) — current rendering is the Phase 02 baseline.
