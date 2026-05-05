# Phase 06 — Adversarial Distractor Mining (cos≥0.85 ∧ NLI-contradicts)

**Goal.** For each evaluation claim, mine 5 distractor passages that satisfy *both* conditions in the spec: cosine similarity ≥ 0.85 to the claim, AND opposite semantic meaning (operationalized as NLI-contradicts probability ≥ 0.8). Spec requirement #2b. Fixes the most-explicit gap in V1 and the entire missing miner in V2.

**Effort.** 1 day.
**Compute.** CPU enough; Colab speeds NLI scoring ~5×.
**Depends on.** Phase 04 (encoder + corpus embeddings), Phase 02 (config).

## Why this exists

Both V1 and V2 only checked cosine similarity. The spec is explicit that distractors must have "opposite semantic meaning." Without this, "distractors" are just other-on-topic passages that confuse the model into wrong-but-not-adversarial errors. V3 fixes this with a two-stage filter.

## Inputs

- `artifacts/corpus_embeddings.npy` (or `_ft.npy` if Phase 05 shipped a fine-tune) — corpus embeddings.
- `artifacts/corpus.parquet` — for passage text lookup.
- `cross-encoder/nli-deberta-v3-base` — for stage-2 NLI scoring.
- HoVer dev claims (the eval set we'll inject distractors into).

## Deliverables

- `src/adversarial/mine.py` — `mine_distractors(claims, corpus_embeddings, nli_model, k=5, cos_threshold=0.85, nli_contra_threshold=0.8) -> dict[claim_id, list[distractor]]`.
- `src/adversarial/inject.py` — `inject(top_passages, distractors, mode="mix") -> list[Passage]` — mixes 5 distractors into the top-50 dense pool before reranking.
- `artifacts/distractors_v3.json` — committed mined distractors with metadata (cos sim, nli_contra prob, source corpus row, model versions).
- `artifacts/distractor_sanity_check.md` — manual notes on 20 randomly sampled (claim, distractor) pairs verifying that the NLI filter is doing what it claims.
- `tests/test_adversarial.py` — unit tests for the mining and injection logic.

## Technical approach

- **Stage 1 — cosine candidates.**
  - Encode the claim with the same encoder used for corpus indexing (bge-small base or fine-tune).
  - Dot product against `corpus_embeddings` (already L2-normalized) → cosine similarity.
  - Filter: drop the gold doc_ids for that claim; keep candidates with `cos ≥ 0.85`.
  - Take the top-200 candidates by cos sim, sorted descending.
- **Stage 2 — NLI contradiction filter.**
  - For each candidate passage, score `(claim, passage_text)` with `cross-encoder/nli-deberta-v3-base`. Output is a 3-way softmax over `(contradiction, entailment, neutral)`.
  - Keep candidates with `contradiction_prob ≥ 0.8`.
  - Take the top-5 by `cos × contradiction_prob` (multiplicative) so we get *both* lexically-similar AND semantically-opposed candidates.
- **Sanity check (mandatory).**
  - Random-sample 20 mined records.
  - For each: read claim and distractor side-by-side. Confirm distractor genuinely contradicts the claim or implies its negation. Note the failure rate in `distractor_sanity_check.md`.
  - If sanity-check failure rate > 25%, raise NLI threshold and re-run.
- **Injection (used only at eval time).**
  - `mode=mix`: inject the 5 distractors at random positions in the top-50 dense pool before reranking. Reranker may filter them; that's the point of the robustness eval.
  - `mode=replace_bottom`: replace the 5 lowest-scored dense passages with distractors (more aggressive).
  - `mode=replace_random`: random replacement.
  - Default `mix` for the headline robustness number; the others available for ablation.

## Implementation steps

1. Implement Stage 1 cos-sim filter; verify it returns plausible candidates for 5 hand-picked claims.
2. Implement Stage 2 NLI filter. Batch NLI scoring at batch_size=32 on CPU; ~1 s per batch.
3. Run mining on the full HoVer dev (~4k claims) — budget 30–60 min on CPU, less on Colab.
4. Save `distractors_v3.json` with: claim_id, distractor records (doc_id, title, text, cos, contra_prob, neutral_prob, entail_prob), threshold metadata.
5. Random-sample 20 records; write `distractor_sanity_check.md` with manual ratings.
6. Implement `inject.py` with three modes; unit-test on toy passage lists.
7. Wire into `pipeline.py` behind a config flag `eval.adversarial_mode: clean | adversarial`.
8. Smoke test in adversarial mode — pipeline must still complete.

## Exit criteria

- [ ] `artifacts/distractors_v3.json` covers all HoVer dev claims, with ≥4 of the requested 5 distractors per claim (some claims may have fewer high-cos+high-contra candidates; document rate).
- [ ] Manual sanity check on 20 random distractors shows ≤25% failure rate (distractor truly does contradict the claim).
- [ ] Injection function preserves passage type and ordering invariants (verified by unit test).
- [ ] `make smoke` passes in both `clean` and `adversarial` modes.

## Risks and gotchas

- For some claims, no candidates pass both filters (especially highly specific claims). Pad to 5 by relaxing the contradiction threshold to 0.5 *with a tag in the JSON marking those records as low-confidence* — never silently substitute non-adversarial passages.
- NLI cross-encoders read claim/passage *as a pair* — the order matters. Verify with `cross-encoder/nli-deberta-v3-base`'s docs which order yields the contradiction logit.
- Stage 1's cos≥0.85 threshold is strict; on focused 200k corpora some claims have <10 candidates. Lower to 0.80 if the per-claim candidate count drops below 20 for >30% of claims.
- Don't include the gold passages themselves in the candidate pool. Filter aggressively by gold doc_id and gold title.

## What NOT to do

- Do not use the LLM (Qwen 3B) for the contradiction filter. NLI cross-encoders are 10–20× faster and discriminatively trained for this exact task.
- Do not mine distractors at inference time. Mine once, save to JSON, inject deterministically at eval. This is also a reproducibility property.
- Do not skip the manual sanity check. The whole point of this phase is that V1's "cos-only" approach was technically wrong; we must verify our fix works.

## Outcome (filled at end of phase)

> Append: per-claim distractor count distribution, sanity-check failure rate, NLI threshold finally used, any low-confidence padding rate.
