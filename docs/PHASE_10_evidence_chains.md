# Phase 10 — Evidence Chain Generation

**Goal.** For every verdict, produce a structured `EvidenceChain` (sub-claim → cited passage IDs → reasoning → final verdict) that is machine-checkable and human-auditable. Spec requirement #4.

**Effort.** 1 day.
**Compute.** CPU (no model calls beyond what Phase 07 already produced).
**Depends on.** Phase 07 (verifier outputs), Phase 08 (calibrated final verdict).

## Why this exists

The spec asks for a step-by-step chain showing which passages support which sub-claims, and that chain must be human-readable and auditable. V1 produced 200 chains with 100% citation validity and a 3.04/5 human quality rating; V3 ports that design forward and adds machine-checkable structural validation.

This phase has no model — it's deterministic post-processing of the per-(claim, sub-claim) traces from Phase 07.

## Inputs

- `artifacts/per_subclaim_traces.jsonl` from Phase 07.
- Calibrated verdicts from Phase 08.
- HoVer dev claims.

## Deliverables

- `src/evidence/chain.py` — `build_chain(claim, decomposition, verifications, calibrated_verdict) -> EvidenceChain`.
- `src/evidence/render.py` — `render_text(chain, max_passage_chars=200) -> str` and `render_json(chain) -> dict`.
- `src/evidence/validate.py` — structural checks: every cited passage_id maps to a real corpus row; every sub-claim has at least one citation OR an explicit no-evidence flag; depends_on dag is acyclic.
- `artifacts/evidence_chains.jsonl` — full HoVer dev chains (one JSON per line).
- `artifacts/evidence_chain_render_examples.txt` — 10 rendered text chains across SUPPORTED / REFUTED / NEI for the final report.
- `tests/test_evidence_chain.py` — round-trip serialization, validator tests, render output not empty.

## Technical approach

- **Build.**
  - For each sub-claim from decomposer:
    - Look up `SubClaimVerification` from Phase 07.
    - Cited passages = top-3 by reranker score among the top-10 retrieved (the verifier's own attention is captured implicitly via its reasoning text; we don't ask the LLM to emit citations because parse-failure rate goes up).
    - Reasoning text = the LLM's `reasoning` field from its JSON output, plus a one-line summary of the NLI veto if one fired.
  - Final verdict = calibrator output from Phase 08.
  - Final confidence = calibrator max-prob.
- **Render (text).**
  - Header: claim, final verdict, confidence.
  - For each sub-claim in dependency order:
    - "Sub-claim N: <text>"
    - "Verdict: <SUPPORTED/REFUTED/NEI> (confidence X.XX)"
    - "Cited: [#a] <title> — <text excerpt up to 200 chars>"
    - "Reasoning: <one-paragraph reasoning>"
  - Footer: "Aggregated final verdict: ..." with the rule that fired (e.g., "any-REFUTED → REFUTED" or "all-SUPPORTED → SUPPORTED" or "calibrator-NEI → NEI").
- **Render (JSON).** Mirror the dataclass exactly, with passage IDs and excerpts preserved.
- **Validate.**
  - 100% of citations must resolve to corpus rows.
  - 100% of `depends_on` must reference earlier sub-claim IDs.
  - 100% of sub-claims must have either `cited_passage_ids` or `cited_passage_ids = []` with `verdict=NEI` and `reasoning` containing "no evidence".

## Implementation steps

1. Implement `chain.py` `build_chain`.
2. Implement `validate.py` structural checks.
3. Implement `render.py` text + JSON renderers.
4. Run over full HoVer dev → `artifacts/evidence_chains.jsonl`.
5. Stratified-sample 10 chains (SUP/REF/NEI balanced) → `evidence_chain_render_examples.txt`.
6. Wire into pipeline: every `verify()` call returns a fully-built `EvidenceChain`.
7. Smoke test verifies the chain.

## Exit criteria

- [ ] `artifacts/evidence_chains.jsonl` covers every HoVer dev claim.
- [ ] Validator passes 100% (assert in tests).
- [ ] Rendered text chains are syntactically clean (no truncation artifacts, no broken passage indices) — verified by spot-checking the 10 examples.
- [ ] `EvidenceChain` JSON round-trips: load → serialize → load returns equal object.
- [ ] `make smoke` passes and produces an `EvidenceChain` per claim.

## Risks and gotchas

- Passage excerpts in renders truncate at character boundaries; truncate at word boundaries instead and append "..." to avoid mid-word cuts.
- If the LLM's reasoning text is noisy or incomplete, the renderer still must succeed — render whatever string is there, don't crash.
- For NEI verdicts, citations are often empty by design. Don't fail validation on that — accept the explicit `cited=[]` + `reasoning≈"no evidence"` pattern.

## What NOT to do

- Do not ask the LLM to emit citation indices in its JSON. V1 noted that doing so increases parse failures. We derive citations from the rerank order instead.
- Do not generate citations *post-hoc* by calling another LLM ("which passage supports this verdict"). That's a separate design and is out of scope for V3.
- Do not embed the full passage text in the JSON chain; embed `passage_id` references and look them up at render time. Keeps JSON files small.

## Outcome (filled at end of phase)

> Append: chain count, validator pass rate, render-stat distribution (avg sub-claims per chain, avg citations per chain).
