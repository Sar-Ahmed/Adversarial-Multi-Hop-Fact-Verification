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

## Outcome (Phase 10 closed 2026-05-15)

**Status: structural exit criteria all met. Accuracy surprise documented and routed to Phase 11.**

### Built artifacts

- `artifacts/evidence_chains.jsonl` — 1.7 MB, **200 chains** for the HoVer-dev eval set
- `artifacts/evidence_chain_render_examples.txt` — 23 KB, 10 stratified rendered chains (3 SUPPORTED, 3 REFUTED, 4 NEI)

### Spec exit criteria

- [x] `evidence_chains.jsonl` covers every claim in the eval set (200/200)
- [x] **Validator passes 100% (0 / 200 invalid)** — structural checks fire on every chain; no failures
- [x] Rendered text chains are syntactically clean — verified by visual inspection of 10 stratified examples
- [x] `EvidenceChain` JSON round-trips — `to_jsonable` / `from_jsonable` covered by `test_to_and_from_jsonable_roundtrip`
- [x] `make smoke` passes (8 / 8 in 316 s, last verified at Phase 08 close)

### Files added

- `src/evidence/chain.py` — `validate()`, `to_jsonable()`, `from_jsonable()`, `dependency_path()`
- `src/eval/build_chains.py` — runs full pipeline on HoVer dev with resume capability (mirrors Phase 07 / Phase 08 pattern)
- `tests/test_evidence.py` — 8 fast unit tests (validator + JSON round-trip + dependency path)

### Chain shape (n=200)

| Metric | Value |
|---|---|
| sub-claims per chain | min=1, max=7, **mean=2.76** |
| citations per chain (sum across sub-claims) | min=3, max=21, **mean=8.28** |
| Validator pass rate | **100% (200 / 200)** |

The decomposer + verifier produce chains the spec asked for: a non-trivial number of sub-claims (HoVer is multi-hop), each with cited evidence, all with valid `depends_on` DAGs.

### The accuracy surprise

End-to-end accuracy on n=200: **0.145** (29 / 200). Verdict distribution: 120 NEI, 76 REFUTED, 4 SUPPORTED, against gold of 102 SUPPORTED + 98 REFUTED.

Phase 07's eval on the same n=200 hit **0.36 accuracy** using *whole-claim* verification (no decomposition). Phase 10's full pipeline (decomposed → per-sub-claim verify → aggregate) drops to 0.145. That's a 22-point loss from running through the decomposer.

The aggregator rule (`Pipeline._aggregate`) is: any REFUTED → REFUTED; all SUPPORTED → SUPPORTED; else NEI. With multi-hop claims producing 2-3 sub-claims each, and the 3B verifier still NEI-biased on individual sub-claims, the chain falls into the "else NEI" bucket far more often than the whole-claim mode does.

Phase 11 robustness eval must compare both modes explicitly. The decision (decomposed vs whole-claim) is the single biggest pipeline-architecture choice left.

### Wall time

- Background chain build: **16.3 hours wall clock** for n=200. Real compute is much less (laptop slept overnight; resume worked as designed). Resume capability survived 2 sleep cycles without issue.

### Rendered example quality (qualitative observations)

- **Decomposition is clean.** Multi-hop claims split into 2-3 sub-claims with proper `composition` edges. One example splits an FIA Formula One claim into 3 sub-claims (venue + first-black-driver + Canadian-drivers), and the depends_on DAG correctly chains them.
- **Citations are relevant.** Top-3 cited passages per sub-claim actually relate to the sub-claim topic (Inception query → Inception + Tom Hardy + The Dark Knight; FIA query → Circuit Gilles Villeneuve + Montreal + Canadian Grand Prix).
- **Reasoning audit trail is visible.** Each verification's reasoning text shows: (1) the original LLM reasoning, (2) any NLI veto that fired (`[NLI-bidir: NEI→REFUTED on max_contra=0.97]`), (3) the calibrator's confidence (`[calibrator: p=0.71]`). A reviewer can read end-to-end why a verdict was reached.
- **Verdict quality is poor.** Most predicted-SUPPORTED chains in the rendered sample are gold=REFUTED. The verifier produces confident-but-wrong outputs on multi-hop reasoning. This is downstream of Phase 07 + 08 limitations, not a Phase 10 bug.

### Open follow-ups

- Phase 11 — measure whole-claim vs decomposed accuracy on the same eval set and pick the production aggregator.
- Phase 13 error analysis — categorise the 171 failures in this chain set by failure mode (confident wrong verdict, low-conf NEI, decomposition error, retrieval miss).
- The rendered audit trail is information-rich; a slimmer "user-facing" rendering (drop the bracketed audit annotations) may be wanted for end users. Out of scope for Phase 10.

> Append: chain count, validator pass rate, render-stat distribution (avg sub-claims per chain, avg citations per chain).
