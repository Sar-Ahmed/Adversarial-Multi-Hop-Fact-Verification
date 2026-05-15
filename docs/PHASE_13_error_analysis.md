# Phase 13 — Error Analysis (50 Stratified Failure Cases)

**Goal.** Stratified-sample 50 failures from the Phase 11 eval, tag each with a failure mode, and write a narrative analysis. Spec deliverable: "Error analysis document: 50 failure cases categorized by failure mode."

**Effort.** 1 day.
**Compute.** CPU; mostly manual annotation.
**Depends on.** Phase 11 (eval results with full per-example traces).

## Why this exists

The spec asks for it explicitly. More usefully: V1's error analysis revealed that 76% of failures were REFUTED→SUPPORTED, which is what motivated the NLI veto. V3's error analysis triggers Phase 09 (temporal) and informs the V3 final report.

## Inputs

- `artifacts/per_subclaim_traces.jsonl` from Phase 07.
- `artifacts/eval_main.json` from Phase 11 (per-example correct/incorrect flags).
- `artifacts/evidence_chains.jsonl` from Phase 10.

## Deliverables

- `src/analysis/categorize.py` — semi-automated tagger that proposes a failure category per error using rules + LLM-classifier (if available); the engineer reviews and overrides.
- `artifacts/failures_tagged.json` — 50 records with `claim`, `gold`, `pred`, `category`, `gold_in_top10`, `notes`.
- `docs/ERROR_ANALYSIS.md` — narrative writeup with patterns, examples, and which categories matter.
- The bucket counts feed [Phase 09](PHASE_09_temporal.md) gate (decision rule: `temporal_error ≥ 10%` triggers implementation).

## Failure-mode taxonomy

| Tag | Definition | Example |
|---|---|---|
| `refuted_as_supported` | Predicted SUPPORTED, gold REFUTED, gold passages in top-10 | "Bond actor born in Wales" matched to passage about a Welsh-born costume designer |
| `supported_as_refuted` | Predicted REFUTED, gold SUPPORTED, gold passages in top-10 | NLI veto fired on a partial-entailment passage that actually supports the claim |
| `nei_miscalibration` | Predicted NEI but gold is SUP/REF (or vice versa), passages were sufficient | Calibrator forced NEI on a low-confidence-but-correct verdict |
| `retrieval_miss` | Gold passages NOT in top-10 | The actual evidence wasn't retrieved |
| `entity_confusion` | Sub-tag of `refuted_as_supported` — wrong entity matched in passage | Claim about "Inception", passage about "Tenet" |
| `negation_blindness` | Sub-tag — model missed an explicit negation | Passage says "X did NOT win" but predicted SUPPORTED |
| `partial_match_as_full` | 2 of 3 facts confirmed, model treats as full SUPPORTED | "Born in Wales and Oscar winner" — born-in-Wales OK, Oscar is wrong, predicted SUPPORTED |
| `temporal_error` | Wrong year, wrong sequel, wrong era | "the 2008 film" matched to passage about the 2012 sequel |
| `decomposition_error` | Decomposer split wrongly (lost info or mis-attributed depends_on) | Compound claim split into independent sub-claims when it required dependency |
| `other` | Anything that doesn't fit above | Use sparingly; require a `notes` field |

## Sampling strategy

- Take all failures from `eval_main.json` (HoVer dev n=200, expect ~80–100 failures).
- Stratify by:
  - Predicted-vs-gold class transition (4 cells: SUP↔REF, SUP↔NEI, REF↔NEI, plus correct-class confusions if any).
  - Whether gold passages were in top-10 (retrieval-miss vs reasoning-miss).
- Sample 50 across the strata, prioritizing larger cells but ensuring ≥3 from each non-empty cell.

## Implementation steps

1. Implement `categorize.py` — heuristic tagger (regex + retrieval-coverage check) + a few-shot LLM tagger if Qwen 3B is available. Outputs proposed tags.
2. Sample 50 stratified failures.
3. Manual review of every proposed tag — engineer overrides where the heuristic is wrong. Save `notes` field for each.
4. Write `docs/ERROR_ANALYSIS.md`:
   - Top-line: bucket distribution, biggest single cause.
   - 5–10 representative examples in detail.
   - For each major bucket: root cause hypothesis + suggested next step.
5. Read the `temporal_error` count → decide [Phase 09](PHASE_09_temporal.md) gate.

## Exit criteria

- [ ] `artifacts/failures_tagged.json` has 50 records, all with a category and notes.
- [ ] `docs/ERROR_ANALYSIS.md` includes bucket counts as a table, plus narrative for the top 3 buckets.
- [ ] Phase 09 gate decision recorded in this doc's Outcome section AND in PHASE_09_temporal.md's outcome.
- [ ] No bucket exceeds 60% of failures unless we can name a single root cause and a fix path (otherwise we're not learning anything).

## Risks and gotchas

- Tagging is subjective at the margins; document edge-case decisions in `notes` rather than fighting over taxonomy purity.
- Heuristic tagger may be wrong on majority of cases; treat its output as a starting point, not authority.
- Avoid the "post-hoc rationalization" trap of inventing a fix the model "should" have done. The taxonomy is about *what went wrong*, not *what we should build next* (that goes in the narrative section).

## What NOT to do

- Do not tag automatically and skip review. Manual review is the whole point — it's where the team builds intuition.
- Do not categorize successes alongside failures. This phase is *about* failures.
- Do not skip the Phase 09 gate read — it's the trigger for the next phase.

## Outcome (Phase 13 closed 2026-05-15)

**Status: 50 failures tagged. Phase 09 gate decision: Path B (scope out temporal). Full per-failure tags in [`artifacts/failures_tagged.json`](../artifacts/failures_tagged.json); narrative in [`docs/ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md).**

### Bucket distribution

| Category | Count | % | Note |
|---|---|---|---|
| **nei_miscalibration** | **28** | **56%** | verifier punts on partial multi-hop signal — single biggest bucket |
| partial_match_as_full | 12 | 24% | claim has multiple facts; LLM/NLI weights the wrong one |
| entity_confusion | 10 | 20% | NLI flags adjacent-entities as contradiction |
| temporal_error | 0 | 0% | **Phase 09 gate: Path B** |
| retrieval_miss | 0 | 0% | sampling stratified by (gold, pred) — Phase 04 H@10 = 0.93 means most failures have gold retrieved |
| negation_blindness | 0 | 0% | no clear case in sample |
| decomposition_error | 0 | 0% | whole-claim mode bypasses decomposer |

The 56% nei_miscalibration bucket is approaching the doc's 60% cap, but it has a named root cause (3B verifier returns NEI on partial multi-hop signal when bidirectional NLI threshold of 0.95 isn't reached) and three named fix paths (prompt softening, 7B sweep, calibrator with LLM-verdict feature). Ships as documented.

### Phase 09 gate decision: Path B

Rule from `docs/PHASE_09_temporal.md`:
> If `temporal_error` is **≥ 10%** of failures: implement Phase 09 (Path A).
> If `temporal_error` is **< 10%**: scope out, document (Path B).

We measured **0 / 50 = 0%**. Path B. Temporal handling stays scoped out; `docs/SCOPED_OUT.md` will collect this decision for Phase 15's final report.

Caveat documented honestly: many of the failed claims *do* contain temporal expressions (#10 "1982 Bavarian Championships", #20 "Rihanna born 1948", #6 "Trần Ích Tắc lived in a city after failing"). The failures are not specifically *temporal-reasoning* failures — they're NEI-bias and entity-binding failures that happen to be on temporal claims. A working temporal extractor wouldn't fix them.

### Failure-vs-not-failure: where retrieval lands

100% of the sampled failures had at least one gold passage in the retrieved top-10 — confirming Phase 11's read that retrieval isn't the bottleneck. The verifier is.

(Note this is a sampling artifact: I stratified by `(gold, pred)` transitions. Phase 04 retrieval-only metrics already covered the retrieval-miss case at the recall level.)

### Files added

- `src/analysis/__init__.py`, `src/analysis/categorize.py` — stratified failure sampler with auto-computed gold_in_top10 and a tagging-template markdown emitter
- `artifacts/failures_for_tagging.md` — 50 blocks with empty `category:` fields (this is the artifact the human tagger fills in)
- `artifacts/failures_tagged.json` — structured tagged result; one row per failure with category + notes
- `docs/ERROR_ANALYSIS.md` — narrative breakdown with three representative examples per top bucket, fix paths, and Phase 15 forward recommendations

### Implications for Phase 15

The headline accuracy ceiling on this pipeline is the 3B Qwen verifier's NEI bias on partial multi-hop evidence. The bidirectional NLI rule (Phase 07 / Phase 12) recovered the easy cases (+33 pts); what's left is genuinely hard. Three named recovery paths are recorded in `docs/ERROR_ANALYSIS.md`:

1. **Prompt softening** — targets the 56% nei_miscalibration bucket directly.
2. **7B model swap** — targets nei_miscalibration + partial_match_as_full together.
3. **Entity-aware NLI gating** — targets the 20% entity_confusion bucket.

None of these are V3 work; they're recorded for whoever picks up the codebase.

> Append: bucket distribution, top-3 patterns, Phase 09 gate decision (Path A or Path B), any taxonomy additions.
