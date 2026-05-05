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

## Outcome (filled at end of phase)

> Append: bucket distribution, top-3 patterns, Phase 09 gate decision (Path A or Path B), any taxonomy additions.
