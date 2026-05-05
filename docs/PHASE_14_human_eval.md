# Phase 14 — Evidence Chain Human Evaluation

**Goal.** Stratified-sample 100 evidence chains and rate them on 5 dimensions to satisfy the spec's "Evidence chain quality via human evaluation (sample 100 examples)" requirement.

**Effort.** 1 day.
**Compute.** Manual.
**Depends on.** Phase 10 (chains exist), Phase 11 (per-example correct/incorrect flags for stratification).

## Why this exists

Spec deliverable. V1 ran a similar 28-sample human eval (smaller than spec's 100) and learned the most useful insight in the project: chain quality correlates with verdict correctness (3.50/5 for correct, 2.20/5 for incorrect predictions). V3 hits the spec target of 100 samples and reports per-rater statistics.

## Inputs

- `artifacts/evidence_chains.jsonl` from Phase 10.
- `artifacts/eval_main.json` from Phase 11 (for stratification by correctness).

## Deliverables

- `src/evidence/sample_for_eval.py` — produces `artifacts/human_eval_sample_v3.csv` with 100 chains for rating.
- `docs/HUMAN_EVAL_PROTOCOL.md` — rubric document defining the 5 dimensions and 1–5 scale.
- `artifacts/evidence_human_eval_v3.csv` — completed ratings (after manual eval).
- `artifacts/human_eval_summary.json` — aggregate scores with bootstrap CIs, plus correlation between dimensions and correctness.

## Sampling strategy

- 100 chains stratified by:
  - Predicted verdict (35 SUPPORTED, 35 REFUTED, 30 NEI, adjusted to actual class distribution).
  - Correctness (50 correctly-classified, 50 incorrectly-classified).
- Each chain shown to the rater with claim, sub-claims, cited passages (full text), and final verdict — but **without** the gold label, to avoid bias.

## Rating dimensions (each 1–5)

| Dimension | Definition |
|---|---|
| **Decomposition quality** | Are the sub-claims atomic, faithful to the original claim, and well-ordered? |
| **Citation correctness** | Do the cited passages actually relate to the sub-claim being verified? |
| **Reasoning quality** | Is the reasoning text logically sound given the citations? |
| **Faithfulness** | Does the reasoning text reflect what the cited passages actually say (no hallucinations)? |
| **Overall auditability** | Could a non-expert auditor follow the chain and reach the same verdict? |

5 = excellent / no issues; 3 = mixed / partially correct; 1 = wrong / unfollowable.

## Implementation steps

1. Write `HUMAN_EVAL_PROTOCOL.md` with the rubric, examples for each score, and a calibration test (3 known-good chains + 3 known-bad to anchor the rater).
2. Sample 100 chains; export to CSV with one row per chain and one column per rating dimension.
3. (Optional) recruit a second rater for a subset (e.g., 20 chains) to compute inter-rater agreement (Cohen's kappa or weighted-kappa).
4. Manual rating — budget ~3–5 hours.
5. Compute aggregate scores with CIs; cross-tab against correctness from Phase 11.

## Exit criteria

- [ ] 100 chains rated on all 5 dimensions.
- [ ] `artifacts/human_eval_summary.json` has per-dimension means + 95% CIs.
- [ ] Correlation between each dimension and `correct` (boolean) reported.
- [ ] If a second rater participated: inter-rater kappa reported.
- [ ] Summary table ready for inclusion in `docs/FINAL_REPORT.md`.

## Risks and gotchas

- Single-rater eval is subjective. Be transparent about it. If possible, rope in a second engineer for ≥20 samples to anchor agreement.
- "Faithfulness" and "Reasoning quality" are correlated but distinct — protocol document must call out the difference (faithfulness = no hallucinations vs reasoning quality = sound argument).
- Rater fatigue is real; rate in two batches with a break.

## What NOT to do

- Do not show the rater the gold label. They rate the chain on its own merits.
- Do not aggregate-and-publish without CIs. n=100 with subjective scores has meaningful variance.
- Do not skip the protocol document. The protocol is what makes the rating reproducible.

## Outcome (filled at end of phase)

> Append: per-dimension mean ± CI, correctness correlation, inter-rater kappa (if computed), top 3 chain failure modes from the rater's perspective.
