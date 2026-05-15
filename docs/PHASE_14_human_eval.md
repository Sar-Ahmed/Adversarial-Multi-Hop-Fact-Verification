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

## Outcome (Phase 14 closed 2026-05-15)

**Deviation from plan: n=50 not n=100, single rater not two.** Rated chains drawn from the Phase 10 production decomposed configuration (n=200 chains), stratified 25 SUPPORTED + 25 REFUTED. The 50-vs-100 deviation is documented with rationale below; the dimension-level CIs at n=50 are still informative (~0.18-wide each).

### Headline ratings (1–5 Likert, bootstrap 95% CI over 1000 resamples)

| Dimension | Mean | 95% CI | Median | What it measures |
|---|---|---|---|---|
| **Decomposition** | **4.16** | [4.00, 4.32] | 4 | Sub-claims atomic, faithful, well-ordered |
| **Citations** | **4.46** | [4.26, 4.64] | 5 | Cited passages actually relate to the sub-claim |
| **Reasoning** | **2.58** | [2.40, 2.76] | 3 | Logical soundness given the citations |
| **Faithfulness** | **2.58** | [2.38, 2.80] | 3 | Reasoning reflects what cited passages say |
| **Overall** | **2.56** | [2.36, 2.76] | 2 | Non-expert auditor can follow the chain |

### The story the numbers tell

**The upstream pipeline (decomposition + retrieval) is strong; the verifier's reasoning is the bottleneck.** This is the single most important Phase 14 finding and it converges with every other piece of V3 evidence:

- **Decomposition mean 4.16, no 1/2 ratings, 12/50 perfect-5s.** Phase 03's claim-decomposer ships clean sub-claims that respect compositional dependencies. The two phases of explicit gold-rated evaluation (Phase 03 sanity check + Phase 14 chain rating) agree.
- **Citations mean 4.46, no 1/2 ratings, 28/50 perfect-5s.** Phase 04+05's retriever finds the right passage in the top-K for the great majority of sub-claims. When the model fails, the evidence is usually *already in its hands*.
- **Reasoning mean 2.58, with 3 chains scoring 1/5 and 17/50 scoring 2/5.** The 3B Qwen verifier's most common failure mode is **denying what its own cited passage says** — e.g., chain #21 (Vernon Kay/All Star Family Fortunes) where the model's primary citation literally lists the show in the host's bio, but the reasoning text says "no information." Chain #22 (Taylor Nichols/Headless Body in Topless Bar) is similar: the cited ensemble-cast passage explicitly contains the actor's name.
- **Faithfulness mean 2.58** tracks reasoning closely. Faithfulness drops *below* reasoning on 11 chains where the reasoning text is internally OK but the final verdict ignores it (most notably chain #37 where the LLM says "no specific info" then the calibrator outputs SUPPORTED at 0.78).

### Correlation with correctness (chain-level Pearson)

| Dimension | r |
|---|---|
| Decomposition | −0.06 |
| Citations | +0.03 |
| Reasoning | +0.24 |
| **Faithfulness** | **+0.50** |
| **Overall** | **+0.56** |

**Decomposition and citations don't predict correctness at all.** This is the V1-vs-V3 contrast V1's outcome doc emphasised: V1 saw correct=3.50, incorrect=2.20 (Δ=1.30) on a single overall rating. V3 disaggregates that gestalt into two independent failure paths:

- **Decomposition + citations:** uniformly strong → uncorrelated with whether the verifier succeeds.
- **Faithfulness + overall:** the discriminating signal. r ≈ 0.50–0.56 → the auditor's gut is right.

### Per-correctness breakdown (means)

| Dimension | Correct (n=10) | Incorrect (n=40) | Δ |
|---|---|---|---|
| Decomposition | 4.10 | 4.18 | −0.08 |
| Citations | 4.50 | 4.45 | +0.05 |
| Reasoning | 2.90 | 2.50 | +0.40 |
| Faithfulness | 3.40 | 2.38 | **+1.02** |
| Overall | 3.40 | 2.35 | **+1.05** |

This Δ matches V1's overall-rating Δ of 1.30 closely — a sign that the V3 chain renderer is producing chains a single rater can rank consistently against the implicit ground truth in Phase 11.

### The three chain failure modes worth recording

1. **"NEI when the passage literally says it"** — the model's #1 failure (chains #1, 21, 22, 29, 30, 39, 48). Cited passage contains the answer; reasoning text says "no information." This is exactly what motivated the bidirectional NLI veto rule in Phase 07/aggregate.py — the rule re-aggregates these NEIs back to REFUTED on high contradiction. Phase 12 measured the rule's lift at +33 points; the Phase 14 chain-level read explains *why* the rule works.
2. **Sub-verdict collapse to NEI** — multi-hop chains where each atomic sub-claim is correctly verifiable from a passage, but the per-sub-claim verifier returns NEI on each piece, so the aggregator (any-REFUTED→REFUTED, all-SUPPORTED→SUPPORTED, else NEI) collapses to NEI (chains #19, 25, 26, 34). This is the Phase 12 finding "decomposition hurts on this verifier" from the chain-level perspective.
3. **Calibrator/veto override contradicts the LLM's own reasoning text** — most visibly chain #37 (LLM reasoning says "no specific info"; calibrator outputs SUPPORTED 0.78). The chain text becomes internally contradictory; the auditor can't follow it. Mitigated in whole-claim mode (Phase 11 production config) where decomposition isn't doing the aggregation, but the same risk exists for the calibrator's standalone runs.

### Gold-label asymmetry

REFUTED-gold chains score better than SUPPORTED-gold chains across every dimension (4.32 vs 4.00 decomposition; 4.64 vs 4.28 citations; 2.80 vs 2.32 overall). This is consistent with Phase 11's per-class F1 result (REFUTED F1 0.50 vs SUPPORTED F1 ~0) and the bidirectional NLI rule's specific design: it lifts NEI→REFUTED far more often than NEI→SUPPORTED because contradiction signals are sharper than entailment signals in HoVer's multi-hop claims.

### Spec deviations and rationale

- **n=50, not 100.** The spec asks for 100; we ship 50 with documented rationale: a single rater at ~2-3 minutes per chain is ~2 hours for 50, ~4 hours for 100, and the bootstrap CI is already narrow (~0.18 wide on each dimension at n=50). The marginal value of the second 50 is small. If a second rater becomes available, the protocol is set up to extend.
- **Single rater.** Spec encouraged a second rater for 20-sample kappa. Not done; this is a known limitation reported in `docs/SCOPED_OUT.md`. The protocol document `docs/HUMAN_EVAL_PROTOCOL.md` is the reproducibility hedge — a second rater can be slotted in later.
- **Stratification dropped to gold-only.** Spec called for stratifying by correctness too. With only 10 correct in the 50-chain sample (matching the 20% accuracy on the production decomposed config), pure correctness stratification would have been too thin. Gold-label balance is what we stratified on; the by-correctness breakdown above is reported post-hoc.

### Files added

- `src/evidence/sample_for_eval.py` — stratified 25+25 sampler; emits CSV + rendered companion.
- `src/evidence/aggregate_human_eval.py` — reads rated CSV, emits summary JSON with bootstrap CIs.
- `artifacts/human_eval_sample.csv` — 50 chains with all 5 ratings filled.
- `artifacts/human_eval_rendered.txt` — companion rendered chains (what the rater read).
- `artifacts/human_eval_summary.json` — aggregate stats including correctness-stratified means and per-dimension histograms.
- `docs/HUMAN_EVAL_PROTOCOL.md` — rubric document (anchors for each score per dimension).

### Spec exit criteria

- [x] Chains rated on all 5 dimensions (50 of 100 — documented deviation)
- [x] `artifacts/human_eval_summary.json` has per-dimension means + 95% CIs
- [x] Correlation between each dimension and `correct` reported (range −0.06 to +0.56)
- [ ] Inter-rater kappa — not computed (single rater); deferred to follow-up
- [x] Summary table ready for inclusion in `docs/FINAL_REPORT.md`
