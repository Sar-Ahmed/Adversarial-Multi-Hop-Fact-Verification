# Phase 15 — Final Report and Handoff

**Goal.** Produce the final evaluation report, handoff documentation, and an honest list of known limitations. Spec deliverable: "Evaluation report: F1, accuracy, robustness score, evidence chain quality."

**Effort.** 0.5 day.
**Compute.** None.
**Depends on.** Phases 11, 12, 13, 14 (every metric and analysis ready to cite).

## Why this exists

The point of the report is *traceability*. A reviewer should be able to pick any number in the report, follow it back to the artifact JSON it came from, follow that to the script that produced it, and re-run the script to reproduce the number. V1's `FINAL_REPORT.md` is a good template; V3 follows it with stricter requirements around CIs.

## Inputs

- All `artifacts/*.json` from prior phases.
- `docs/ERROR_ANALYSIS.md` from Phase 13.
- `docs/ABLATION_TABLE.md` from Phase 12.
- `artifacts/human_eval_summary.json` from Phase 14.
- Outcome notes appended to each phase doc in this folder.

## Deliverables

- `docs/FINAL_REPORT.md` — the report.
- `docs/SETUP_AND_REPRODUCE.md` — instructions to reproduce every reported number from a clean checkout.
- `docs/SCOPED_OUT.md` — explicit scope-out items (e.g., temporal if Phase 09 chose Path B).
- `docs/HANDOFF.md` — a short "where to look first" doc for the next engineer (or auditor).
- A tagged release / final commit on the `main` branch.

## Required sections of FINAL_REPORT.md

1. **TL;DR** — one paragraph: headline accuracy, macro-F1, robustness score, all with CIs and n.
2. **Pipeline overview** — diagram (lifted from `ARCHITECTURE.md`), three sentences per component.
3. **Headline numbers** — table with: HoVer dev acc, HoVer dev macro-F1, FEVER dev NEI recall, robustness delta, evidence chain overall quality. Every cell `point ± CI (n=...)`.
4. **Ablation results** — embed Phase 12 table verbatim. Annotate which components proved load-bearing.
5. **Error analysis** — embed Phase 13's bucket distribution table + 3 representative examples.
6. **Evidence chain quality** — embed Phase 14's per-dimension table.
7. **Adversarial robustness** — explain the test (cos≥0.85 + NLI-contradicts injection), report the delta with CI, comparison to spec target (<5%).
8. **Known limitations** — each on a separate line:
   - REFUTED-detection bottleneck (3B verifier).
   - NEI calibration (the gain we got, the residual gap).
   - Temporal handling (whatever Path 09 decided).
   - Eval set size (n=200 default; n=4k full-run if it was done).
9. **Honest negative results** — list every experiment that failed and why we kept the artifact (e.g., a fine-tune that lost). This section is mandatory; do not skip.
10. **Reproducibility notes** — Python version, model checksums, seeds, git SHA.

## Implementation steps

1. Read every phase doc's "Outcome" section.
2. Copy headline numbers into the report from the JSON artifacts (don't re-derive — cite the JSON path).
3. Write the report.
4. Cross-link to phase docs and artifacts.
5. Write `SETUP_AND_REPRODUCE.md` with a single command per major artifact (e.g., "to reproduce eval_main.json: `make eval-main`").
6. Tag the release.

## Exit criteria

- [ ] Every numerical claim in `FINAL_REPORT.md` has an `n=` and a 95% CI.
- [ ] Every numerical claim has a footnote / link pointing to the JSON artifact it came from.
- [ ] `SETUP_AND_REPRODUCE.md` lets a fresh checkout reproduce the headline numbers in <8 hours of compute (or notes the Colab steps required).
- [ ] `SCOPED_OUT.md` exists and is referenced from the report's "Known limitations" section.
- [ ] `HANDOFF.md` exists with "first 30 minutes" instructions.
- [ ] No `PHASE_X_COMPLETE.md` victory-lap files have been created. (This rule applies to the entire V3 repo.)
- [ ] Final commit tagged.

## Risks and gotchas

- The temptation to round CIs ("60% ± 7%" → "60%") for readability is strong. Resist. The CI is the headline.
- Temptation to compare to V1/V2 numbers in the report. Do so *only* if the eval setup is identical (same eval split, same n, same seed). If not, omit — apples-to-apples comparison is more important than self-promotion.
- Don't bury bad news. If something failed, the report's "Known limitations" and "Honest negative results" sections name it.

## What NOT to do

- Do not let this phase expand. 0.5 day is the budget. If you find yourself running new experiments to "make the numbers better", you've left the phase. New experiments belong in V4 planning, not V3 final report.
- Do not commit a draft report with TODOs. Every section is filled in or explicitly marked as "out of scope".
- Do not bury the limitations section at the bottom in 8pt font. Limitations get equal billing with results.

## Outcome (Phase 15 closed 2026-05-15)

**Status: all four Phase 15 deliverables shipped. V3 tagged `v3.0`.**

### Files added

- [`docs/FINAL_REPORT.md`](FINAL_REPORT.md) — the headline report. Every numerical claim has `n=` and a 95% CI; every CI cites the JSON artifact it came from. Sections: TL;DR, pipeline overview, headline numbers, ablation results, error analysis, evidence chain quality, adversarial robustness, known limitations (9 items), honest negative results (5 items — Phase 05 fine-tune, Phase 06 distractors, Phase 07 3B accuracy, Phase 08 calibrator HoVer regression, Phase 11 decomposed mode), reproducibility notes, spec deliverable map.
- [`docs/SETUP_AND_REPRODUCE.md`](SETUP_AND_REPRODUCE.md) — clone-to-headline in 12 steps, ~30 min minimal vs ~8 h full reproduction. Each artifact gets a single command.
- [`docs/HANDOFF.md`](HANDOFF.md) — 30-minute onboarding for the next engineer: reading order, the one thing to know (verifier is the bottleneck), three fix paths with effort/lift, landmines that look scary but aren't.
- [`docs/SCOPED_OUT.md`](SCOPED_OUT.md) — was already at 77 lines; Phase 15 added the Phase 14 n=50/single-rater section so the deviation is in one place.

### Headline numbers reaffirmed in the report

- HoVer dev accuracy: **0.360 [0.295, 0.425]** (n=200; whole-claim mode, NLI bidir veto, no calibrator)
- Macro-F1: 0.233; REFUTED F1 0.53; SUPPORTED F1 0.17
- FEVER NEI recall: **0.670** (vs V1 baseline 0.000) — V3's largest single win on a spec metric
- Adversarial Δ: **−0.020 [−0.060, 0.000]** (n=50 paired) — spec target ≤0.05 ✓
- Phase 14 chain ratings: decomp 4.16, citations 4.46, reasoning 2.58, faithfulness 2.58, overall 2.56
- Phase 13 bucket distribution: 28/50 nei_miscalibration (verifier), 12/50 partial_match, 10/50 entity_confusion, 0/50 temporal/retrieval/decomp/negation

### Spec exit criteria

- [x] Every numerical claim in `FINAL_REPORT.md` has an `n=` and a 95% CI
- [x] Every numerical claim links to the JSON artifact it came from
- [x] `SETUP_AND_REPRODUCE.md` lets a fresh checkout reproduce the headline numbers within ~8 hours of compute
- [x] `SCOPED_OUT.md` exists and is referenced from `FINAL_REPORT.md` § Known limitations
- [x] `HANDOFF.md` exists with "first 30 minutes" instructions
- [x] No `PHASE_X_COMPLETE.md` victory-lap files were created (rule held across the whole V3 repo)
- [x] Final commit tagged `v3.0`

### Project totals

- **`src/` LOC.** 6,670 across 16 phases.
- **Phases.** 00–15 (this doc closes 15).
- **Wall time.** 7 days (2026-05-09 → 2026-05-15).
- **Compute envelope.** Single laptop CPU + ~25 h Colab T4 GPU (Phase 05 fine-tune, optional).
- **Tag.** `v3.0` on master.
