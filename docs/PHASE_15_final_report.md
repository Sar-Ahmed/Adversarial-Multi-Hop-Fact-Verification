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

## Outcome (filled at end of phase)

> Append: link to the tagged release, link to FINAL_REPORT.md, total project wall time, total LOC of `src/`.
