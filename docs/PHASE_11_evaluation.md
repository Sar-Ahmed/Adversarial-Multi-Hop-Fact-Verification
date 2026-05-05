# Phase 11 — Evaluation Framework

**Goal.** Compute macro-F1, accuracy, per-class metrics, and adversarial robustness with bootstrap 95% CIs on every reported number. Spec requirements: macro-F1, label accuracy, robustness score.

**Effort.** 1 day.
**Compute.** CPU (running pipeline over n=200 to n=1000 claims; mostly already cached if Phase 07/08 ran).
**Depends on.** Phase 10 (full pipeline output).

## Why this exists

V1 reported point estimates on n=200 with no CIs. On 200 examples, ±3% is easy noise. V3 makes confidence intervals mandatory and runs both clean and adversarial conditions for the robustness score.

## Inputs

- HoVer dev (full ~4k or n=200 fast subset; both supported via config).
- FEVER dev (for the cross-dataset NEI signal).
- `artifacts/distractors_v3.json` from Phase 06.
- Built `EvidenceChain`s from Phase 10.

## Deliverables

- `src/eval/metrics.py` — accuracy, macro/per-class precision-recall-F1, confusion matrix, with `bootstrap_ci(samples, metric_fn, n=1000) -> (point, lo, hi)`.
- `src/eval/run_eval.py` — runs full pipeline in `clean` mode, then `adversarial` mode, on the configured eval split. Saves `artifacts/eval_main.json` and `artifacts/eval_adversarial.json`.
- `src/eval/robustness.py` — computes Adversarial Robustness Score = `acc_clean - acc_adversarial` with CI on the *delta* (paired bootstrap).
- `artifacts/eval_main.json` — clean-mode results.
- `artifacts/eval_adversarial.json` — adversarial-mode results.
- `artifacts/robustness_eval.json` — paired delta with CI.
- `artifacts/per_class_breakdown.json` — per-class precision-recall-F1 with CIs.

## Technical approach

- **Bootstrap.** Sample example indices with replacement n=1000 times, recompute metric, take 2.5%/97.5% percentiles for the 95% CI. Use a fixed seed.
- **Paired bootstrap (for robustness delta).** Each example contributes a `(clean_correct, adv_correct)` pair; bootstrap-resample paired indices and compute the delta inside each resample.
- **Per-class metrics.** sklearn `classification_report` for the point estimates; bootstrap our own CIs for each class's precision / recall / F1.
- **Eval splits.**
  - Default: HoVer dev n=200 stratified subset (fast, matches V1 baseline).
  - Validation: HoVer dev full ~4k (run once per major change; `make eval-full`).
  - Cross-dataset: FEVER dev n=300 stratified by class (100 SUPPORTED, 100 REFUTED, 100 NEI) — the only place we measure NEI recall on real NEI labels.

## Implementation steps

1. Implement `metrics.py` with bootstrap and paired-bootstrap helpers; unit-test against trivial cases.
2. Implement `run_eval.py` to execute the pipeline in clean and adversarial modes back-to-back on the same examples (so the paired bootstrap is valid).
3. Implement `robustness.py`.
4. Run on n=200 HoVer dev (clean + adv). Save JSONs.
5. Run on FEVER dev n=300. Save JSON. (NEI metrics live here.)
6. Optional: full HoVer dev n=4k overnight or on Colab.

## Exit criteria

- [ ] All headline numbers in the project carry a 95% CI. No bare point estimates accepted in any artifact.
- [ ] `acc_clean - acc_adversarial` ≤ 0.05 (spec target; use paired-bootstrap CI to be sure).
- [ ] HoVer dev macro-F1 ≥ 0.40 (V1 hit ~0.39; we want to match or exceed).
- [ ] FEVER dev NEI recall ≥ 0.40 (Phase 08 calibrator should achieve this; this is the validation).
- [ ] All eval JSONs include: model fingerprint hashes, git SHA, seed, n, CI bounds.

## Risks and gotchas

- Paired bootstrap for the robustness delta requires that the same examples are evaluated in both conditions. Make sure `run_eval.py` runs them in lockstep, not in two unrelated passes.
- Adversarial evaluation is more expensive than clean (NLI scoring on injected distractors); budget time accordingly.
- Reporting sample sizes: if anything is reported on n<100, flag it (the CIs will be wide and qualitative claims should be muted).

## What NOT to do

- Do not report a "the system gets 60%" without n and CI. Always: "60% ± 7%, n=200, 95% CI".
- Do not subset eval based on what's "easy" — that's eval gaming. Stratification by gold class is fine; cherry-picking is not.
- Do not run only one of clean/adversarial. Both, paired, every time.

## Outcome (filled at end of phase)

> Append: macro-F1 / accuracy with CI for clean and adversarial, robustness delta with CI, per-class F1, FEVER NEI recall with CI.
