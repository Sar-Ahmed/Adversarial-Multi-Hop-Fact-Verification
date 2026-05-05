# Phase 12 — Ablation Study

**Goal.** Measure each major component's contribution by running the pipeline with it disabled. Spec deliverable: "Ablation study: performance with/without each component."

**Effort.** 1–2 days (most compute is just re-running eval with config flags).
**Compute.** CPU. Same eval set as Phase 11.
**Depends on.** Phase 11 (eval framework with CIs).

## Why this exists

The spec asks for it; more importantly, an ablation is the only way to defend the architecture. V1's ablation showed that the reranker and NLI veto carried real weight while per-sub-claim CoT did not — that data drove the production config.

## Inputs

- The full v3 pipeline at end of Phase 10/11.
- HoVer dev eval set, n=200 (paired with Phase 11 for comparability).

## Deliverables

- `src/eval/run_ablations.py` — runs each ablation config in turn, calls the same `run_eval.py` machinery.
- `configs/ablations/*.yaml` — one config file per ablation, derived from `default.yaml` by overriding one section.
- `artifacts/ablation_results.json` — table with one row per ablation: name, accuracy + CI, macro-F1 + CI, delta-from-full + paired-CI.
- `docs/ABLATION_TABLE.md` — narrative table for inclusion in the final report.

## Ablation matrix (the rows)

| # | Name | Disabled component | Hypothesis |
|---|---|---|---|
| 1 | `full` | (nothing — baseline) | The production config |
| 2 | `bm25_only` | dense retriever (use BM25 instead) | Dense > BM25 by ~3 R@10 points (V1 confirmed) |
| 3 | `no_reranker` | cross-encoder reranker | Reranker filters cos-sim distractors and is the adversarial-robustness lever |
| 4 | `base_retriever` | Phase 05 fine-tune (use base bge-small) | Validates Phase 05 decision |
| 5 | `no_nli_veto` | NLI veto in aggregate.py | NLI veto is the main REFUTED-detection mechanism (V1 measured ~+4–6 points) |
| 6 | `no_calibrator` | Phase 08 NEI calibrator | Calibrator drives FEVER NEI recall from 0% → 0.40 |
| 7 | `no_distractors` | adversarial distractors at eval (clean only) | Sanity: clean ≥ adversarial; tells us the gap |
| 8 | `decomposer_off` | Single-sub-claim mode (whole claim treated as one) | Tests whether decomposition helps at all on HoVer |
| 9 | `temporal_off` (if Phase 09 implemented) | Temporal extractor + features | Validates Phase 09 work |

(Rows 8 and 9 are conditional on whether Phase 03 and Phase 09 produced shippable components.)

## Technical approach

- One YAML override per ablation in `configs/ablations/`. Each is `default.yaml` with the relevant section disabled or replaced.
- `run_ablations.py` iterates the configs, calls `run_eval.py` for each, captures output JSONs, builds a single combined `ablation_results.json`.
- Use *paired* bootstrap deltas (same eval examples across all ablations) so the delta-from-full CIs are tight.

## Implementation steps

1. Write the ablation YAMLs (mostly stripping or stubbing single keys).
2. Implement `run_ablations.py` (looping orchestrator).
3. Run all 7–9 ablations on n=200 HoVer dev. Budget: ~3–6 hours total CPU.
4. Build the combined JSON + the markdown table.

## Exit criteria

- [ ] `artifacts/ablation_results.json` contains every ablation row with its CI.
- [ ] Each ablation's delta-from-full has a paired 95% CI.
- [ ] `docs/ABLATION_TABLE.md` is committed and ready for inclusion in the final report.
- [ ] At least one ablation shows a *negative* delta (component-on > component-off) with non-overlapping CIs — proving that *some* component is doing real work. (If none does, the architecture is suspect.)

## Risks and gotchas

- An ablation that *improves* the pipeline (component-off > component-full) is a strong signal that component is misconfigured or actually harmful. Investigate before publishing.
- Some ablations break the smoke test (e.g., `decomposer_off` may need a special path). Ensure `run_ablations.py` doesn't fail-fast on the first error — it should run all ablations and report failures separately.
- Don't compare ablations from different eval splits or dates — must be the same n=200 set.

## What NOT to do

- Do not run ablations on the training data. HoVer dev only.
- Do not skip the paired-CI computation. Differences without CIs are not deltas.
- Do not write a narrative interpretation in the JSON. The JSON is data; interpretation goes in `docs/ABLATION_TABLE.md`.

## Outcome (filled at end of phase)

> Append: full table with each ablation's accuracy + CI + delta + paired-CI. Note any surprises (e.g., a component that turned out not to matter).
