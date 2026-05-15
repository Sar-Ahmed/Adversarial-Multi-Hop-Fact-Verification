# V3 Handoff — First 30 Minutes

You are the next engineer or auditor on this repo. This doc gets you oriented in 30 minutes.

## Read in this order

1. **[`FINAL_REPORT.md`](FINAL_REPORT.md)** (10 min) — what shipped, what didn't, with every number cited.
2. **[`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md)** § "Three fix paths" (5 min) — where the next iteration's lift lives.
3. **[`ABLATION_TABLE.md`](ABLATION_TABLE.md)** § "Component-attribution summary" (5 min) — what does and does not carry weight.
4. **[`SCOPED_OUT.md`](SCOPED_OUT.md)** (5 min) — what V3 intentionally does not ship, with reasons.
5. **[`README.md`](README.md)** (5 min) — phase index with one-line outcome per phase.

If you only have 5 minutes, read just `FINAL_REPORT.md` § "TL;DR" and § "Known limitations".

## The one thing you need to know

**The verifier is the bottleneck.** Every piece of independent evidence converges:

- Phase 11: 0.36 accuracy on HoVer dev. Spec target 0.55.
- Phase 12: bidirectional NLI rule alone contributes +33 pts. The rule exists *because* the 3B verifier defaults to NEI on 95% of multi-hop claims.
- Phase 13: 28/50 failures are `nei_miscalibration` (model returns NEI despite passage containing the answer).
- Phase 14: reasoning rated 2.58/5; correlation with verdict-correctness is r=0.50 for faithfulness. Decomposition (4.16/5) and citations (4.46/5) are strong and r ≈ 0.

Retrieval, decomposition, and reranking are not the problem. The 3B verifier is.

## The highest-leverage next experiment

From [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md) and [`SCOPED_OUT.md`](SCOPED_OUT.md):

| Path | Effort | Expected lift | Status | Where to start |
|---|---|---|---|---|
| ~~A. Soft-prompt the 3B verifier~~ | ~1.5 h | +5–8 pts | **tried, scope-out 2026-05-15** | [`docs/PHASE_16_soft_prompt.md`](PHASE_16_soft_prompt.md) — sanity n=5 showed 0/5 verdict shifts; reasoning got worse. Bottleneck is model capacity, not prompt wording. |
| **B. 7B verifier sweep** ← **start here** | ~3 h Colab | +10–15 pts | unstarted; Path A failure strengthens this | [`notebooks/phase07_verifier_sweep.ipynb`](../notebooks/phase07_verifier_sweep.ipynb) stub — Qwen 2.5-7B-Instruct via vLLM |
| **C. Entity-aware retrieval** | ~half day | +2–4 pts (entity_confusion bucket) | unstarted | [`src/retrieval/dense.py`](../src/retrieval/dense.py) — add spaCy NER, require distractors to share at least one claim entity |

Path A was the cheapest test of the bottleneck hypothesis and it ruled out prompt rewording as the fix. Path B is now the highest-leverage next experiment. Path A infrastructure (`SYSTEM_PROMPT_V2`, `--prompt-variant` flag, `softprompt_sanity.py`, `paired_compare.py`) is committed and reusable if a future iteration wants to try yet another prompt variant against a 7B model — the variant slot is just a string.

## Where the numbers live

Every artifact in [`artifacts/`](../artifacts/) is text and committed. Headline mapping:

| Want | File |
|---|---|
| HoVer dev headline accuracy | `eval_main.json` → `hover_dev_whole_claim.accuracy` |
| FEVER NEI recall | `calibration_eval.json` → `results.fever_dev.per_class.NEI` |
| Adversarial Δ | `robustness_eval.json` → `paired_bootstrap.delta` |
| 7 ablation rows | `ablation_results.json` |
| 50-failure taxonomy | `failures_tagged.json` → `summary.bucket_distribution` |
| 50 human-rated chains | `human_eval_summary.json` + `human_eval_sample.csv` |
| Per-example LLM traces | `per_subclaim_traces.jsonl` (one row per sub-claim, with cached NLI signals) |

## Where the code lives

Top-level structure:

```
src/
  decomposer/        — Phase 03: Qwen 3B few-shot decomposition
  retrieval/         — Phase 01/04/05: BM25 + dense bge-small + fine-tune
  reranker/          — Phase 04: BAAI cross-encoder reranker
  verifier/          — Phase 07: LLM + NLI ensemble + aggregator with bidir rule
  calibration/       — Phase 08: logistic regression NEI calibrator
  evidence/          — Phase 10/14: chain construction + human eval sampler
  adversarial/       — Phase 06: distractor mining
  analysis/          — Phase 13: failure categorization
  eval/              — Phase 11/12: run_eval, robustness, ablation
  pipeline.py        — top-level orchestration
  schema.py          — frozen dataclasses (Passage, SubClaim, EvidenceChain)
  config.py          — pydantic PipelineConfig (frozen)
```

The pipeline entry point is `src.pipeline.Pipeline.verify`. Two aggregation modes live in `src/verifier/aggregate.py`: `llm_plus_nli_bidir` (production) and the legacy `llm_plus_nli_veto` (kept for ablation row 4).

## Things that look like landmines but aren't

- **`src/temporal/__init__.py` is empty.** This is deliberate — Phase 09 scoped out temporal because Phase 13 measured `temporal_error = 0/50`. The empty file is the documented placeholder.
- **`Pipeline.verify` still defaults to decomposed mode** even though Phase 11/12 showed whole-claim wins. The decomposer's output is used for the Phase 10 audit-trail chains, so it's still computed; the one-line flip is in `SCOPED_OUT.md`'s "Switching production default" section.
- **The calibrator is off by default in `configs/default.yaml`.** Phase 08 found it helps FEVER 3-class but hurts HoVer-only. Turn it on with a config override, not code change.
- **`checkpoints/bge-small-v3-hn`** is the Phase 05 fine-tune. Its CIs overlap base bge-small — base ships. The fine-tune is retained for ablation traceability.
- **Test markers.** `pytest -m smoke` runs the 8-test smoke set in ~5s. `pytest -m "not slow"` is the regular CI suite. `pytest -m slow` runs the integration tests against the real LLM (~30s).

## Things to NOT do

- **Do not add a `temporal/extract.py` to "improve coverage"** without first re-running [`src.analysis.categorize`](../src/analysis/categorize.py) and confirming the `temporal_error` bucket is ≥10%. The gate in [`PHASE_09_temporal.md`](PHASE_09_temporal.md) is binding.
- **Do not amend the production calibrator config to always-on.** It hurts HoVer accuracy by 8.5 pts (Phase 12 row 5 vs row 2). The dataset-conditional shipping config is the right answer.
- **Do not delete the `bge-small-v3-hn` fine-tune** even though it doesn't ship. Phase 13's `entity_confusion` bucket suggests a future iteration's fix may need entity-aware retrieval, and the hard-negatives JSONL is the most expensive thing to rebuild.
- **Do not skip CIs** in any new metric you report. Phase 15 spec says every numerical claim needs an `n=` and a 95% CI. The bootstrap helper is in [`src/eval/metrics.py`](../src/eval/metrics.py).
- **Do not write `PHASE_X_COMPLETE.md` victory-lap docs.** The phase doc itself is the contract; outcomes go in the same file's `## Outcome` section.

## If you want to test the pipeline end-to-end on your machine

After install (steps 1-3 of [`SETUP_AND_REPRODUCE.md`](SETUP_AND_REPRODUCE.md)):

```bash
make smoke                                   # ~5 s install verification
make corpus                                  # ~25 min corpus + FAISS
python -m src.eval.run_eval --n 20           # ~3 min headline reproduction (using cached traces)
cat artifacts/eval_main.json | python -m json.tool | head -30
```

## Open questions for the next iteration

1. Does Path A (soft-prompt) recover the NEI miscalibration bucket? Cheapest test of the bottleneck.
2. Does Path B (7B verifier) actually beat 3B by the predicted 10–15 pts, or does the calibrator's design-point shift?
3. If Path B works, does decomposition become net-positive again on the stronger verifier? (Currently −21.5 pts at 3B.)
4. Is the calibrator's dataset-conditional behavior fixable by adding LLM verdict as a feature (Phase 08 deferred work)?

If any of those tests fire, update [`FINAL_REPORT.md`](FINAL_REPORT.md)'s "Known limitations" section and re-tag the release.

## How this repo was built

Phase-driven plan-then-execute over 16 phases (00–15), each with a binding exit criteria checklist. Outcome notes appended to each phase doc. Bootstrap CIs on every metric. Every artifact regenerable from a single command. The full plan was written first (`PHASE_00_setup.md` through `PHASE_15_final_report.md`) and only then executed; the plan is the contract.

If you find yourself wanting to deviate from a phase doc, **update the phase doc first**, then deviate. The doc is the source of truth.
