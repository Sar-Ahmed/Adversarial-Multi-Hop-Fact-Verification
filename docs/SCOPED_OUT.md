# V3 scoped-out items

A single catalogue of capabilities that V3 explicitly does not ship, with each decision's data link. Referenced from `docs/FINAL_REPORT.md`.

## Temporal reasoning (Phase 09)

**Scoped out** on 2026-05-15.

**Why:** Phase 13's error analysis (50 stratified failures from production whole-claim mode) measured 0 / 50 = 0% temporal_error failures. The phase doc's binding rule says scope out under 10%.

**Caveat:** Many failing HoVer claims do contain temporal expressions ("the 1982 Bavarian Championships", "the 2012 sequel"). The failures aren't *temporal-reasoning* failures — they're NEI-bias and entity-binding failures on claims that happen to mention time. A temporal extractor wouldn't fix them.

**How to revisit:** if a future iteration runs `python -m src.analysis.categorize` and finds temporal_error ≥ 10%, re-trigger Phase 09 Path A. The gate isn't a one-shot.

**Data source:** [artifacts/failures_tagged.json](../artifacts/failures_tagged.json) — bucket distribution at `summary.bucket_distribution`.

## Entity-aware adversarial distractor mining (Phase 06 follow-up)

**Documented gap, not run.**

**Why:** Phase 06's NLI-only mining produces "weakly adversarial" distractors — 18 / 20 manual sanity-check failures (90%). NLI cross-encoders treat different-entity-with-similar-attribute pairs as contradictions, so the distractors are on-topic-not-contradictory rather than strict semantic opposites. The Phase 06 sanity-check exit criterion (≤ 25% fail) is not met.

**Fix path on the table:** entity-aware filtering — spaCy NER on claim, require distractor passage to mention at least one claim entity, then NLI-filter from there. Estimated half-day implementation.

**Why not done in V3:** Phase 11's adversarial Δ on the current distractors was -0.020 (no degradation). The current distractors don't bite the reranker, so the entity-aware refinement is gated on whether harder distractors would actually expose a reranker weakness.

## LLM verdict as calibrator feature (Phase 08 follow-up)

**Documented gap, not run.**

**Why:** Phase 08 spec called for LLM verdict one-hot in the calibrator's feature set. Computing LLM verdict on ~600 FEVER train examples would cost ~5 h of CPU LLM inference. Phase 08 dropped it for compute reasons; uses NLI + retrieval + lexical features only.

**Expected lift if added:** ~3–5 points of FEVER macro-F1 (current 0.417, target 0.45) and possibly recovers the HoVer accuracy drop (current −8.5 pts).

**Fix path:** rerun `src.calibration.build_features` on FEVER + HoVer with LLM verdict added to the feature vector, retrain.

## Soft-prompt fix on the 3B Qwen verifier (Phase 16 — tried and scoped out)

**Tried on 2026-05-15. Negative result. Full write-up in [`docs/PHASE_16_soft_prompt.md`](PHASE_16_soft_prompt.md).**

**What I tried:** added a second prompt variant `SYSTEM_PROMPT_V2` that replaces the strict "if on-topic but doesn't address claim → NEI" rule with "if any part of the claim is supported or contradicted, lean toward a verdict; NEI only when evidence is off-topic." Wired a `--prompt-variant` flag through `verifier_eval.py`, ran a fast n=5 sanity that re-uses the v3.0 cached passages.

**What happened:** 5/5 cases identical NEI verdicts (the v3.0 v1 baseline was also 5/5 NEI). Pass criterion "v2 NEI rate < v1 NEI rate" failed cleanly. Worse, the v2 *reasoning text* on those same 5 falsely claims entities are not mentioned when they are — anchoring on the new "off-topic only" few-shot example degrades faithfulness even when the verdict doesn't change. Examples in the Phase 16 outcome doc.

**Why it failed:** the 3B model is too anchored on the NEI-on-uncertain heuristic for prompt rewording alone to dislodge. The bidir NLI rule (Phase 12, +33 pts) already captures the recoverable NEI→{SUPPORTED,REFUTED} signal at aggregation time. What remains is model-capacity-limited.

**Infrastructure kept:** `SYSTEM_PROMPT_V2`, `LLMVerifier(prompt_variant=...)`, the `--prompt-variant` CLI flag, `softprompt_sanity.py`, `paired_compare.py`. All committed for future prompt-variant experiments. A future iteration that wants to try another wording can just add `SYSTEM_PROMPT_V3` and re-run the sanity.

**Where the effort goes:** the 7B verifier sweep (next section). Path A's negative result strengthens the case for Path B.

## 7B verifier sweep (Phase 07 follow-up)

**Documented gap, not run.**

**Why:** Phase 07's headline accuracy (0.36 on HoVer, 0.42 on FEVER) is below the spec target (0.55) and below trivial baselines (always-SUPPORTED = 0.51). The phase doc anticipated this: *"only invoked if the CPU 3B baseline shows <55% accuracy after the NLI veto."* We're at 36% — this trigger fires. Not run because compute budget already absorbed Phase 05's 25 h, Phase 07's ~3 h, and Phase 08's 12 h.

**Fix path:** `notebooks/phase07_verifier_sweep.ipynb` (stub committed at Phase 07 part 1). Use Qwen 2.5-7B-Instruct via vLLM on Colab T4/A100. ~2-3 h.

## Three expensive ablations (Phase 12 follow-up)

**Listed but not run.**

- `bm25_only` — replace dense retriever with BM25. ~1-2 h end-to-end.
- `no_reranker` — bypass the cross-encoder. ~30 min end-to-end.
- `base_vs_finetune_retriever` — Phase 05 already showed retrieval-CIs overlap; end-to-end ablation would likely confirm a wash.

Why not run: Phase 12 already produced 7 ablation rows from cached artifacts. These three would each need a new pipeline run. Listed in `ablation_results.json.summary.open_follow_ups_not_run`.

## Switching production default to whole-claim mode

**Documented gap, not executed.**

**Why:** Phase 11 and Phase 12 both concluded whole-claim mode beats decomposed mode by 12-21 points on this 3B verifier. `Pipeline.verify` still runs decomposed mode by default because the decomposer's output is needed for the Phase 10 audit-trail chains.

**Fix path:** one-line change in `src/pipeline.py` to bypass the decomposer loop and run verifier once on the whole claim. Keep the decomposer's output as metadata in the chain's `sub_claims` field for the Phase 10 audit trail. Phase 15 final-report decision.

## Phase 14 human eval — n=50 not 100, single rater not two

**Documented deviation, not a future-work item.**

**Why n=50:** A single rater at ~2-3 min/chain is ~4 hours for n=100; the bootstrap 95% CIs are already ±0.18 wide on each dimension at n=50, so the marginal information from the second 50 is small. Rationale stated in `docs/PHASE_14_human_eval.md` outcome section.

**Why single rater:** Spec suggested a second rater on a 20-chain subset for Cohen's-kappa inter-rater agreement. Not done; only one developer on the project. Mitigation: `docs/HUMAN_EVAL_PROTOCOL.md` defines the rubric with anchored examples so a second rater could be slotted in later and the protocol is reproducible.

**Why this matters:** the spec deliverable is partially met (50/100 chains, 1/2 raters). The cross-correlation results (reasoning r=0.24, faithfulness r=0.50, overall r=0.56 with verdict-correctness) are still informative and converge with Phase 11/12/13's evidence that the verifier is the bottleneck.

**Fix path:** rope in a second engineer for ~1 hour on the existing 50 chains for a kappa estimate, or expand to n=100 with the same rater. Both are additive — the existing 50 ratings need not be redone.

## Larger eval set

**Documented gap.**

V3 evaluates on HoVer-dev n=200 (stratified) and FEVER-dev n=300 (balanced). Phase 11 ran adversarial robustness on n=50 paired. Full HoVer-dev is ~4,000 examples; the CIs would tighten by ~4× at full sample. Not run because each n=200 pass already takes 1-16 h on CPU.

**Fix path:** all eval scripts already have `--n` flags and resume capability. Re-run with `--n 0` for the full sample on a machine with overnight access.
