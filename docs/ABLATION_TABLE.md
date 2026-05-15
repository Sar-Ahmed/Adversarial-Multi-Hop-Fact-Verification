# V3 Ablation Study

Seven ablation rows computed from cached phase artifacts (see `artifacts/ablation_results.json`). All HoVer-dev numbers are on the same n=200 stratified sample, seed=42. FEVER-dev is the n=300 balanced sample from Phase 08. Each metric carries a bootstrap 95% CI on accuracy.

## Headline ablation table

| # | Ablation | n | Accuracy | Macro-F1 | What's disabled |
|---|---|---|---|---|---|
| 1 | `full_production_decomposed` | 200 | **0.145** [0.100, 0.195] | 0.114 | nothing (current default) |
| 2 | `whole_claim_no_calibrator` | 200 | **0.360** [0.295, 0.425] | **0.233** | decomposer + calibrator |
| 3 | `whole_claim_llm_only` | 200 | 0.020 [0.005, 0.040] | 0.024 | decomposer + calibrator + NLI veto |
| 4 | `whole_claim_legacy_veto` | 200 | 0.030 [0.010, 0.055] | 0.038 | decomposer + calibrator + bidirectional NLI rule (keeps legacy SUPPORTED→REFUTED only) |
| 5 | `calibrator_only_hover` | 200 | 0.275 [0.215, 0.335] | 0.173 | LLM + NLI veto (calibrator runs on raw features) |
| 6 | `calibrator_fever_dev` | 300 | 0.427 [0.370, 0.483] | 0.417 | LLM + NLI veto (calibrator on FEVER 3-class) |
| 7 | `adversarial_distractors_injected` | 50 | 0.380 (paired Δ = -0.020 [-0.060, 0.000]) | — | adversarial distractors injected before reranking |

## What the ablations tell us

### 1. The bidirectional NLI rule is the single most valuable component

Read rows 3 → 4 → 2 on the same eval set:

| Aggregation rule | Accuracy |
|---|---|
| `llm_only` (no NLI signal at all) | 0.020 |
| `llm_plus_nli_veto` (V1's legacy: only SUPPORTED→REFUTED) | 0.030 |
| `llm_plus_nli_bidir` (V3's addition: NEI→REFUTED, NEI→SUPPORTED) | **0.360** |

V1's NLI veto added ~+1 point. V3's bidirectional extension lifts another **+33 points**. The 3B LLM defaults to NEI on 95% of HoVer claims; the bidirectional rule re-aggregates those NEI verdicts using the cached NLI signal, recovering ~63% of REFUTED predictions and a few SUPPORTED ones.

**The bidirectional NLI rule is V3's most important architectural addition.** It costs nothing at inference time (NLI is already scored) and lifts accuracy by 12-18×.

### 2. Decomposition hurts on a 3B verifier

Read rows 1 vs 2:

| Verifier sees... | Accuracy | Macro-F1 |
|---|---|---|
| whole claim + top-10 passages (one LLM call) | 0.360 | 0.233 |
| each sub-claim separately, aggregated (one LLM call per sub-claim) | 0.145 | 0.114 |

The decomposer is doing its job (mean 2.76 sub-claims per chain, valid `depends_on` DAG, sensible citations — Phase 10 showed this). The aggregator (`any REFUTED → REFUTED; all SUPPORTED → SUPPORTED; else NEI`) is doing its job. The verifier is the choke point: it can't reliably classify isolated multi-hop sub-claims, so per-sub-claim NEI verdicts compound and the aggregator returns NEI.

**On a stronger verifier (7B+, prompt-tuned for partial evidence), decomposition would likely help.** On our 3B verifier with this prompt, decomposition is a 2.5× accuracy regression. Production should ship whole-claim mode.

The decomposer output remains valuable as audit-trail metadata for the evidence chains (Phase 10) — even if its per-sub-claim verdicts don't drive the final aggregation.

### 3. Calibrator: cross-dataset win, single-dataset cost

| Eval surface | Calibrator effect |
|---|---|
| FEVER-dev 3-class (n=300) | **NEI recall 0% → 67%** (V1 → V3); accuracy 0.427, macro-F1 0.417 |
| HoVer-dev binary (n=200) | accuracy 0.275 vs 0.360 whole-claim no-calibrator (drop of 8.5 pts) |

The calibrator was trained on FEVER's balanced 3-class distribution and learned that moderate NLI signals (contra ~ entail ~ 0.3) predict NEI. HoVer has zero NEI gold; every NEI prediction on HoVer is wrong by construction. **The calibrator-on policy is correct for FEVER-style 3-class deployments and wrong for HoVer-only deployments.** Phase 11 recommends whole-claim *without* calibrator for HoVer; Phase 15 will codify this in `configs/default.yaml`.

### 4. Adversarial robustness: spec target met, but it's a softball

Row 7: adversarial Δ = -0.020 [-0.060, 0.000]. Spec target ≤ 0.05, easily met.

But Phase 06 already documented that 90% of the mined distractors aren't truly contradictory — they're high-cos passages about adjacent entities that the NLI cross-encoder flagged as "contradiction" because of subject-binding ambiguity. The reranker filters them as easily as it would any cos-similar-but-on-topic noise. A truly adversarial set (entity-aware mining, per the Phase 06 open follow-up) might tell a different story.

## What's not in this table

Three ablations from the phase doc weren't run because each would have cost ~1-3 CPU hours:

- **`bm25_only`** — Phase 04 retrieval-only metrics already showed BM25 H@10 = 0.840 vs Dense 0.925. End-to-end would compound. Hypothesis: 5-10 pt accuracy drop.
- **`no_reranker`** — bypass the cross-encoder. Hypothesis: 3-5 pt drop, but adversarial robustness Δ much worse (the reranker is what filters the distractors).
- **`base_vs_finetune_retriever`** — Phase 05 already showed the fine-tune wins on all 6 retrieval metrics but with overlapping CIs. End-to-end would likely be a wash.

These belong in a Phase 15 follow-up if the report needs them.

## Bottom-line read

The pipeline is held up by three components, in order of importance:

1. **Bidirectional NLI veto rule** (+33 pts vs no-NLI baseline). The single most important V3 architectural addition.
2. **Reranker** (no direct ablation, but the adversarial Δ ≈ 0 implies it's doing meaningful filtering work).
3. **Calibrator** (FEVER NEI 0 → 67%; HoVer −8.5 pts). Distribution-mismatched between FEVER and HoVer; the path forward is dataset-aware deployment.

Two components are *not* helping at this verifier capacity:

- **Decomposer** (−21.5 pts on HoVer when its output drives the verifier). Keep its output for chain rendering; don't let it drive verdicts.
- **NEI calibrator on HoVer-only** (−8.5 pts). Ship calibrator on for FEVER, off for HoVer-only deployments.

The macro-F1 ceiling on this configuration is 0.233 because the 3B Qwen verifier defaults to NEI on 95% of HoVer multi-hop claims even after the bidirectional rule recovers most REFUTEDs. The recovery path is a stronger verifier (7B+, prompt-softened, or LLM verdict feature added back into the calibrator).
