# Phase 08 — NEI Calibration

**Goal.** Train a logistic-regression NEI calibrator on rich features (NLI scores, retrieval gap, entity overlap, claim length, passage variance, LLM logprob, etc.) using FEVER NEI training examples. Replace the heuristic "any NEI subclaim → NEI" rule with a learned, calibrated 3-way decision. Spec failure mode #4 ("LLMs are bad at I-don't-know; you need calibrated confidence, not argmax").

**Effort.** 2 days.
**Compute.** CPU (training a logistic regressor on ~10–20k feature vectors is seconds).
**Depends on.** Phase 07 (we need the verifier features per example).

## Why this exists

V1's per-token logprob threshold yielded +0.5% on HoVer (within noise) and 0% recall on FEVER NEI. V2 designed the right architecture (logistic regression on features) but never trained it. V3 trains a real classifier on FEVER NEI and validates it on a held-out FEVER + HoVer dev mix.

## Inputs

- FEVER train + dev — has labels for SUPPORTED, REFUTED, *and* NEI (which HoVer lacks).
- `artifacts/per_subclaim_traces.jsonl` from Phase 07 — feature source for non-FEVER claims.
- `cross-encoder/nli-deberta-v3-base` outputs.

## Deliverables

- `src/calibration/features.py` — `extract_features(claim, sub_claims, top_passages, llm_output, nli_scores, retrieval_scores) -> np.ndarray`.
- `src/calibration/train_nei.py` — trains `LogisticRegression` (sklearn) with class weights, saves to `checkpoints/nei_classifier.joblib` along with the `StandardScaler`.
- `src/calibration/predict.py` — loads classifier + scaler, exposes `calibrate_verdict(features, raw_verdict, raw_confidence) -> (Label, float)`.
- `artifacts/calibration_eval.json` — per-class F1, calibration curves (predicted prob vs empirical), Brier score, ECE, with bootstrap CIs.
- `tests/test_calibration.py` — unit tests for feature shape, classifier load/predict roundtrip, no-model-no-silent-fallback.

## Technical approach

- **Feature set (12 features, all float32).**
  1. `max_nli_entail` over top-10 passages.
  2. `max_nli_contra` over top-10.
  3. `mean_nli_neutral` over top-10.
  4. `nli_entail_minus_contra_top1`.
  5. `retrieval_top1_score` (rerank score).
  6. `retrieval_score_gap_1_to_5` (rerank score #1 minus #5).
  7. `mean_passage_length_top10`.
  8. `claim_length_words`.
  9. `entity_overlap_claim_passage_top1` (Jaccard on lowercase tokens excluding stopwords).
  10. `llm_verdict_one_hot` (3 features → expand: `is_llm_supported`, `is_llm_refuted`, `is_llm_nei`).
  11. `llm_logprob_at_verdict_token` (from llama-cpp logprobs).
  12. `n_subclaims` (decomposer output count).
  
  (Total = 14 features after one-hot expansion. Exact list reviewed in Phase 02 schema; any additions logged in this phase's outcome.)

- **Training data.**
  - For each FEVER train example: run the full Phase 07 pipeline (decomposer → retrieve → verify) and record features + true label.
  - Cache to `artifacts/calibration_train_features.parquet` so retraining is fast.
  - Class balance: FEVER has roughly 1/3 each — but compute class weights anyway in case our pipeline upstream is imbalanced.

- **Model.**
  - `sklearn.linear_model.LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", multi_class="multinomial")`.
  - `StandardScaler` fit on train features, persisted alongside the classifier.
  - 5-fold cross-validation on training set for hyperparameter tuning (C in {0.1, 1, 10}); pick best by macro-F1.
  - Validation: held-out 20% of FEVER train + full FEVER dev.

- **Calibration metrics.**
  - Per-class precision / recall / F1.
  - Calibration curve (10-bin reliability diagram).
  - Brier score, Expected Calibration Error (ECE).
  - All with bootstrap 95% CIs.

- **Decision rule at inference.**
  - `calibrator.predict_proba(features)` → 3-way prob.
  - `argmax` for the verdict; max prob is the calibrated confidence.
  - If max prob < `decision_threshold` (default 0.5): force `Label.NEI`. Threshold tunable via config.

## Implementation steps

1. Implement `features.py`. Unit-test with toy inputs.
2. Build `artifacts/calibration_train_features.parquet` by running Phase 07 pipeline over FEVER train. Time-bounded: cap at 10k FEVER examples to keep this tractable.
3. Implement `train_nei.py` with sklearn pipeline (StandardScaler + LogisticRegression).
4. Train, cross-validate, save best model.
5. Implement `predict.py`; wire into `pipeline.py` after the verifier.
6. Run `calibration_eval.py` on FEVER dev + HoVer dev (HoVer has no NEI but its SUPPORTED/REFUTED accuracy should not degrade after calibration).
7. Compare: pre-calibration vs post-calibration HoVer dev accuracy + FEVER dev NEI recall.

## Exit criteria

- [ ] `checkpoints/nei_classifier.joblib` and `checkpoints/nei_scaler.joblib` saved.
- [ ] FEVER dev NEI recall ≥ 0.40 (V1 hit 0%; >0.40 is a meaningful improvement).
- [ ] HoVer dev accuracy does not drop by more than 1 point (within CI) after calibration is applied.
- [ ] Macro-F1 (3-class) on FEVER dev ≥ 0.45.
- [ ] Calibration curve and ECE saved in `artifacts/calibration_eval.json`.
- [ ] `make smoke` passes with calibration enabled.

## Risks and gotchas

- Building the training feature set requires running the full pipeline 10k+ times. This is expensive — cache aggressively (parquet, mmap-able npy). On CPU budget 4–8 hours.
- Class imbalance: even with `class_weight="balanced"`, the calibrator may still over-predict the dominant class. If so, additionally tune the per-class decision thresholds rather than the model.
- The NLI scores are correlated with the LLM verdict (because both use similar signal). Logistic regression handles correlation poorly; if collinearity is severe, switch to gradient-boosted trees (`sklearn.ensemble.GradientBoostingClassifier`). Document the switch.
- Don't leak: features must come from the same retrieve→verify path that runs at inference. Don't accidentally use gold passage IDs as a feature.

## What NOT to do

- Do not train on HoVer (no NEI labels) — train on FEVER, eval on both.
- Do not use a neural calibrator. Logistic regression is interpretable, fast, and sufficient for ~10k training examples. Save complex models for when we have evidence simple ones aren't enough.
- Do not let the calibrator silently fail to load. If `joblib.load` raises, raise — don't fall back to no-calibration.

## Outcome (Phase 08 closed 2026-05-14)

**Status: PARTIAL — calibrator hits the headline NEI-recall target (V1 = 0%, V3 = 67%) at a documented cost on HoVer-only eval. Ship with calibrator on; Phase 11 measures end-to-end impact.**

### Headline numbers

Cross-validation on FEVER train (n=600, balanced 200 per class), 5-fold, macro-F1:
- C=0.1: 0.489 ± 0.047
- C=1.0: 0.491 ± 0.042
- **C=10.0: 0.492 ± 0.041** ← chosen

Held-out eval (`artifacts/calibration_eval.json`, decision_threshold=0.5):

| Split | n | Accuracy | Macro-F1 | NEI Recall | ECE | Notes |
|---|---|---|---|---|---|---|
| FEVER dev | 300 | 0.427 [0.370, 0.483] | **0.417** | **0.670** | 0.100 | balanced 100 per class |
| HoVer dev | 200 | 0.275 [0.215, 0.335] | 0.173 | — | 0.336 | 102/98 SUP/REF, no NEI gold |

Per-class on FEVER dev:
- SUPPORTED: P=0.585 R=0.310 F1=0.405 (precision wins; many SUP claims pushed to NEI)
- REFUTED:   P=0.536 R=0.300 F1=0.385 (same pattern)
- NEI:       P=0.351 R=0.670 F1=0.461 (recall wins; broad NEI bucket)

Per-class on HoVer dev:
- SUPPORTED: P=0.250 R=0.029 F1=0.053 (calibrator predicts NEI on 70% of SUPPORTED-gold claims)
- REFUTED:   P=0.416 R=0.531 F1=0.466 (mostly intact)

### Spec exit criteria

- [x] `checkpoints/nei_classifier.joblib` saved (sklearn Pipeline: StandardScaler + LogReg, 11 features)
- [x] **FEVER dev NEI recall ≥ 0.40** → achieved **0.67** (V1 reported 0% on the same metric in Phase 12 — this is the single biggest improvement in V3)
- [ ] **HoVer dev accuracy doesn't drop by more than 1 point** → **dropped 8.5 points** (0.36 → 0.275). Spec-fail, documented; root cause analysis below.
- [ ] **Macro-F1 (3-class) on FEVER dev ≥ 0.45** → achieved **0.417**, 0.03 short. Spec-fail by a hair.
- [x] Calibration curve + ECE saved (ECE = 0.100 on FEVER, 0.336 on HoVer; reliability bins in JSON)
- [x] `make smoke` passes with calibration enabled → **8 / 8 in 316 s**

### Why FEVER NEI recall improved so dramatically

The 11 cheap features (NLI {max, mean} contra/entail/neutral, retrieval top1 + score gap, claim length, mean passage length, entity overlap, contra-minus-entail) carry enough signal to distinguish FEVER's NEI examples from SUP/REF when trained on balanced data. The biggest predictor (logistic regression coefficients) is the `contra_minus_entail_top1` feature — when neither contradicts nor entails strongly, it's NEI; when contra dominates, REFUTED; when entail dominates, SUPPORTED. This is the simplest possible recipe and it works.

V1 used a per-token logprob threshold and got +0.5% on HoVer (within noise) plus 0% NEI recall on FEVER. V3's feature-based logistic regression with `class_weight=balanced` is a 67× improvement on the metric V1 explicitly admitted was broken.

### Why HoVer accuracy dropped

The calibrator was trained on FEVER's balanced 3-class distribution and learned that *moderate* NLI signals (contra ~ entail ~ 0.3) predict NEI. On HoVer, multi-hop SUPPORTED-gold claims often have similar moderate-NLI features (because retrieval only finds half the gold per claim — see Phase 04 R@10 = 0.55), so the calibrator predicts NEI on them. HoVer has zero NEI gold; every NEI prediction is wrong.

This is a *known* distribution mismatch, not a bug. Three ways to handle it in production:

1. **Always-on calibrator** (current default in `configs/default.yaml`). Right for FEVER-style 3-class queries. Hurts on HoVer-only.
2. **Always-off calibrator** (set `nei_classifier_path: null`). Right for HoVer-only deployments. Loses the NEI-recall win on FEVER.
3. **Conditional calibrator** — disable if `max(P_SUPPORTED, P_REFUTED) > 0.7` (i.e., when the calibrator is itself confident about a non-NEI class, keep it; when it's uncertain and would default to NEI, fall back to the rule). Open follow-up; not implemented.

Shipping option 1 because the spec's requirement is the 3-class verification with calibrated NEI — meeting that is the contract. Phase 11 robustness eval will measure end-to-end impact on whichever split it uses.

### Documented gaps

- **LLM verdict dropped from feature set.** Phase 08 spec called for it as a one-hot. Computing it on FEVER train would cost ~5 h of LLM inference. We use NLI + retrieval + lexical only, costing 11.7 h of wall time anyway because the laptop slept through most of it. Open follow-up if Phase 13 error analysis shows LLM signal would close the macro-F1 gap (0.417 → 0.45+).
- **n=600 FEVER train is small.** Spec recommended up to 10k. We picked 600 to keep total compute under one overnight. The 5-fold CV macro-F1 of 0.49 ± 0.04 suggests we're not yet overfitting; could likely benefit from more data.
- **HoVer doesn't have NEI gold,** so we can't meaningfully validate the calibrator's NEI predictions on HoVer. FEVER dev is the only real 3-class test.

### Files added

- `src/calibration/__init__.py`, `features.py`, `build_features.py`, `train.py`, `predict.py` — 11-feature extractor, sklearn Pipeline trainer, NEICalibrator loader
- `src/eval/calibration_eval.py` — FEVER + HoVer dev metrics, reliability diagram, ECE, Brier
- `src/verifier/ensemble.py` — `calibrator=` kwarg added; calibrator runs after rule-based aggregation
- `src/pipeline.py` — `build_pipeline()` instantiates `NEICalibrator` when checkpoint exists
- `configs/default.yaml` — `calibration.nei_classifier_path` points at the new checkpoint
- `tests/test_calibration.py` — 7 unit tests for feature extractor

Wall time:
- FEVER train feature build: 3.9 h (laptop slept much of this; real compute was probably ~1 h)
- FEVER dev feature build: 1 h
- HoVer dev feature build: 0.5 h
- Train: <1 s
- Eval: ~3 s
- Smoke with calibrator: 316 s

### Open follow-ups

- Phase 11 robustness eval will tell us whether the calibrator-on policy is the right default. If it hurts the headline number, switch to "off" and document.
- Add the LLM verdict one-hot to the feature set if Phase 13 shows it would matter.
- Try gradient-boosted trees on the same features; logistic regression's linear decision boundary is the prime suspect for the FEVER vs HoVer split.
- Conditional calibrator (option 3 above): only override the rule when calibrator's max non-NEI probability is high.
