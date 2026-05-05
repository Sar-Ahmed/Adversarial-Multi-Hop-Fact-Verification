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

## Outcome (filled at end of phase)

> Append: feature set finally used, training set size, FEVER NEI recall before vs after, HoVer accuracy delta, ECE, decision threshold finally chosen.
