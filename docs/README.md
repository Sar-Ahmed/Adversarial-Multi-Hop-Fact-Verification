# ClaimVerification v3 — Multi-Hop Claim Verification with Adversarial Distractors

This folder is the **planning + design home** for the third iteration of the multi-hop claim verification pipeline. There is no source code in this folder — every doc here is a contract for what will be built and how it will be measured.

## Background

Two prior implementations:

- **V1** (`../Task - Fact Verification/`) — phased, real artifacts, runs end-to-end. Hit a wall at REFUTED detection (3B verifier capacity); fine-tuned retriever degraded HoVer recall and was reverted; NEI=0% on FEVER cross-eval. ~60K LOC, 457 files.
- **V2** (`../ClaimVerification-v2/`) — heavy markdown, broken pipeline (decomposer crashes on schema mismatch), retriever/reranker not wired, no artifacts. ~6.4K LOC, 46 files.

V3 carries V1's tooling stack and discipline forward, adopts V2's richer dataclass schema, and fixes the gaps both versions left.

## Five things V3 must do that neither prior version did

1. **NLI-filtered adversarial distractors.** Spec says "cos≥0.85 AND opposite meaning." V1/V2 only checked cos. V3 adds an NLI-contradiction filter on the candidate pool.
2. **Trained NEI calibrator.** V2 designed it but never trained. V1 used per-token logprob thresholding (gain ~0.5%, within noise). V3 trains a logistic regressor on rich features (NLI scores, retrieval gap, entity overlap, claim length, passage variance, etc.) on FEVER NEI.
3. **Retriever fine-tune that actually helps.** V1's hard-negative fine-tune dropped HoVer R@10 by ~2.5 points. V3 mines hard negatives from the *target* distribution (HoVer + FEVER mixed, with HoVer-side validation gating).
4. **Real integration smoke test from day one.** V2's tests didn't catch the `SubClaim(sub_claim_id=...)` field mismatch because they didn't exercise the real call path. V3 wires a 5-example end-to-end smoke test in Phase 02 that runs in <5 min on CPU and is the gate for every PR.
5. **Confidence intervals on every reported metric.** Bootstrap 95% CIs on n=200 eval and (where compute permits) confirmation runs on n=1000.

## How to read this folder

Phases run roughly in numerical order with one exception: Phase 09 (temporal reasoning) is gated by Phase 13 (error analysis) — temporal handling only gets implemented if the data says it pays off.

Read in this order:

1. [REQUIREMENTS_TRACE.md](REQUIREMENTS_TRACE.md) — task spec to phase mapping.
2. [ARCHITECTURE.md](ARCHITECTURE.md) — target system, data flow.
3. [TOOLING.md](TOOLING.md) — pinned models + library versions + Colab usage.
4. [DECISIONS.md](DECISIONS.md) — explicit carry-over rules from V1/V2.
5. Phase docs in numerical order.

## Phase index

| # | Phase | Goal | Effort | Compute |
|---|---|---|---|---|
| 00 | [Project setup](PHASE_00_setup.md) | Repo skeleton, configs, observability, smoke harness | 1 day | CPU |
| 01 | [Data + corpus](PHASE_01_data_corpus.md) | HoVer + FEVER loaders, focused Wikipedia corpus, FAISS index | 1–2 days | CPU |
| 02 | [Schema + pipeline scaffolding](PHASE_02_schema_pipeline.md) | Dataclasses, pipeline config, end-to-end smoke test | 1 day | CPU |
| 03 | [Claim Decomposer](PHASE_03_decomposer.md) | Qwen2.5-3B-Q4 few-shot decomposer with fallback | 1–2 days | CPU |
| 04 | [Retrieval baseline](PHASE_04_retrieval_baseline.md) | BM25 / dense / cross-encoder rerank | 1–2 days | CPU |
| 05 | [Hard-negative mining + retriever fine-tune](PHASE_05_retriever_finetune.md) | bge-small fine-tune that beats the base on HoVer | 2–3 days | **Colab GPU** |
| 06 | [Adversarial distractor mining](PHASE_06_adversarial_distractors.md) | cos≥0.85 + NLI-contradiction filter | 1 day | CPU (or Colab for speed) |
| 07 | [Cross-document verifier](PHASE_07_verifier_ensemble.md) | LLM + DeBERTa-NLI veto ensemble | 2 days | CPU; **Colab for 7B sweep** |
| 08 | [NEI calibration](PHASE_08_nei_calibration.md) | Logistic NEI classifier on rich features | 2 days | CPU |
| 09 | [Temporal reasoning (gated)](PHASE_09_temporal.md) | Decide implement vs scope-out; trigger from Phase 13 | 1–3 days | CPU |
| 10 | [Evidence chains](PHASE_10_evidence_chains.md) | Structured chain + citation validator | 1 day | CPU |
| 11 | [Evaluation framework](PHASE_11_evaluation.md) | Macro-F1, accuracy, robustness with bootstrap CIs | 1 day | CPU |
| 12 | [Ablation study](PHASE_12_ablation.md) | Per-component contribution | 1–2 days | CPU |
| 13 | [Error analysis](PHASE_13_error_analysis.md) | 50 stratified failure cases | 1 day | CPU |
| 14 | [Human eval](PHASE_14_human_eval.md) | Evidence chain quality — 50 chains rated (deviation from 100); decomposition 4.16 / citations 4.46 / reasoning 2.58 / faithfulness 2.58 / overall 2.56 | 1 day | manual |
| 15 | [Final report](PHASE_15_final_report.md) | Headline 0.360 acc [0.295, 0.425] on HoVer dev; FEVER NEI recall 0→0.67; robustness Δ=−0.02 — see [FINAL_REPORT.md](FINAL_REPORT.md), [SETUP_AND_REPRODUCE.md](SETUP_AND_REPRODUCE.md), [HANDOFF.md](HANDOFF.md) | 0.5 day | — |

**Total estimate:** 18–25 days for one engineer (CPU-only baseline; Colab GPU access cuts Phase 05 and parts of 07 by ~40%).

## Out of scope for V3

- Web UI / API service.
- Real-time indexing of full English Wikipedia (we use a focused ~200k-passage corpus seeded from HoVer + FEVER gold titles, same approach as V1).
- Multilingual claims (English only).
- Streaming inference.
- Distributed training (single-machine fine-tune is enough for bge-small on a Colab T4).

## Anti-goals (things explicitly *not* to do, learned from V1/V2)

- Do not write a `PHASE_X_COMPLETE.md` victory-lap document. The phase doc itself is the contract; the only thing that ships at end-of-phase is updated metrics in `artifacts/`.
- Do not hard-code paths, thresholds, or model names in modules. Everything reads from `configs/default.yaml` via `PipelineConfig`.
- Do not ship a unit test that does not exercise the real call path. Mocks are allowed; mock-only test suites are not.
- Do not return hardcoded fallback values from a verifier or scorer (V2's `general_verifier.py` returned `{"entail":0.5,"contra":0.25,"neutral":0.25}` when its model was missing — that is silent lying). If a component is missing, raise.
- Do not claim "robustness" via a binary did-the-verdict-flip metric. Robustness means calibrated confidence delta + accuracy delta + per-class delta.
