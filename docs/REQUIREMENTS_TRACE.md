# Requirements Trace — Spec to Phase Mapping

Every requirement and known failure mode from `Task 1.PDF` mapped to the phase that owns it and the artifact that proves delivery.

## Technical requirements

| # | Requirement (verbatim from spec) | Owning phase(s) | Proof-of-delivery artifact |
|---|---|---|---|
| 1 | **Claim Decomposer:** Use an LLM to break compound claims into atomic sub-claims. Each sub-claim must be independently verifiable. Handle nested logic ("X who did Y also did Z"). | [Phase 03](PHASE_03_decomposer.md) | `artifacts/decomposer_eval.json` (n≥30 manual quality rating ≥4/5) |
| 2a | **Retriever with Hard Negative Mining:** Implement dense passage retrieval (DPR or Contriever) fine-tuned on FEVER. | [Phase 04](PHASE_04_retrieval_baseline.md), [Phase 05](PHASE_05_retriever_finetune.md) | `checkpoints/bge-small-v3-hn/` + `artifacts/retriever_eval.json` showing R@10 vs base |
| 2b | For each claim, retrieve top-20 passages, then inject 5 adversarial distractors that have >0.85 cosine similarity but opposite semantic meaning. | [Phase 06](PHASE_06_adversarial_distractors.md) | `artifacts/distractors_v3.json` with both `cos≥0.85` and `nli_contra_prob≥0.8` per record |
| 3 | **Cross-Document Reasoner:** Build an attention-based module that takes all retrieved passages and performs multi-hop reasoning. Must handle coreference, temporal disambiguation, and partial entailment. | [Phase 07](PHASE_07_verifier_ensemble.md), [Phase 09](PHASE_09_temporal.md) (gated) | `artifacts/verifier_eval.json` + per-failure-mode breakdown |
| 4 | **Evidence Chain Generator:** For each verdict, output a step-by-step evidence chain showing which passages support which sub-claims. Must be human-readable and auditable. | [Phase 10](PHASE_10_evidence_chains.md) | `artifacts/evidence_chains.jsonl` (full eval set) + `evidence_chain_render_examples.txt` |
| 5 | **Adversarial Robustness Evaluation:** Measure accuracy drop between clean retrieval vs. adversarial retrieval. Target: <5% accuracy degradation. | [Phase 11](PHASE_11_evaluation.md) | `artifacts/robustness_eval.json` with clean and adversarial macro-F1 + 95% CIs |

## Evaluation metrics required

| Metric | Owning phase | Artifact |
|---|---|---|
| Macro F1 on 3-class classification (SUPPORTED / REFUTED / NEI) | [Phase 11](PHASE_11_evaluation.md) | `artifacts/eval_main.json` |
| Label accuracy on HoVer dev set | [Phase 11](PHASE_11_evaluation.md) | same |
| Adversarial Robustness Score = Acc_clean − Acc_adversarial | [Phase 11](PHASE_11_evaluation.md) | `artifacts/robustness_eval.json` |
| Evidence chain quality via human evaluation (sample 100 examples) | [Phase 14](PHASE_14_human_eval.md) | `artifacts/evidence_human_eval.csv` |

## Failure modes called out in the spec

| # | Failure mode (verbatim) | Mitigation phase(s) | How V3 addresses it |
|---|---|---|---|
| 1 | Surface-level keyword overlap will destroy your retriever. You need semantic filtering, not just cosine similarity. | [Phase 04](PHASE_04_retrieval_baseline.md) reranker; [Phase 05](PHASE_05_retriever_finetune.md) fine-tune | Cross-encoder reranker on top-50 → 10; fine-tune on hard negatives (BM25-skip-top-10 trick) sourced from HoVer + FEVER |
| 2 | Multi-hop reasoning failure — LLM tries to answer from a single passage. Without explicit chain-of-thought enforcement, accuracy drops ~20%. | [Phase 07](PHASE_07_verifier_ensemble.md) | Per-sub-claim verification with explicit dependency-DAG ordering, all signals fed to NLI veto and aggregator |
| 3 | Temporal reasoning is the hidden killer. | [Phase 09](PHASE_09_temporal.md) (gated by Phase 13) | Gated decision: implement (entity, attribute, value, time-window) extraction + comparison if Phase 13 shows ≥10% temporal failures; else explicitly scope-out and document |
| 4 | NEI is the hardest class — LLMs are bad at "I don't know"; need calibrated confidence, not argmax. | [Phase 08](PHASE_08_nei_calibration.md) | Trained logistic regression on 12 features (NLI scores, retrieval gap, entity overlap, etc.) on FEVER NEI |

## Required deliverables

| Deliverable (verbatim) | Phase that produces it | File |
|---|---|---|
| End-to-end pipeline code with modular components (decomposer, retriever, reasoner, verifier) | [Phase 02](PHASE_02_schema_pipeline.md) (skeleton) → [Phase 07](PHASE_07_verifier_ensemble.md) (full) | `src/pipeline.py` + `src/cli.py` |
| Fine-tuned retriever checkpoint with hard negative training logs | [Phase 05](PHASE_05_retriever_finetune.md) | `checkpoints/bge-small-v3-hn/` + `artifacts/retriever_finetune_log.jsonl` |
| Evaluation report: F1, accuracy, robustness score, evidence chain quality | [Phase 11](PHASE_11_evaluation.md), [Phase 14](PHASE_14_human_eval.md), [Phase 15](PHASE_15_final_report.md) | `docs/FINAL_REPORT.md` |
| Error analysis document: 50 failure cases categorized by failure mode | [Phase 13](PHASE_13_error_analysis.md) | `docs/ERROR_ANALYSIS.md` + `artifacts/failures_tagged.json` |
| Ablation study: performance with/without each component | [Phase 12](PHASE_12_ablation.md) | `artifacts/ablation_results.json` + table in final report |

## Coverage summary

| Spec area | Covered? | Phase(s) |
|---|---|---|
| Decomposition | Yes | 03 |
| Dense retrieval | Yes | 04 |
| Hard-negative fine-tune | Yes | 05 |
| Adversarial distractor injection (cos + opposite meaning) | Yes | 06 |
| Multi-hop reasoning | Yes | 07 |
| Coreference | Optional / inside LLM context | 07 (no separate module unless triggered by 13) |
| Temporal | Conditional | 09 (gated by 13) |
| Evidence chain | Yes | 10 |
| 3-class classification with NEI | Yes | 07 + 08 |
| Robustness target <5% drop | Yes | 11 |
| Macro F1, accuracy | Yes | 11 |
| Evidence chain human eval (n=100) | Yes | 14 |
| Error analysis (n=50) | Yes | 13 |
| Ablation | Yes | 12 |

No spec line item is unowned.
