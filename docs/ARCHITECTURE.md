# V3 Target Architecture

## Data flow

```
Claim
  │
  ▼
[Decomposer]  ─→ SubClaim graph (id, text, depends_on, reasoning_type)
  │
  ▼  for each sub-claim:
[Dense Retriever]  (FAISS flat-IP, top-50)
  │
  ▼
[Cross-Encoder Reranker]  ─→ top-10 evidence candidates
  │
  ├── [Adversarial Distractor Injector]  (eval mode only)
  │       ↳ inject 5 NLI-filtered cos≥0.85 distractors
  ▼
[LLM Verifier]              ─┐
[DeBERTa NLI Cross-Encoder] ─┤── [Verdict Aggregator]
[Retrieval / overlap features]┘    │
                                   ▼
                          [NEI Calibrator]
                          (logistic regression
                           on rich features)
                                   │
                                   ▼
                          [Evidence Chain Builder]
                          → SubClaim → cited passage IDs
                          → reasoning → verdict + confidence
```

## Modules and ownership

| Module path | Backbone | Role | Phase |
|---|---|---|---|
| `src/data/` | — | HoVer + FEVER loaders, corpus builder | 01 |
| `src/schema.py` | — | Dataclasses (`Passage`, `SubClaim`, `EvidenceChain`, `Verdict`, `Label`) | 02 |
| `src/config.py` | — | `PipelineConfig` frozen dataclass loaded from YAML | 02 |
| `src/decomposer/` | Qwen2.5-3B-Instruct Q4_K_M (llama-cpp) | Compound → atomic sub-claims with depends_on edges | 03 |
| `src/retrieval/dense.py` | `BAAI/bge-small-en-v1.5` (or fine-tune from Phase 05) | Top-50 dense retrieval, FAISS flat-IP | 04 |
| `src/retrieval/bm25.py` | rank_bm25 | Baseline + hard-negative source | 04 |
| `src/reranker/` | `BAAI/bge-reranker-base` cross-encoder | Top-50 → top-10 | 04 |
| `src/retrieval/finetune/` | sentence-transformers + MNRL | Hard-negative fine-tune | 05 |
| `src/adversarial/` | bge-small + DeBERTa NLI | Mine cos≥0.85 ∧ NLI-contradicts distractors | 06 |
| `src/verifier/llm.py` | Qwen 3B Q4 (llama-cpp, logprobs) | Generative SUPPORTED / REFUTED / NEI | 07 |
| `src/verifier/nli.py` | `cross-encoder/nli-deberta-v3-base` | Per-passage entail / contradict / neutral scores | 07 |
| `src/verifier/aggregate.py` | rule + features | Combine LLM verdict + NLI veto + features | 07 |
| `src/calibration/nei.py` | Logistic regression (sklearn) on 12 features | Trained NEI calibrator | 08 |
| `src/temporal/` | (gated by Phase 09) | Optional date/time-window resolver | 09 |
| `src/evidence/chain.py` | Deterministic builder | Structured + human-readable | 10 |
| `src/eval/` | sklearn + bootstrap | Macro-F1, accuracy, CIs, robustness | 11–14 |
| `src/cli.py` | typer or argparse | Single entry point: `python -m src.cli verify "<claim>"` | 02 |

## Single source of configuration

Every hyperparameter lives in `configs/default.yaml` and is loaded into a frozen `PipelineConfig` dataclass at startup. **No module hard-codes a path, threshold, or model name.** This is the explicit fix for V1's vestigial `default.yaml` and V2's pipeline.py with `self.retriever = None` placeholders.

Example (illustrative — final keys decided in Phase 02):

```yaml
corpus:
  parquet_path: artifacts/corpus.parquet
  faiss_path: artifacts/corpus.faiss
retriever:
  encoder: BAAI/bge-small-en-v1.5
  finetune_path: checkpoints/bge-small-v3-hn   # set null to use base
  top_k: 50
  query_prefix: "Represent this sentence for searching relevant passages: "
reranker:
  model: BAAI/bge-reranker-base
  top_k: 10
verifier:
  llm_path: checkpoints/qwen2.5-3b-instruct-q4_k_m.gguf
  nli_model: cross-encoder/nli-deberta-v3-base
  contra_veto_threshold: 0.95
  entail_threshold: 0.7
adversarial:
  distractors_path: artifacts/distractors_v3.json
  cos_threshold: 0.85
  nli_contra_threshold: 0.8
calibration:
  nei_classifier_path: checkpoints/nei_classifier.joblib
  decision_threshold: 0.5
eval:
  seed: 42
  n_bootstrap: 1000
```

## Inference modes

| Mode | When | Components used |
|---|---|---|
| `clean` | Production-style eval | Dense → rerank → verifier → calibrate → chain |
| `adversarial` | Robustness eval | Same + distractor injector mixes 5 distractors into the rerank pool |
| `ablation:<component>` | Phase 12 | Disables one named component (`bm25_only`, `no_reranker`, `no_nli_veto`, `no_calibrator`, `no_finetune`) |

## Determinism and reproducibility

- Global seed (`numpy.random.seed`, `torch.manual_seed`, `random.seed`, `dataset.shuffle(seed=...)`) set once in `src/utils/seed.py` and called from every entry point.
- Eval scripts emit `artifacts/<phase>_<timestamp>_<git_sha>.json` so reruns never overwrite each other and any number can be traced to a commit.
- `requirements.txt` uses `==` pins (not ranges); fix for V1's `>=2.18,<3.0` looseness.
