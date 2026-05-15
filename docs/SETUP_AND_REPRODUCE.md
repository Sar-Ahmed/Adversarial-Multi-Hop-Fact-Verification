# Setup and Reproduce

How to get a fresh checkout running and reproduce every number in [`FINAL_REPORT.md`](FINAL_REPORT.md). Each artifact has a single command to regenerate it. Compute budget for the full reproduction is **~8 hours of CPU + 25 hours of Colab T4 GPU** (the GPU portion is Phase 05's retriever fine-tune, optional — base bge-small ships and the fine-tune CIs overlap).

If you only need the headline number ([Section: Headline numbers](FINAL_REPORT.md#headline-numbers)), skip to the §"Minimal reproduction" at the bottom (~30 min CPU + model downloads).

## 0. Prerequisites

- **Python 3.11** (any 3.11.x). The lockfiles target 3.11.
- **Git LFS** is *not* required — all tracked artifacts are text JSON / CSV / Markdown; large binaries (corpus FAISS index, GGUF model weights) regenerate from scripts.
- **OS.** Linux/macOS/Windows all work. The repo was developed on Windows 11 with PowerShell; Bash commands below also work in WSL or Linux. PowerShell equivalents are mostly identical (substitute `python` for `python`).
- **Disk.** ~8 GB total: ~2 GB GGUF model, ~1.5 GB HuggingFace cache (bge-small + reranker + NLI), ~3 GB corpus + FAISS index, ~1 GB artifacts.

## 1. Clone and install

```bash
git clone git@github.com:Sar-Ahmed/Adversarial-Multi-Hop-Fact-Verification.git ClaimVerification-v3
cd ClaimVerification-v3

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

make setup        # installs requirements.txt (CPU-only base deps)
make setup-ml     # installs requirements-ml.txt (torch, sentence-transformers, llama-cpp-python CPU wheel)
```

If `llama-cpp-python` fails on Windows, the prebuilt CPU wheel index is referenced in `requirements-ml.txt`:
```
--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

## 2. Smoke test (verifies the install)

```bash
make smoke
```

Should run 8 tests in ~5 seconds and exit 0. If it fails, do not proceed — the install is broken.

## 3. Fetch models

```bash
make fetch-models   # downloads Qwen 2.5-3B-Q4 GGUF (~2 GB) to checkpoints/
```

The bge-small, reranker, and NLI cross-encoder are pulled by HuggingFace on first use; no separate fetch step needed.

## 4. Build the corpus and FAISS index (Phase 01)

```bash
make corpus
# = make build-corpus + make encode-corpus
```

~25 min on CPU. Produces:
- `data/corpus.jsonl` (~200k passages, ~150 MB)
- `artifacts/corpus_stats.json`
- `artifacts/corpus.faiss` + `artifacts/corpus.npy` (FAISS IndexFlatIP, bge-small embeddings)

## 5. Reproduce every headline number

Order matters where one artifact feeds another. Each `python -m ...` writes to `artifacts/` and is **resume-safe** (rerun = pick up where it left off; see `--resume` flags).

### 5.1 Decomposer evaluation (Phase 03)

```bash
python -m src.decomposer.eval_decomposer
# → artifacts/decomposer_eval.json (fallback rate, atomic-claim ratio)
```
Runtime: ~5 min.

### 5.2 Retrieval baselines (Phase 04)

```bash
make eval-retrieval N=200
# → artifacts/retrieval_eval_baseline.json
# Reproduces Hit@10 for BM25 / dense / dense+rerank rows of FINAL_REPORT.md § Retrieval
```
Runtime: ~25 min CPU.

### 5.3 (Optional, GPU) Retriever fine-tune (Phase 05)

Run in Colab — notebook stub at `notebooks/phase05_finetune.ipynb`. On a T4: ~25 h.

```python
# Inside Colab:
!git clone <repo>
!pip install -r requirements-colab.txt
!python -m src.retrieval.finetune.mine_hard_negatives
!python -m src.retrieval.finetune.train_bge --epochs 1 --no-fp16
# Download checkpoints/bge-small-v3-hn back to local
```

Then locally:

```bash
python -m src.data.encode_corpus_finetune --model checkpoints/bge-small-v3-hn --suffix _ft
python -m src.eval.retrieval_eval_finetune --n 200
# → artifacts/retriever_eval_finetune.json
```

**This step is optional** — the fine-tune's CIs overlap the base retriever's per Phase 05's outcome doc. The headline production config ships base.

### 5.4 Adversarial distractor mining (Phase 06)

```bash
make mine-distractors N=200
# → artifacts/distractors_v3.json + artifacts/distractor_sanity_check.md
```
Runtime: ~30 min CPU.

### 5.5 Verifier evaluation (Phase 07)

```bash
make eval-verifier N=200
# → artifacts/per_subclaim_traces.jsonl + artifacts/verifier_eval_phase07.json
```
Runtime: **~16 hours CPU**. Resume-safe — laptop sleep is fine; rerun the command and it picks up from the last cached row. This step generates the LLM verdicts + NLI signals that everything downstream re-uses.

### 5.6 NEI calibrator (Phase 08)

```bash
python -m src.calibration.build_features
python -m src.calibration.train
python -m src.calibration.eval
# → artifacts/calibration_features_*.parquet, checkpoints/nei_classifier.joblib, artifacts/calibration_eval.json
```
Runtime: ~12 hours CPU (most of it is FEVER train feature extraction via LLM-cached signals).

### 5.7 Evidence chain construction (Phase 10)

```bash
python -m src.eval.build_chains --n 200
# → artifacts/evidence_chains.jsonl (200 chains, ~1.7 MB)
```
Runtime: ~10 min CPU (re-aggregates cached traces).

### 5.8 Main eval (Phase 11) — **headline accuracy + macro-F1**

```bash
python -m src.eval.run_eval --n 200
# → artifacts/eval_main.json, artifacts/per_class_breakdown.json
```
Runtime: ~5 min CPU. **This produces FINAL_REPORT.md's headline 0.360 [0.295, 0.425].**

### 5.9 Adversarial robustness (Phase 11)

```bash
python -m src.eval.robustness --n 50 --paired
# → artifacts/robustness_eval.json, artifacts/adversarial_traces.jsonl
```
Runtime: ~3 hours CPU. Re-uses cached LLM signals from §5.5 where available; only the distractor-injected examples require new LLM calls. **Produces the −0.020 [−0.060, 0.000] robustness Δ.**

### 5.10 Ablation study (Phase 12)

```bash
python -m src.eval.ablation
# → artifacts/ablation_results.json
```
Runtime: ~5 min CPU (re-aggregation only; no new LLM calls).

### 5.11 Error analysis (Phase 13)

```bash
python -m src.analysis.categorize
# → artifacts/failures_for_tagging.md, artifacts/failures_tagged.json
```
Runtime: ~2 min. Produces the 50-failure taxonomy: 28 nei_miscalibration, 12 partial_match_as_full, 10 entity_confusion, 0 others.

### 5.12 Human eval (Phase 14)

```bash
python -m src.evidence.sample_for_eval
# → artifacts/human_eval_sample.csv (empty rating cols), artifacts/human_eval_rendered.txt

# (manual rating step — ~2 hours, follow docs/HUMAN_EVAL_PROTOCOL.md)
# Fill in decomposition, citations, reasoning, faithfulness, overall columns 1-5

python -m src.evidence.aggregate_human_eval
# → artifacts/human_eval_summary.json
```

The CSV in this repo is already filled — running `aggregate_human_eval` will reproduce the existing summary JSON from the committed ratings.

## 6. Minimal reproduction (headline only, ~30 min)

If you only want to verify the headline 0.360 accuracy without running 16-hour verifier sweeps:

```bash
# 1. Install (steps 1-3 above)
# 2. Build the corpus (~25 min)
make corpus

# 3. Skip Phase 07 sweep — the cached traces are in artifacts/per_subclaim_traces.jsonl (committed)
# 4. Re-aggregate from cache:
python -m src.eval.run_eval --n 200
cat artifacts/eval_main.json | python -m json.tool | head -50
```

This works because `per_subclaim_traces.jsonl` is the expensive LLM output, and it's a committed artifact. The re-aggregation step is pure Python and takes ~5 min.

## 7. Seeds and determinism

All eval scripts default to `seed=42`. Bootstrap uses 1000 resamples. The LLM verifier (Qwen 3B Q4 via llama-cpp-python) uses `temperature=0` for verdict prompts, but llama.cpp is not bit-deterministic across CPU revisions — expect ±1 example difference on a fresh laptop run vs the committed traces. The bootstrap CIs absorb this.

## 8. Cache layout

- `artifacts/` — all evaluation outputs. Tracked in git as text (committed JSON/CSV/JSONL).
- `checkpoints/` — model weights. Gitignored. Regenerable from `make fetch-models` + Phase 05 fine-tune.
- `data/` — corpus + dataset cache. Gitignored. Regenerable from `make corpus`.
- `.venv/` — Python venv. Gitignored.

## 9. If something fails

- Smoke test fails → install is broken; re-run `make setup setup-ml`.
- `make eval-retrieval` returns near-zero R@10 → check `data/corpus.jsonl` is non-empty and `artifacts/corpus.faiss` exists.
- `make eval-verifier` hangs → llama-cpp is loading the GGUF; the first call takes ~30s. Subsequent calls are <1s/inference.
- Resume after laptop sleep → just rerun the same command. All long-running scripts use `fh.flush()` after each row and a `--resume` flag (default on).

## 10. Where to look next

After reproducing the headline, the highest-leverage reading order for an auditor:

1. [`docs/FINAL_REPORT.md`](FINAL_REPORT.md) — start here.
2. [`docs/ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md) — three fix paths with expected lift.
3. [`docs/ABLATION_TABLE.md`](ABLATION_TABLE.md) — what is and isn't load-bearing.
4. [`docs/SCOPED_OUT.md`](SCOPED_OUT.md) — what's deliberately not in V3.
5. [`docs/HANDOFF.md`](HANDOFF.md) — first 30 minutes for the next engineer.
