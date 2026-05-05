# V3 Tooling — Pinned models, libraries, and Colab usage

All versions are pinned with `==` in `requirements.txt` and `requirements-ml.txt`. No ranges. A separate `requirements-colab.txt` exists for the two GPU notebook phases.

## Python

- **Python 3.11.x** — chosen over 3.12 because `llama-cpp-python` prebuilt wheels and `sentence-transformers` are best-tested on 3.11. Pinned in `.python-version` (currently `3.11.9`, the version available on the dev machine).

## Two requirements files (split during Phase 00)

| File | Contents | When installed |
|---|---|---|
| `requirements.txt` | `pandas`, `pyarrow`, `numpy`, `joblib`, `pyyaml`, `pydantic`, `typer`, `loguru`, `tqdm`, `pytest`, `ruff`, `black` | `make setup` (Phase 00) |
| `requirements-ml.txt` | `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`, `llama-cpp-python`, `datasets`, `rank-bm25`, `scikit-learn`, `spacy`, `dateparser` | `make setup-ml` (Phases 03+) |

The split exists because `llama-cpp-python==0.3.2` does not publish Windows wheels on PyPI and would otherwise force every contributor to install MSVC build tools just to start Phase 00. The first install attempt failed exactly this way (CMake error: `CMAKE_C_COMPILER not set`), and the resolution was to (a) defer ML deps and (b) add the abetlen prebuilt-wheel index.

`requirements-ml.txt` declares two wheel indexes:

```
--extra-index-url https://download.pytorch.org/whl/cpu
--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

The second is the official `llama-cpp-python` prebuilt CPU wheel index — used instead of building from source.

## Core libraries (CPU baseline, full set)

| Library | Pinned version | Why |
|---|---|---|
| `torch==2.6.0` (`+cpu` wheel) | CPU inference for NLI, encoder, reranker | |
| `transformers==4.49.0` | NLI cross-encoder, tokenizers | |
| `sentence-transformers==3.4.1` | bge-small encoding + fine-tuning | |
| `faiss-cpu==1.10.0` | Flat-IP index over corpus embeddings | |
| `llama-cpp-python==0.3.2` | Local Qwen 3B Q4 GGUF inference, logprobs | |
| `datasets==2.20.0` | HoVer + FEVER loaders | |
| `rank-bm25==0.2.2` | Baseline retriever + hard-negative source | |
| `scikit-learn==1.5.2` | Logistic regression NEI classifier, metrics | |
| `numpy<2.0` | torch/faiss compat | |
| `pandas==2.2.2` | Parquet IO for corpus + eval results | |
| `pyarrow==17.0.0` | Parquet engine | |
| `pyyaml==6.0.2` | Config loading | |
| `pydantic==2.9.2` | Optional: validate `PipelineConfig` from YAML | |
| `typer==0.12.5` | CLI entry point | |
| `loguru==0.7.2` | Structured logging (replaces V1's print-debug, V2's basic logging) | |
| `tqdm==4.66.5` | Progress bars during corpus encoding + eval loops | |
| `pytest==8.3.3` | Tests, including the integration smoke test | |

## Models

All models live in `checkpoints/` (gitignored) and are downloaded by `scripts/fetch_models.sh` on first run.

| Model | Size | License | Role | Phase |
|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | ~130 MB | MIT | Dense encoder (base + fine-tune source) | 04, 05 |
| `BAAI/bge-reranker-base` | ~280 MB | Apache 2.0 | Cross-encoder reranker | 04 |
| `cross-encoder/nli-deberta-v3-base` | ~430 MB | MIT | Contradiction / entailment scores; veto signal | 06, 07, 08 |
| `Qwen/Qwen2.5-3B-Instruct-GGUF` (Q4_K_M) | ~2.4 GB | Apache 2.0 | Decomposer + verifier LLM | 03, 07 |
| (Optional, Colab) `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | Apache 2.0 | Verifier ablation; only if Phase 07 sweep justifies | 07 |

## Colab usage

V1 demonstrated that Colab is a fine accelerator for the two GPU-bound phases. We use it the same way and check the notebooks in alongside the code.

### Phase 05 — retriever fine-tune (Colab T4 recommended)

- Notebook: `notebooks/phase05_finetune_retriever.ipynb`
- Runtime: Colab → Runtime → Change runtime type → T4 GPU
- Reads: `artifacts/fever_hover_hard_negatives.jsonl` (uploaded to Drive or generated in-notebook)
- Writes: `checkpoints/bge-small-v3-hn/` back to local via `gcloud` or manual download
- Wall time: ~30–45 min for 1 epoch over 20–30k triplets
- CPU fallback: possible but ~6–8 h; not recommended

### Phase 07 — verifier sweep (Colab T4 / A100 optional)

- Notebook: `notebooks/phase07_verifier_sweep.ipynb`
- Only invoked if the CPU 3B baseline shows <55% accuracy after the NLI veto. Then we run a 7B FP16 sweep to confirm whether the bottleneck is model-capacity or prompt-design.
- vLLM (OpenAI-compatible API) for batched inference on Colab.

### Phase 06 — distractor mining (CPU enough; Colab speeds it up ~5×)

- Optional Colab usage to encode the 200k corpus and run NLI over candidate distractor pairs faster. Outputs land in `artifacts/distractors_v3.json` either way.

## What we are NOT using (and why)

| Library | Why excluded |
|---|---|
| **LangChain** | Adds 2–3 layers of indirection over plain function calls; obscures the actual prompt sent to the LLM and the actual passages retrieved. Both V1 and V2 (rightly) avoided it. |
| **LlamaIndex** | Same reason. |
| **Haystack** | Heavy framework for a focused pipeline; we want each component readable in <300 LOC. |
| **vLLM in production** | Requires GPU. Used only in the optional Colab Phase 07 sweep. |
| **Outlines / grammar-constrained decoding** | V1 evaluated and rejected — finicky integration with llama-cpp. We rely on regex JSON extraction with a single retry, same as V1, given that V1 reported 0% fallback rate on n=200. |
| **Weaviate / Chroma / Pinecone** | FAISS flat-IP at 200k passages is plenty fast (single-digit ms). External vector DBs add deployment surface for no benefit. |
| **wandb** | Optional. Loguru + JSON eval dumps cover the observability gap V1 had without adding a SaaS dependency. Engineers may opt in via `WANDB_PROJECT=...` env var. |

## Reproducibility checklist (verified at end of Phase 00)

- [ ] `requirements.txt` uses only `==` pins.
- [ ] `.python-version` pins the patch version.
- [ ] `make smoke` runs the 5-example integration test on a fresh checkout in <5 min and exits 0.
- [ ] Every entry point sets the global seed.
- [ ] Eval result filenames embed git SHA and timestamp.
- [ ] `scripts/fetch_models.sh` is idempotent and verifies model checksums after download.
