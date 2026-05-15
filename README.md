# ClaimVerification v3

Multi-hop claim verification with adversarial distractors. Built on the V1 + V2 audit (see [`docs/`](docs/README.md)).

**All planning, architecture, and phase docs live in [`docs/`](docs/README.md).** Start there.

## Quick start

```bash
make setup     # install base deps (Phase 00–02 scope, ~1 min)
make setup-ml  # install heavy ML deps when starting Phase 03 (~5-10 min)
make smoke     # run the integration smoke test (Phase 00: trivial; Phase 02 onward: real pipeline)
make test      # run all tests
make lint      # ruff + black --check
```

`requirements.txt` is intentionally light so Phase 00 setup never fails on `llama-cpp-python` or `torch` build issues. Heavy ML libs live in `requirements-ml.txt` and are installed by their owning phase. Reasoning: see [docs/PHASE_00_setup.md](docs/PHASE_00_setup.md) Outcome section.

On Windows without GNU Make installed, run the underlying commands directly, e.g.
`python -m pytest tests/ -m smoke -v`.

## Layout

```
ClaimVerification-v3/
├── docs/             # 21 planning + design docs (start at docs/README.md)
├── src/              # source code, populated phase by phase
│   └── utils/        # seed + logging helpers (Phase 00)
├── tests/            # unit + integration tests; smoke test gates every PR
├── configs/          # YAML configs (single source of truth, no hard-coded params anywhere)
├── scripts/          # model downloader, batch runners
├── notebooks/        # Colab runners (Phase 05 retriever fine-tune, Phase 07 7B sweep)
├── artifacts/        # eval JSONs, embeddings, indexes — gitignored
└── checkpoints/      # model weights — gitignored
```

## Status

| Phase | State |
|---|---|
| 00 — project setup + smoke harness | **closed** (2026-05-05) — `make smoke` 1 passed / 1 skipped, ruff + black clean, JSON log sink verified |
| 01 — data + corpus | **closed** (2026-05-05) — 177,317 passages from HoVer + FEVER gold titles (98.3% coverage); FAISS index + embeddings on disk; Inception sanity-check returns the right article at top-1 |
| 02 — schema + pipeline scaffolding | **closed** (2026-05-06) — `src/schema.py`, frozen `PipelineConfig`, end-to-end `Pipeline.verify`, `python -m src.cli verify`; 13 schema tests pass in 60 ms, 7 smoke tests in 80 s |
| 03 — claim decomposer | **closed** (2026-05-06) — Qwen2.5-3B-Q4 GGUF few-shot decomposer with retry + safe fallback; 30-claim eval shows **0% fallback rate**, avg 1.8 sub-claims; smoke test passes in 69 s with the real decomposer wired in |
| 04 — retrieval baseline | **closed** (2026-05-06) — BM25 + Dense + cross-encoder rerank, with bootstrap 95% CIs on R@K and H@K (n=200 HoVer dev). Dense+rerank: **H@10 = 0.960**, R@10 = 0.556; beats V1's reported H@10 = 0.92. Smoke test 7/7 in 110 s |
| 05 — retriever fine-tune | **closed (negative result)** (2026-05-13) — 39,872 HoVer+FEVER hard-negative triplets, Colab T4 fine-tune (7 min). Fine-tune +0.9–2.0 pts on all 6 metrics but **CIs overlap on all 6**; under the binding decision rule, production stays on **base**. Fine-tune preserved for Phase 12 ablation. Full writeup: [docs/PHASE_05_DECISION.md](docs/PHASE_05_DECISION.md) |
| 06 — adversarial distractors | **closed (partial — documented gap)** (2026-05-13) — Two-stage miner (cos≥0.55 ∧ NLI-contradicts ≥0.8) shipped 198/200 claims with full 5 distractors. **Manual sanity check: 18/20 Fail (90%)** because NLI cross-encoders flag "different entities with similar attribute patterns" as contradiction. Distractors are still harder than V1's cos-only baseline; Phase 11 robustness eval is the real test of whether they bite. Open follow-up: entity-aware re-mining if needed |
| 07 — verifier ensemble | **closed (negative result)** (2026-05-14) — Qwen 3B + DeBERTa NLI; bidirectional veto mode hits **0.36 [0.30, 0.43] acc / 0.23 macro-F1** on HoVer-dev n=200. Below trivial baselines (always-SUPPORTED = 0.51) because the 3B model predicts NEI 95% of the time on multi-hop claims. Phase 08 calibrator + optional Colab 7B sweep are the recovery paths. Smoke 8/8 in 319 s (just over `<5 min` budget). |
| 08 — NEI calibration | **closed (partial)** (2026-05-14) — Logistic regression on 11 cheap features, trained on 600 FEVER balanced. **FEVER dev NEI recall = 0.67** (vs V1's reported 0%). Macro-F1 = 0.417 (0.03 short of 0.45 target). HoVer-only accuracy drops 8.5 pts because the calibrator (trained on FEVER's 3-class) over-predicts NEI on multi-hop SUP-gold claims — known distribution mismatch, ships with calibrator on; Phase 11 will measure end-to-end. Smoke 8/8 in 316 s. |
| 10 — evidence chains | **closed** (2026-05-15) — 200/200 chains built, **validator passes 100%**, 10 stratified rendered examples. Chain shape: mean 2.76 sub-claims, 8.28 citations per chain, audit trail (LLM reasoning + NLI veto + calibrator prob) visible in every verification. End-to-end accuracy is **0.145** — 22 pts lower than Phase 07's whole-claim mode on the same n=200; Phase 11 will decide whole-claim vs decomposed mode. |
| 09 / 11–15 | not started; Phase 09 is gated by Phase 13. |

Each phase has a binding doc in `docs/PHASE_NN_*.md` with goal, deliverables, exit criteria, and risks.
