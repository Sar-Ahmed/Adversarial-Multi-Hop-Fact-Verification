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
| 03–15 | not started; see [`docs/README.md`](docs/README.md) |

Each phase has a binding doc in `docs/PHASE_NN_*.md` with goal, deliverables, exit criteria, and risks.
