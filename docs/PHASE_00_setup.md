# Phase 00 — Project Setup

**Goal.** Create the V3 repo skeleton, pin all dependencies, wire `loguru` structured logging, and stand up the CI smoke harness that every later phase will gate against.

**Effort.** 1 day.
**Compute.** CPU.
**Depends on.** Nothing.

## Why this exists

V1 had no tests and no observability — long-running evals failed silently. V2 had tests that didn't catch a one-line schema bug. V3 is gated by an integration smoke test starting now, so by the time we add real models in Phase 03+ there is nowhere to hide.

## Inputs

- The audit (`../README.md` of this folder).
- `Task 1.PDF` for the requirements check.

## Deliverables

- `ClaimVerification-v3/` repo root with:
  - `src/` (empty `__init__.py` files for now), `tests/`, `configs/`, `scripts/`, `notebooks/`, `artifacts/`, `checkpoints/`, `docs/`.
  - `src/utils/seed.py` — single global seed setter.
  - `src/utils/logging.py` — `loguru` JSON sink config.
  - `configs/default.yaml` — full schema (placeholder values where unknown).
  - `requirements.txt` — `==`-pinned versions per [TOOLING.md](TOOLING.md).
  - `requirements-colab.txt` — Colab-specific overrides.
  - `.python-version` — pinned Python.
  - `.gitignore` — excludes `checkpoints/`, `artifacts/`, `data/raw/`, `*.faiss`, `*.npy`, `*.gguf`, `wandb/`, `__pycache__/`.
  - `Makefile` with targets: `setup`, `smoke`, `test`, `lint`, `clean`.
  - `pyproject.toml` (ruff + black config; pytest config).
  - `tests/test_smoke_placeholder.py` — passes trivially today; replaced in Phase 02.
  - `scripts/fetch_models.sh` — idempotent model downloader (HF hub).
  - `README.md` (in repo root, not this folder) — quick-start and `make smoke` instructions.

## Technical approach

- **Logging:** loguru with two sinks — stderr human-readable for dev, `artifacts/logs/run_<timestamp>.jsonl` for replay. Always include `phase`, `claim_id`, and `git_sha` keys when available.
- **Seeding:** one function `set_global_seed(seed: int)` that sets `random`, `numpy.random`, `torch`, and the `PYTHONHASHSEED` env var. Called once at every entry-point.
- **Config:** YAML loaded by `pydantic` model `PipelineConfig` (full schema in Phase 02; just stubs here). Frozen dataclass after load — no mutation downstream.
- **CI:** GitHub Actions workflow (or local `make smoke`) running `pytest tests/ -m smoke` on every commit. Phase 00 ships with a no-op smoke test; Phase 02 replaces it.

## Implementation steps

1. `mkdir -p src tests configs scripts notebooks artifacts checkpoints docs` and add empty `__init__.py` files.
2. Write `requirements.txt` from the [TOOLING.md](TOOLING.md) table — every line `==`-pinned.
3. Set up venv: `python -m venv .venv && pip install -r requirements.txt`. Verify `pytest`, `ruff`, `black` import.
4. Implement `src/utils/seed.py` and `src/utils/logging.py` as described.
5. Write `configs/default.yaml` as a placeholder (sections for retriever / reranker / verifier / calibration / eval — values can be `null` until phase that owns them).
6. Write `Makefile` with the targets above; verify `make smoke` exits 0 with the placeholder test.
7. Add `scripts/fetch_models.sh` (idempotent — checks hash before download).
8. Commit with message `phase-00: project skeleton, smoke harness, pinned deps`.

## Exit criteria

- [ ] `make smoke` exits 0 on a clean checkout in <30 s.
- [ ] `pip install -r requirements.txt` succeeds on Python 3.11 with no warnings about version conflicts.
- [ ] `ruff check src/` returns no findings.
- [ ] `python -c "from src.utils.seed import set_global_seed; set_global_seed(42)"` runs without error.
- [ ] `loguru` JSON sink writes to `artifacts/logs/run_*.jsonl` after a one-line test invocation.
- [ ] Folder skeleton matches [ARCHITECTURE.md](ARCHITECTURE.md) section "Modules and ownership".

## Risks and gotchas

- Llama-cpp wheels on Windows are sometimes ABI-mismatched; document the verified install command in `docs/SETUP.md`. (Phase 03 will exercise this.)
- `faiss-cpu` on Windows needs `swig`; alternative is to develop on WSL2.
- Pinning is strict but `torch+cpu` wheels need a wheel index (`--extra-index-url https://download.pytorch.org/whl/cpu`) — encode this in `requirements.txt` via `--find-links` or a separate `pip install` step in `make setup`.

## What NOT to do

- No `PHASE_00_COMPLETE.md` victory-lap file in the repo. Exit criteria + commit message are sufficient.
- No `requirements.txt` ranges. `==` only.
- No global state in `src/utils/`. Pure helpers.

## Outcome (Phase 00 closed 2026-05-05)

**Wall time.** ~30 min total including the failed first install attempt.

**Deviations from plan.**

1. **Requirements split.** Original plan was a single `requirements.txt`. First install failed because `llama-cpp-python==0.3.2` ships no Windows wheels on PyPI and the dev machine has no MSVC build toolchain (CMake error: `CMAKE_C_COMPILER not set`). Resolution: split into `requirements.txt` (light, Phase 00–02 scope) and `requirements-ml.txt` (heavy ML libs deferred to their owning phases). The ML file adds the `abetlen` prebuilt CPU wheel index for `llama-cpp-python` so future contributors don't need an MSVC toolchain either.
2. **Python version.** `.python-version` pinned to `3.11.9` (what `py -3.11` resolved to on the dev machine) instead of `3.11.10`. Both are 3.11 patch releases; no functional difference.
3. **Makefile.** Added `make setup-ml` target alongside `make setup`. Phase 03 docs direct contributors to run it.

**Verified against exit criteria.**

- `make setup` (base requirements) clean install in <2 min.
- `pytest -m smoke` → 1 passed, 1 skipped (the SubClaim regression test, intentionally deferred to Phase 02 because the schema doesn't exist yet). Wall time 1.62 s.
- `ruff check src/ tests/` → all checks passed.
- `black --check src/ tests/` → all 6 files clean.
- `loguru` JSON sink writes structured records to `artifacts/logs/run_YYYYMMDD_HHMMSS.jsonl` — confirmed via direct invocation.

**Open follow-ups.**

- Phase 02 will replace `tests/test_smoke_placeholder.py` with the real 5-claim end-to-end smoke test.
- Phase 03 will be the first to run `make setup-ml` and verify the abetlen prebuilt wheel actually resolves on Windows + Python 3.11.
