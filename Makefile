# ClaimVerification v3 — Makefile.
# Cross-platform note: targets call `python -m <tool>` so they work on Windows
# (with Make from chocolatey/scoop or WSL) and on Linux/macOS.

.PHONY: setup smoke test test-all lint format clean fetch-models

PYTHON := python

# === Phase 00 targets ===

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

setup-ml:
	$(PYTHON) -m pip install -r requirements-ml.txt
	$(PYTHON) -m spacy download en_core_web_sm

smoke:
	$(PYTHON) -m pytest tests/ -m smoke -v

test:
	$(PYTHON) -m pytest tests/ -v -m "not slow"

test-all:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m black --check src/ tests/

format:
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', '.ruff_cache')]"

fetch-models:
	bash scripts/fetch_models.sh

# === Phase 01 targets ===

build-corpus:
	$(PYTHON) -m src.data.build_corpus

encode-corpus:
	$(PYTHON) -m src.data.encode_corpus

corpus: build-corpus encode-corpus

# === Phase 11 / 12 targets (placeholders, populated when phases land) ===

eval-main:
	@echo "Phase 11: not yet implemented"

eval-full:
	@echo "Phase 11: not yet implemented"

ablations:
	@echo "Phase 12: not yet implemented"
