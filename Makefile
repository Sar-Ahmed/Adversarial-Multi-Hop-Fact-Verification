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

# === Phase 03 targets ===

download-qwen:
	$(PYTHON) -m scripts.download_qwen

eval-decomposer:
	$(PYTHON) -m src.decomposer.eval_decomposer

# === Phase 04 targets ===

# N=0 evaluates the full HoVer dev (~4k examples, ~2 hours on CPU).
# Defaults to N=500 which finishes in ~13 min and gives tight bootstrap CIs.
N ?= 500
eval-retrieval:
	$(PYTHON) -m src.eval.retrieval_eval --n $(N)

# === Phase 05 targets ===

mine-hard-negatives:
	$(PYTHON) -m src.retrieval.finetune.mine_hard_negatives

train-bge:
	$(PYTHON) -m src.retrieval.finetune.train_bge

# === Phase 06 targets ===

mine-distractors:
	$(PYTHON) -m src.adversarial.mine --n $(N)

# === Phase 07 targets ===

eval-verifier:
	$(PYTHON) -m src.eval.verifier_eval --n $(N)

# === Phase 11 / 12 targets (placeholders, populated when phases land) ===

eval-main:
	@echo "Phase 11: not yet implemented"

eval-full:
	@echo "Phase 11: not yet implemented"

ablations:
	@echo "Phase 12: not yet implemented"
