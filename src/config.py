"""Frozen pipeline configuration loaded from YAML.

Single source of truth — every module reads from this object. No module is
permitted to hard-code a path, threshold, or model name.

Design (per docs/DECISIONS.md):

- pydantic v2 BaseModel with `frozen=True` so accidental mutation downstream
  raises rather than silently corrupting state.
- Each component owns a sub-model. Phase 02 ships placeholder values where
  the owning phase hasn't landed yet — `null` in YAML → `None` in Python.
- Loader is `PipelineConfig.load(path)`; constructor stays vanilla.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvalConfig(_Frozen):
    seed: int = 42
    n_bootstrap: int = 1000
    default_split: str = "hover_dev_n200"


class CorpusConfig(_Frozen):
    parquet_path: str
    faiss_path: str
    embeddings_path: str


class DecomposerConfig(_Frozen):
    llm_path: str | None
    n_ctx: int = 4096
    max_tokens: int = 512
    temperature: float = 0.0
    prompt_examples: int = 6


class RetrieverConfig(_Frozen):
    encoder: str
    finetune_path: str | None
    top_k: int = 50
    query_prefix: str = ""


class RerankerConfig(_Frozen):
    model: str
    top_k: int = 10


class AdversarialConfig(_Frozen):
    distractors_path: str
    cos_threshold: float = 0.85
    nli_contra_threshold: float = 0.8
    inject_mode: str = "mix"
    k: int = 5


class VerifierConfig(_Frozen):
    llm_path: str | None
    nli_model: str
    mode: str = "llm_plus_nli_veto"
    contra_veto_threshold: float = 0.95
    entail_threshold: float = 0.7


class CalibrationConfig(_Frozen):
    nei_classifier_path: str | None
    scaler_path: str | None
    decision_threshold: float = 0.5


class PipelineConfig(_Frozen):
    eval: EvalConfig
    corpus: CorpusConfig
    decomposer: DecomposerConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig
    adversarial: AdversarialConfig
    verifier: VerifierConfig
    calibration: CalibrationConfig

    @classmethod
    def load(cls, path: str | Path = "configs/default.yaml") -> PipelineConfig:
        """Load + validate config from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
