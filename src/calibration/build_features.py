"""Compute calibrator training features over FEVER train / dev splits.

For each claim:
  1. retrieve top-50 via DenseRetriever
  2. rerank to top-10 with CrossEncoderReranker
  3. NLI-score (claim, passage) pairs
  4. extract feature vector
  5. emit (uid, label, features) row

Reads:  HoVer + FEVER (via src.data.load), cfg.retriever + cfg.reranker
Writes: artifacts/calibration_features_<split>.parquet

The LLM is NOT called — features here are NLI + retrieval + lexical only.
Per-claim cost ~5 s on CPU (vs ~30 s with LLM), so n=600 FEVER train fits in
~50 min and n=300 dev in ~25 min.

Run:
    python -m src.calibration.build_features --split fever_train --n 600
    python -m src.calibration.build_features --split fever_dev --n 300
    python -m src.calibration.build_features --split hover_dev --n 200
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _features_path(split: str) -> Path:
    return ROOT / "artifacts" / f"calibration_features_{split}.parquet"


def _select_examples(split: str, n: int, seed: int):  # noqa: ANN202
    """Return list of (uid, claim, label) for the given split."""
    from src.data.load import load_fever, load_hover

    rng = random.Random(seed)

    if split == "fever_train":
        fever = load_fever()
        train = fever["train"]
        # Stratify across the three classes so the calibrator sees balanced data.
        by_label: dict[str, list] = {"SUPPORTED": [], "REFUTED": [], "NEI": []}
        for ex in train:
            if ex.label in by_label:
                by_label[ex.label].append(ex)
        per_class = n // 3
        selected = []
        for cls, exs in by_label.items():
            sampled = rng.sample(exs, min(per_class, len(exs)))
            selected.extend(sampled)
            logger.info("  fever/train/{}: sampled {}/{}", cls, len(sampled), len(exs))
        rng.shuffle(selected)
        return [(ex.id, ex.claim, ex.label) for ex in selected]

    if split == "fever_dev":
        fever = load_fever()
        dev = fever["labelled_dev"]
        by_label = {"SUPPORTED": [], "REFUTED": [], "NEI": []}
        for ex in dev:
            if ex.label in by_label:
                by_label[ex.label].append(ex)
        per_class = n // 3
        selected = []
        for cls, exs in by_label.items():
            sampled = rng.sample(exs, min(per_class, len(exs)))
            selected.extend(sampled)
            logger.info("  fever/labelled_dev/{}: sampled {}/{}", cls, len(sampled), len(exs))
        rng.shuffle(selected)
        return [(ex.id, ex.claim, ex.label) for ex in selected]

    if split == "hover_dev":
        hover = load_hover()
        dev = [ex for ex in hover["validation"] if ex.supporting_facts]
        if n and len(dev) > n:
            dev = rng.sample(dev, n)
        return [(ex.uid, ex.claim, ex.label) for ex in dev]

    raise ValueError(f"unknown split: {split!r}")


@app.command()
def main(
    split: str = typer.Option("fever_train", help="fever_train | fever_dev | hover_dev"),
    n: int = typer.Option(600, help="Max examples (per class for fever)."),
    seed: int = typer.Option(42),
) -> None:
    """Compute features for one split and write a parquet."""
    import pandas as pd

    from src.calibration.features import FEATURE_NAMES, extract_features
    from src.config import PipelineConfig
    from src.reranker.cross_encoder import CrossEncoderReranker
    from src.retrieval.dense import DenseRetriever
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed
    from src.verifier.nli import NLIVerifier

    set_global_seed(seed)
    setup_logging()

    cfg = PipelineConfig.load(ROOT / "configs" / "default.yaml")
    examples = _select_examples(split, n, seed)
    logger.info("computing features for split={!r}, n={}, seed={}", split, len(examples), seed)

    retriever = DenseRetriever(cfg.retriever, cfg.corpus)
    reranker = CrossEncoderReranker(cfg.reranker)
    nli = NLIVerifier(model_name=cfg.verifier.nli_model)

    rows: list[dict] = []
    t0 = time.time()
    for i, (uid, claim, label) in enumerate(examples):
        if i % 20 == 0 and i > 0:
            rate = i / max(time.time() - t0, 1e-6)
            eta = (len(examples) - i) / max(rate, 1e-6)
            logger.info("[{}/{}] ({:.2f} cl/s, ETA {:.0f}s)", i, len(examples), rate, eta)
        candidates = retriever.retrieve(claim, top_k=cfg.retriever.top_k)
        top = reranker.rerank(claim, candidates, top_k=cfg.reranker.top_k)
        scores = nli.score(claim, top)
        feats = extract_features(claim, top, scores)
        row: dict = {"uid": str(uid), "label": label}
        for name, val in zip(FEATURE_NAMES, feats, strict=True):
            row[name] = float(val)
        rows.append(row)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    out_path = _features_path(split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    logger.info(
        "wrote {} rows to {} in {:.1f}s ({:.2f} cl/s)",
        len(df),
        out_path,
        elapsed,
        len(df) / max(elapsed, 1e-6),
    )
    logger.info("label distribution: {}", dict(df["label"].value_counts()))


if __name__ == "__main__":
    app()
