"""Phase 04 retrieval baseline evaluation.

Computes Recall@5 / @10 / @20 on a HoVer-dev sample for three configurations:
  - BM25                       (rank_bm25)
  - Dense                      (bge-small-en-v1.5 + FAISS)
  - Dense + cross-encoder      (bge-reranker-base over the dense top-50)

Each metric is reported with a bootstrap 95% CI (n=1000 resamples).

Run:  python -m src.eval.retrieval_eval --n 500
Or:   make eval-retrieval N=500
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "artifacts" / "retrieval_eval_baseline.json"

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _gold_doc_ids(example) -> set[str]:  # noqa: ANN001
    """HoVer supporting_facts -> {f"{title}::{sent_idx}", ...}."""
    return {f"{title}::{sid}" for title, sid in example.supporting_facts}


def _evaluate_dense_or_bm25(retriever, examples, name: str) -> dict:  # noqa: ANN001
    from src.eval.metrics import bootstrap_ci, hit_rate_at_k, recall_at_k

    per_recalls: dict[int, list[float]] = {5: [], 10: [], 20: []}
    per_hits: dict[int, list[float]] = {5: [], 10: [], 20: []}
    t0 = time.time()
    for i, ex in enumerate(examples):
        if i % 50 == 0 and i > 0:
            logger.info("[{}] {}/{}", name, i, len(examples))
        gold = _gold_doc_ids(ex)
        if not gold:
            continue
        results = retriever.retrieve(ex.claim, top_k=20)
        retrieved_ids = [p.doc_id for p in results]
        for k in (5, 10, 20):
            per_recalls[k].append(recall_at_k(retrieved_ids, gold, k))
            per_hits[k].append(hit_rate_at_k(retrieved_ids, gold, k))
    elapsed = time.time() - t0

    out: dict = {"n_examples_with_gold": len(per_recalls[5]), "elapsed_s": round(elapsed, 1)}
    for k, vals in per_recalls.items():
        point, lo, hi = bootstrap_ci(vals)
        out[f"recall_at_{k}"] = {
            "point": round(point, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        }
    for k, vals in per_hits.items():
        point, lo, hi = bootstrap_ci(vals)
        out[f"hit_at_{k}"] = {
            "point": round(point, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        }
    return out


def _evaluate_with_reranker(retriever, reranker, examples, name: str) -> dict:  # noqa: ANN001
    from src.eval.metrics import bootstrap_ci, hit_rate_at_k, recall_at_k

    per_recalls: dict[int, list[float]] = {5: [], 10: [], 20: []}
    per_hits: dict[int, list[float]] = {5: [], 10: [], 20: []}
    t0 = time.time()
    for i, ex in enumerate(examples):
        if i % 50 == 0 and i > 0:
            logger.info("[{}] {}/{}", name, i, len(examples))
        gold = _gold_doc_ids(ex)
        if not gold:
            continue
        candidates = retriever.retrieve(ex.claim, top_k=50)
        reranked = reranker.rerank(ex.claim, candidates, top_k=20)
        retrieved_ids = [p.doc_id for p in reranked]
        for k in (5, 10, 20):
            per_recalls[k].append(recall_at_k(retrieved_ids, gold, k))
            per_hits[k].append(hit_rate_at_k(retrieved_ids, gold, k))
    elapsed = time.time() - t0

    out: dict = {"n_examples_with_gold": len(per_recalls[5]), "elapsed_s": round(elapsed, 1)}
    for k, vals in per_recalls.items():
        point, lo, hi = bootstrap_ci(vals)
        out[f"recall_at_{k}"] = {
            "point": round(point, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        }
    for k, vals in per_hits.items():
        point, lo, hi = bootstrap_ci(vals)
        out[f"hit_at_{k}"] = {
            "point": round(point, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        }
    return out


def _summary_row(name: str, metrics: dict) -> str:
    r10 = metrics["recall_at_10"]
    h10 = metrics["hit_at_10"]
    r20 = metrics["recall_at_20"]
    h20 = metrics["hit_at_20"]
    return (
        f"  {name:14s}  R@10={r10['point']:.3f} (±{(r10['ci_hi']-r10['ci_lo'])/2:.3f})  "
        f"H@10={h10['point']:.3f} (±{(h10['ci_hi']-h10['ci_lo'])/2:.3f})  "
        f"R@20={r20['point']:.3f} (±{(r20['ci_hi']-r20['ci_lo'])/2:.3f})  "
        f"H@20={h20['point']:.3f}  ({metrics['elapsed_s']:.0f}s)"
    )


@app.command()
def main(
    n: int = typer.Option(500, help="Number of HoVer-dev examples to eval (0 = full)."),
    seed: int = typer.Option(42),
) -> None:
    """Run the Phase 04 retrieval baseline."""
    from src.config import PipelineConfig
    from src.data.load import load_hover
    from src.reranker.cross_encoder import CrossEncoderReranker
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import DenseRetriever
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(seed)
    setup_logging()

    cfg = PipelineConfig.load(ROOT / "configs" / "default.yaml")

    splits = load_hover()
    dev_key = "validation" if "validation" in splits else "dev"
    dev = [ex for ex in splits[dev_key] if ex.supporting_facts]
    rng = random.Random(seed)
    if n and len(dev) > n:
        dev = rng.sample(dev, n)
    logger.info("evaluating {} HoVer-dev examples (seed={})", len(dev), seed)

    bm25 = BM25Retriever(cfg.corpus, top_k=50)
    dense = DenseRetriever(cfg.retriever, cfg.corpus)
    reranker = CrossEncoderReranker(cfg.reranker)

    results: dict[str, dict] = {}
    results["bm25"] = _evaluate_dense_or_bm25(bm25, dev, "bm25")
    logger.info(_summary_row("bm25", results["bm25"]))

    results["dense"] = _evaluate_dense_or_bm25(dense, dev, "dense")
    logger.info(_summary_row("dense", results["dense"]))

    results["dense+rerank"] = _evaluate_with_reranker(dense, reranker, dev, "dense+rerank")
    logger.info(_summary_row("dense+rerank", results["dense+rerank"]))

    out = {
        "summary": {
            "n_examples_evaluated": len(dev),
            "seed": seed,
            "models": {
                "dense_encoder": cfg.retriever.encoder,
                "reranker": cfg.reranker.model,
            },
        },
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("wrote {}", RESULT_PATH)
    logger.info("=== summary ===")
    for k in ("bm25", "dense", "dense+rerank"):
        logger.info(_summary_row(k, results[k]))


if __name__ == "__main__":
    app()
