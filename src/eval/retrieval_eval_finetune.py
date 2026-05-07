"""Phase 05 fine-tune-vs-base retrieval eval.

Runs the same metrics as `retrieval_eval.py` (recall@K + hit-rate@K with
bootstrap CIs), but compares two configurations:

  - dense (base)         — bge-small-en-v1.5 over corpus.faiss
  - dense (fine-tuned)   — checkpoints/bge-small-v3-hn over corpus_ft.faiss

The decision rule (recorded in docs/PHASE_05_DECISION.md):
  ship the model with the higher HoVer-dev R@10 with non-overlapping CI
  vs the other; if CIs overlap, ship the simpler model (base) and note
  the negative result.

Output: artifacts/retriever_eval_finetune.json
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "artifacts" / "retriever_eval_finetune.json"
DEFAULT_FT_MODEL = "checkpoints/bge-small-v3-hn"
DEFAULT_FT_INDEX = "artifacts/corpus_ft.faiss"

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _gold_doc_ids(example) -> set[str]:  # noqa: ANN001
    return {f"{title}::{sid}" for title, sid in example.supporting_facts}


def _eval_dense(retriever, examples, name: str) -> dict:  # noqa: ANN001
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
        retrieved_ids = [p.doc_id for p in retriever.retrieve(ex.claim, top_k=20)]
        for k in (5, 10, 20):
            per_recalls[k].append(recall_at_k(retrieved_ids, gold, k))
            per_hits[k].append(hit_rate_at_k(retrieved_ids, gold, k))
    elapsed = time.time() - t0

    out: dict = {"n_examples_with_gold": len(per_recalls[5]), "elapsed_s": round(elapsed, 1)}
    for k, vals in per_recalls.items():
        p, lo, hi = bootstrap_ci(vals)
        out[f"recall_at_{k}"] = {"point": round(p, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
    for k, vals in per_hits.items():
        p, lo, hi = bootstrap_ci(vals)
        out[f"hit_at_{k}"] = {"point": round(p, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
    return out


def _decide(base: dict, ft: dict) -> dict:
    """Apply the Phase 05 decision rule to base vs fine-tune R@10 with CIs."""
    base_r10 = base["recall_at_10"]
    ft_r10 = ft["recall_at_10"]
    base_h10 = base["hit_at_10"]
    ft_h10 = ft["hit_at_10"]

    # Check non-overlap: ft wins if ft.lo > base.hi; base wins if base.lo > ft.hi.
    ft_wins_recall = ft_r10["ci_lo"] > base_r10["ci_hi"]
    base_wins_recall = base_r10["ci_lo"] > ft_r10["ci_hi"]
    ft_wins_hit = ft_h10["ci_lo"] > base_h10["ci_hi"]
    base_wins_hit = base_h10["ci_lo"] > ft_h10["ci_hi"]

    if ft_wins_recall or ft_wins_hit:
        return {
            "winner": "fine-tune",
            "reason": "fine-tune CI is above base on at least one metric",
        }
    if base_wins_recall or base_wins_hit:
        return {
            "winner": "base",
            "reason": "base CI is above fine-tune on at least one metric (negative fine-tune result)",
        }
    # CIs overlap on both metrics — ship the simpler model.
    return {
        "winner": "base",
        "reason": "CIs overlap; defaulting to base (simpler model). Fine-tune kept for ablation only.",
    }


@app.command()
def main(
    n: int = typer.Option(200, help="Number of HoVer-dev examples (0 = full)."),
    seed: int = typer.Option(42),
    ft_model: str = typer.Option(DEFAULT_FT_MODEL),
    ft_index: str = typer.Option(DEFAULT_FT_INDEX),
) -> None:
    """Compare base vs fine-tuned dense retriever on HoVer dev."""
    from src.config import PipelineConfig
    from src.data.load import load_hover
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

    if not Path(ft_model).exists():
        raise FileNotFoundError(
            f"fine-tune checkpoint not found: {ft_model}. "
            "Run training in notebooks/phase05_finetune_retriever.ipynb first."
        )
    if not Path(ft_index).exists():
        raise FileNotFoundError(
            f"fine-tune FAISS index not found: {ft_index}. "
            "Run encode_corpus with --suffix=_ft and the fine-tune model."
        )

    base = DenseRetriever(cfg.retriever, cfg.corpus)  # uses cfg.retriever.encoder + corpus.faiss
    ft = DenseRetriever(
        cfg.retriever,
        cfg.corpus,
        model_override=ft_model,
        index_override=ft_index,
    )

    results: dict[str, dict] = {}
    results["dense_base"] = _eval_dense(base, dev, "dense_base")
    results["dense_finetune"] = _eval_dense(ft, dev, "dense_finetune")
    decision = _decide(results["dense_base"], results["dense_finetune"])

    out = {
        "summary": {
            "n_examples_evaluated": len(dev),
            "seed": seed,
            "base_model": cfg.retriever.encoder,
            "finetune_model": ft_model,
            "finetune_index": ft_index,
        },
        "results": results,
        "decision": decision,
    }
    RESULT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("wrote {}", RESULT_PATH)
    logger.info("=== summary ===")
    for name, r in results.items():
        r10 = r["recall_at_10"]
        h10 = r["hit_at_10"]
        logger.info(
            "  {:14s}  R@10={:.3f} ({:.3f}-{:.3f})  H@10={:.3f} ({:.3f}-{:.3f})",
            name,
            r10["point"],
            r10["ci_lo"],
            r10["ci_hi"],
            h10["point"],
            h10["ci_lo"],
            h10["ci_hi"],
        )
    logger.info("=== decision: {} ===  reason: {}", decision["winner"], decision["reason"])


if __name__ == "__main__":
    app()
