"""Phase 14 — aggregate human evaluation ratings.

Reads:  artifacts/human_eval_sample.csv  (filled by reviewer; 5 rating cols 1-5)
Writes: artifacts/human_eval_summary.json — per-dimension mean + bootstrap 95% CI,
        per-class breakdown, and correlation with prediction-correctness.

The summary deliberately doesn't editorialise — narrative interpretation goes
into `docs/PHASE_14_human_eval.md`.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "artifacts" / "human_eval_sample.csv"
OUT_JSON = ROOT / "artifacts" / "human_eval_summary.json"

DIMENSIONS = ["decomposition", "citations", "reasoning", "faithfulness", "overall"]

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (lo, hi)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(set(xs)) == 1 or len(set(ys)) == 1:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


@app.command()
def main(seed: int = typer.Option(42)) -> None:
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    setup_logging()
    set_global_seed(seed)

    rows: list[dict] = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # Skip unrated rows (reviewer in progress)
            try:
                ratings = {d: int(r[d]) for d in DIMENSIONS}
            except (ValueError, KeyError):
                continue
            r["_ratings"] = ratings
            rows.append(r)
    logger.info("loaded {} rated chains from {}", len(rows), CSV_PATH)
    if not rows:
        raise typer.Exit(code=1)

    # Per-dimension stats
    per_dim: dict[str, dict] = {}
    for d in DIMENSIONS:
        vals = [float(r["_ratings"][d]) for r in rows]
        mean = sum(vals) / len(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lo, hi = _bootstrap_ci(vals, seed=seed)
        per_dim[d] = {
            "mean": round(mean, 3),
            "stdev": round(sd, 3),
            "ci95": [round(lo, 3), round(hi, 3)],
            "median": statistics.median(vals),
            "n": len(vals),
        }

    # Correctness split
    correct_idx = [i for i, r in enumerate(rows) if r["gold_label"] == r["predicted_verdict"]]
    incorrect_idx = [i for i, r in enumerate(rows) if r["gold_label"] != r["predicted_verdict"]]
    by_correct: dict[str, dict] = {}
    for d in DIMENSIONS:
        by_correct[d] = {
            "correct": {
                "n": len(correct_idx),
                "mean": round(sum(float(rows[i]["_ratings"][d]) for i in correct_idx) / max(len(correct_idx), 1), 3),
            },
            "incorrect": {
                "n": len(incorrect_idx),
                "mean": round(sum(float(rows[i]["_ratings"][d]) for i in incorrect_idx) / max(len(incorrect_idx), 1), 3),
            },
        }

    # Correlation between dimensions and correctness (0/1)
    correctness = [1.0 if r["gold_label"] == r["predicted_verdict"] else 0.0 for r in rows]
    corr_with_correctness = {
        d: round(_pearson([float(r["_ratings"][d]) for r in rows], correctness), 3) for d in DIMENSIONS
    }

    # Per-gold-label breakdown
    by_gold: dict[str, dict[str, float]] = {}
    for label in ("SUPPORTED", "REFUTED"):
        idx = [i for i, r in enumerate(rows) if r["gold_label"] == label]
        if not idx:
            continue
        by_gold[label] = {
            "n": len(idx),
            **{
                d: round(sum(float(rows[i]["_ratings"][d]) for i in idx) / len(idx), 3) for d in DIMENSIONS
            },
        }

    # Distribution histogram per dimension
    histograms: dict[str, dict[int, int]] = {}
    for d in DIMENSIONS:
        hist: dict[int, int] = dict.fromkeys(range(1, 6), 0)
        for r in rows:
            hist[r["_ratings"][d]] += 1
        histograms[d] = hist

    summary = {
        "n_rated": len(rows),
        "rater": "single (self-rated by the developer per Phase 14 protocol)",
        "scale": "1-5 Likert; 5 = excellent, 1 = unusable",
        "per_dimension": per_dim,
        "histograms": histograms,
        "by_correctness": by_correct,
        "by_gold_label": by_gold,
        "correlation_with_correctness": corr_with_correctness,
        "n_correct": len(correct_idx),
        "n_incorrect": len(incorrect_idx),
        "accuracy_in_sample": round(len(correct_idx) / len(rows), 3),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote {}", OUT_JSON)
    for d in DIMENSIONS:
        s = per_dim[d]
        logger.info("  {}: mean={} ci95={}", d, s["mean"], s["ci95"])


if __name__ == "__main__":
    app()
