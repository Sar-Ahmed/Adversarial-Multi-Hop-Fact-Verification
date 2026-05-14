"""Phase 08 calibrator eval — apply the trained NEI classifier to feature
parquets and report per-class metrics with bootstrap CIs + a calibration
reliability diagram.

Two eval splits:
  - FEVER dev  → real 3-class metric (HoVer has no NEI gold)
  - HoVer dev  → measures whether calibration hurts/helps the binary task

Reads:  artifacts/calibration_features_{fever_dev,hover_dev}.parquet
        checkpoints/nei_classifier.joblib
Writes: artifacts/calibration_eval.json

Run:    python -m src.eval.calibration_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "artifacts" / "calibration_eval.json"
CHECKPOINT = ROOT / "checkpoints" / "nei_classifier.joblib"

_LABEL_ORDER = ["SUPPORTED", "REFUTED", "NEI"]

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _evaluate_split(split: str, decision_threshold: float) -> dict:
    """Predict over a feature parquet, compute metrics + calibration curve."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report

    from src.calibration.features import FEATURE_NAMES
    from src.calibration.predict import NEICalibrator
    from src.eval.metrics import bootstrap_ci

    feat_path = ROOT / "artifacts" / f"calibration_features_{split}.parquet"
    if not feat_path.exists():
        return {"error": f"features missing: {feat_path}"}

    df = pd.read_parquet(feat_path)
    X = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)  # noqa: N806
    y_true = df["label"].tolist()

    calibrator = NEICalibrator(CHECKPOINT, decision_threshold=decision_threshold)

    y_pred: list[str] = []
    correct: list[float] = []
    max_probs: list[float] = []
    per_class_probs: list[dict[str, float]] = []
    for i in range(len(X)):
        verdict, conf, probs = calibrator.predict(X[i])
        y_pred.append(verdict.value)
        correct.append(1.0 if verdict.value == y_true[i] else 0.0)
        max_probs.append(float(conf))
        per_class_probs.append(probs)

    acc_p, acc_lo, acc_hi = bootstrap_ci(correct)
    report = classification_report(
        y_true,
        y_pred,
        labels=_LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    per_class = {
        cls: {
            "precision": round(float(report[cls]["precision"]), 4),
            "recall": round(float(report[cls]["recall"]), 4),
            "f1": round(float(report[cls]["f1-score"]), 4),
            "support": int(report[cls]["support"]),
        }
        for cls in _LABEL_ORDER
    }

    cm = {t: {p: 0 for p in _LABEL_ORDER} for t in _LABEL_ORDER}
    for t, p in zip(y_true, y_pred, strict=True):
        if t in cm:
            cm[t][p] += 1

    # 10-bin reliability diagram: bucket predictions by confidence, compare
    # to actual accuracy in that bin.
    bins = np.linspace(0.0, 1.0, 11)
    bin_stats: list[dict] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        in_bin = [i for i, p in enumerate(max_probs) if (lo <= p < hi) or (hi == 1.0 and p == 1.0)]
        if not in_bin:
            continue
        bin_correct = float(np.mean([correct[i] for i in in_bin]))
        bin_conf = float(np.mean([max_probs[i] for i in in_bin]))
        bin_stats.append(
            {
                "lo": round(float(lo), 2),
                "hi": round(float(hi), 2),
                "n": len(in_bin),
                "mean_confidence": round(bin_conf, 4),
                "mean_accuracy": round(bin_correct, 4),
                "gap": round(bin_correct - bin_conf, 4),
            }
        )

    # ECE — weighted average of |confidence − accuracy| per bin.
    total = len(max_probs)
    ece = sum(abs(b["gap"]) * b["n"] for b in bin_stats) / total if total else 0.0

    # Brier — for each example, treat target as one-hot and compute per-class
    # squared error.  We compute the SUPPORTED-positive brier as a simple
    # numeric — full multiclass Brier averaged over classes.
    brier = 0.0
    for i, y in enumerate(y_true):
        for cls in _LABEL_ORDER:
            target = 1.0 if y == cls else 0.0
            brier += (per_class_probs[i][cls] - target) ** 2
    brier /= max(len(y_true), 1)

    return {
        "n": int(len(df)),
        "accuracy": {
            "point": round(acc_p, 4),
            "ci_lo": round(acc_lo, 4),
            "ci_hi": round(acc_hi, 4),
        },
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "ece_10bin": round(float(ece), 4),
        "brier_multiclass": round(float(brier), 4),
        "reliability_bins": bin_stats,
    }


@app.command()
def main(decision_threshold: float = typer.Option(0.5)) -> None:
    """Evaluate the calibrator on FEVER dev and HoVer dev."""
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(42)
    setup_logging()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"{CHECKPOINT} not found — run src.calibration.train first")

    results = {
        "fever_dev": _evaluate_split("fever_dev", decision_threshold),
        "hover_dev": _evaluate_split("hover_dev", decision_threshold),
    }
    payload = {
        "summary": {
            "checkpoint": str(CHECKPOINT),
            "decision_threshold": decision_threshold,
        },
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info("=== calibration eval ===")
    for split, r in results.items():
        if "accuracy" not in r:
            logger.warning("  {}: {}", split, r)
            continue
        acc = r["accuracy"]
        logger.info(
            "  {:9s}  n={:3d}  acc={:.3f} [{:.3f}, {:.3f}]  macro_f1={:.3f}  ECE={:.3f}",
            split,
            r["n"],
            acc["point"],
            acc["ci_lo"],
            acc["ci_hi"],
            r["macro_f1"],
            r["ece_10bin"],
        )
        for cls, m in r["per_class"].items():
            if m["support"] == 0:
                continue
            logger.info(
                "    {:10s} P={:.3f} R={:.3f} F1={:.3f} (support={})",
                cls,
                m["precision"],
                m["recall"],
                m["f1"],
                m["support"],
            )
    logger.info("wrote {}", RESULT_PATH)


if __name__ == "__main__":
    app()
