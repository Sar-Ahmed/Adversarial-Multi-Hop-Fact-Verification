"""Train the NEI calibrator (logistic regression on engineered features).

Reads:  artifacts/calibration_features_fever_train.parquet
Writes: checkpoints/nei_classifier.joblib   (sklearn Pipeline: StandardScaler + LogReg)
        artifacts/calibration_training_log.json (hyperparameters + CV metrics)

Run:    python -m src.calibration.train
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FEATURES = ROOT / "artifacts" / "calibration_features_fever_train.parquet"
CHECKPOINT = ROOT / "checkpoints" / "nei_classifier.joblib"
LOG_PATH = ROOT / "artifacts" / "calibration_training_log.json"

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    c_values: str = typer.Option("0.1,1.0,10.0", help="Comma-separated C grid for LR."),
    seed: int = typer.Option(42),
) -> None:
    """Train + cross-validate the calibrator, save the best model."""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from src.calibration.features import FEATURE_NAMES
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(seed)
    setup_logging()

    if not TRAIN_FEATURES.exists():
        raise FileNotFoundError(f"{TRAIN_FEATURES} not found — run build_features.py first")

    df = pd.read_parquet(TRAIN_FEATURES)
    # `X` and `y` are the standard sklearn convention for design-matrix /
    # label-vector; keep the uppercase to match every textbook example.
    X = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)  # noqa: N806
    y = df["label"].to_numpy()
    logger.info("loaded {} rows; label dist = {}", len(df), dict(df["label"].value_counts()))

    cs = [float(x) for x in c_values.split(",")]
    best_c = None
    best_score = -1.0
    cv_results: list[dict] = []
    for c in cs:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=2000,
                        C=c,
                        class_weight="balanced",
                        multi_class="multinomial",
                        random_state=seed,
                    ),
                ),
            ]
        )
        scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro", n_jobs=1)
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        logger.info("C={}: 5-fold macro-F1 = {:.4f} ± {:.4f}", c, mean, std)
        cv_results.append({"C": c, "macro_f1_mean": mean, "macro_f1_std": std})
        if mean > best_score:
            best_score = mean
            best_c = c

    logger.info("best C = {} (mean macro-F1 = {:.4f})", best_c, best_score)

    final = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    C=best_c,
                    class_weight="balanced",
                    multi_class="multinomial",
                    random_state=seed,
                ),
            ),
        ]
    )
    t0 = time.time()
    final.fit(X, y)
    elapsed = time.time() - t0

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": final, "feature_names": list(FEATURE_NAMES), "classes_": list(final.classes_)},
        CHECKPOINT,
    )
    logger.info("saved {} ({:.1f}s training)", CHECKPOINT, elapsed)

    log = {
        "n_train": int(len(df)),
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "seed": seed,
        "C_grid": cs,
        "best_C": best_c,
        "cv_macro_f1": best_score,
        "cv_per_C": cv_results,
        "label_distribution": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "training_time_s": round(elapsed, 2),
    }
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    logger.info("wrote {}", LOG_PATH)


if __name__ == "__main__":
    app()
