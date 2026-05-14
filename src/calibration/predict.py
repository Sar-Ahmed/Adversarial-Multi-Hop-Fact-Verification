"""Load + apply the NEI calibrator at inference time.

Used by the pipeline (Phase 08 wire-in) and by the calibration eval script.
The calibrator is optional — if the joblib file is missing, callers should
fall back to the raw verifier output (no silent default predictions).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.schema import Label

_LABELS = ("SUPPORTED", "REFUTED", "NEI")


class NEICalibrator:
    """Wraps the joblib pipeline + decision-threshold rule."""

    def __init__(self, checkpoint_path: str | Path, decision_threshold: float = 0.5) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"calibrator checkpoint not found: {path}")
        payload = joblib.load(path)
        self.pipeline = payload["pipeline"]
        self.feature_names: list[str] = payload["feature_names"]
        self.classes_: list[str] = payload["classes_"]
        self.decision_threshold = decision_threshold

    def predict(self, features: np.ndarray) -> tuple[Label, float, dict[str, float]]:
        """Return (verdict, confidence, per-class probs).

        If max prob < decision_threshold, the verdict is forced to NEI — this
        is the spec's calibrated-confidence rule (Phase 08).
        """
        probs = self.pipeline.predict_proba(features.reshape(1, -1))[0]
        class_probs = dict(zip(self.classes_, probs.tolist(), strict=True))
        # Build canonical SUPPORTED/REFUTED/NEI dict so callers don't depend
        # on the sklearn class ordering.
        canonical: dict[str, float] = {lbl: float(class_probs.get(lbl, 0.0)) for lbl in _LABELS}

        verdict_str = max(canonical, key=canonical.get)
        max_prob = canonical[verdict_str]
        if max_prob < self.decision_threshold:
            return Label.NEI, max_prob, canonical
        return Label(verdict_str), max_prob, canonical
