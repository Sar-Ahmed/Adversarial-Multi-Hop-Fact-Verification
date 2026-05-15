"""Phase 11 — final evaluation framework with bootstrap CIs.

Computes the headline numbers from cached artifacts produced in earlier phases:

  - Whole-claim mode metrics    ← artifacts/per_subclaim_traces.jsonl    (Phase 07)
  - Decomposed mode metrics     ← artifacts/evidence_chains.jsonl        (Phase 10)
  - FEVER cross-dataset metrics ← artifacts/calibration_eval.json        (Phase 08)

The most important decision Phase 11 records is **which aggregation mode
(whole-claim vs decomposed) is the production default**. Phase 10 closed
with a 22-point accuracy gap between the two on the same n=200 eval set;
Phase 11 makes that call explicit and saves the underlying numbers with
bootstrap 95% CIs.

Adversarial mode is a follow-up (Phase 12 ablation). Phase 11 fast-paths the
clean-mode headline since all the data is already on disk.

Reads:  artifacts/per_subclaim_traces.jsonl, evidence_chains.jsonl,
        calibration_eval.json
Writes: artifacts/eval_main.json
        artifacts/per_class_breakdown.json

Run: python -m src.eval.run_eval
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
TRACES_PATH = ROOT / "artifacts" / "per_subclaim_traces.jsonl"
CHAINS_PATH = ROOT / "artifacts" / "evidence_chains.jsonl"
CALIB_EVAL_PATH = ROOT / "artifacts" / "calibration_eval.json"
RESULT_PATH = ROOT / "artifacts" / "eval_main.json"
PER_CLASS_PATH = ROOT / "artifacts" / "per_class_breakdown.json"

_LABEL_ORDER = ("SUPPORTED", "REFUTED", "NEI")

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _git_sha() -> str:
    """Return the current git SHA so eval JSONs are traceable to a commit."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True)
        return out.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Macro-F1 + per-class metrics + bootstrap accuracy CI + confusion matrix."""
    from sklearn.metrics import classification_report

    from src.eval.metrics import bootstrap_ci

    if not y_true:
        return {"n": 0}

    correct = [1.0 if y == p else 0.0 for y, p in zip(y_true, y_pred, strict=True)]
    acc_p, acc_lo, acc_hi = bootstrap_ci(correct)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(_LABEL_ORDER),
        output_dict=True,
        zero_division=0,
    )

    per_class: dict[str, dict] = {}
    for cls in _LABEL_ORDER:
        per_class[cls] = {
            "precision": round(float(report[cls]["precision"]), 4),
            "recall": round(float(report[cls]["recall"]), 4),
            "f1": round(float(report[cls]["f1-score"]), 4),
            "support": int(report[cls]["support"]),
        }

    cm: dict[str, dict[str, int]] = {t: {p: 0 for p in _LABEL_ORDER} for t in _LABEL_ORDER}
    for t, p in zip(y_true, y_pred, strict=True):
        if t in cm and p in cm[t]:
            cm[t][p] += 1

    return {
        "n": int(len(y_true)),
        "accuracy": {
            "point": round(acc_p, 4),
            "ci_lo": round(acc_lo, 4),
            "ci_hi": round(acc_hi, 4),
        },
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def _whole_claim_metrics() -> dict:
    """Metrics in whole-claim mode (Phase 07 traces, re-aggregated through bidir).

    Phase 07's traces cached `llm_only` and `llm_plus_nli_veto` mode_results.
    The `llm_plus_nli_bidir` mode (and the current production rule) is applied
    inline here from the cached LLM + NLI signals — no need to re-run the LLM.
    """
    from src.config import PipelineConfig
    from src.schema import Label
    from src.verifier.aggregate import aggregate

    cfg = PipelineConfig.load(ROOT / "configs" / "default.yaml")

    rows: list[dict] = []
    with open(TRACES_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    y_true: list[str] = []
    y_pred: list[str] = []
    for r in rows:
        verdict, _conf, _reason = aggregate(
            mode="llm_plus_nli_bidir",
            llm_verdict=Label(r["llm"]["verdict"]),
            llm_confidence=r["llm"]["confidence"],
            llm_reasoning=r["llm"]["reasoning"],
            nli_scores=r["nli"],
            contra_veto_threshold=cfg.verifier.contra_veto_threshold,
            entail_threshold=cfg.verifier.entail_threshold,
        )
        y_true.append(r["gold_label"])
        y_pred.append(verdict.value)
    return _metrics(y_true, y_pred)


def _decomposed_metrics() -> dict:
    """Metrics in decomposed mode (Phase 10 chains, current production)."""
    rows: list[dict] = []
    with open(CHAINS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    y_true = [r["gold_label"] for r in rows]
    y_pred = [r["final_verdict"] for r in rows]
    return _metrics(y_true, y_pred)


def _fever_metrics() -> dict:
    """Cross-dataset metrics on FEVER dev (calibrator output, Phase 08)."""
    payload = json.loads(CALIB_EVAL_PATH.read_text(encoding="utf-8"))
    return payload["results"]["fever_dev"]


@app.command()
def main() -> None:
    """Compute Phase 11 headline metrics from cached phase outputs."""
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(42)
    setup_logging()

    if not TRACES_PATH.exists():
        raise FileNotFoundError(f"{TRACES_PATH} not found — re-run Phase 07")
    if not CHAINS_PATH.exists():
        raise FileNotFoundError(f"{CHAINS_PATH} not found — re-run Phase 10")
    if not CALIB_EVAL_PATH.exists():
        raise FileNotFoundError(f"{CALIB_EVAL_PATH} not found — re-run Phase 08")

    logger.info("computing Phase 11 metrics from cached artifacts")
    whole = _whole_claim_metrics()
    decomp = _decomposed_metrics()
    fever = _fever_metrics()

    # Pick production default by best macro-F1 on the binary HoVer task
    if whole["macro_f1"] >= decomp["macro_f1"]:
        recommended = "whole_claim"
        rationale = (
            f"whole-claim macro-F1 ({whole['macro_f1']}) >= "
            f"decomposed macro-F1 ({decomp['macro_f1']})"
        )
    else:
        recommended = "decomposed"
        rationale = "decomposed macro-F1 beats whole-claim"

    summary = {
        "git_sha": _git_sha(),
        "seed": 42,
        "n_bootstrap": 1000,
        "eval_splits": {
            "hover_dev": {"n": whole["n"]},
            "fever_dev": {"n": fever.get("n", 0)},
        },
        "production_recommendation": {
            "aggregation_mode": recommended,
            "rationale": rationale,
        },
    }

    payload = {
        "summary": summary,
        "hover_dev_whole_claim": whole,
        "hover_dev_decomposed": decomp,
        "fever_dev_calibrated": fever,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    PER_CLASS_PATH.write_text(
        json.dumps(
            {
                "hover_dev_whole_claim": whole["per_class"],
                "hover_dev_decomposed": decomp["per_class"],
                "fever_dev_calibrated": fever.get("per_class", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("=== Phase 11 headline ===")
    logger.info(
        "  HoVer-dev whole-claim   n={}  acc={:.3f} [{:.3f}, {:.3f}]  macro-F1={:.3f}",
        whole["n"],
        whole["accuracy"]["point"],
        whole["accuracy"]["ci_lo"],
        whole["accuracy"]["ci_hi"],
        whole["macro_f1"],
    )
    logger.info(
        "  HoVer-dev decomposed    n={}  acc={:.3f} [{:.3f}, {:.3f}]  macro-F1={:.3f}",
        decomp["n"],
        decomp["accuracy"]["point"],
        decomp["accuracy"]["ci_lo"],
        decomp["accuracy"]["ci_hi"],
        decomp["macro_f1"],
    )
    fa = fever.get("accuracy", {})
    logger.info(
        "  FEVER-dev calibrated    n={}  acc={:.3f} [{:.3f}, {:.3f}]  macro-F1={:.3f}  NEI-recall={:.3f}",
        fever.get("n", 0),
        fa.get("point", 0),
        fa.get("ci_lo", 0),
        fa.get("ci_hi", 0),
        fever.get("macro_f1", 0),
        fever.get("per_class", {}).get("NEI", {}).get("recall", 0),
    )
    logger.info("  recommended mode: {} ({})", recommended, rationale)
    logger.info("wrote {} and {}", RESULT_PATH, PER_CLASS_PATH)


if __name__ == "__main__":
    app()
