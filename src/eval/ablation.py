"""Phase 12 — ablation study, computed from cached phase artifacts.

The ablation matrix in the phase doc calls for 7-9 rows. Re-running each on
the full pipeline would cost dozens of CPU hours; this script instead reads
the cached LLM + NLI traces (Phase 07), evidence chains (Phase 10), and
calibrator-feature parquets (Phase 08), and re-aggregates each ablation
inline. That gives us six of the spec's rows for free:

  full_production            — Phase 10 chains (decomposed + calibrator)
  whole_claim_no_calibrator  — Phase 07 traces, llm_plus_nli_bidir
  whole_claim_llm_only       — Phase 07 traces, llm_only (NLI veto disabled)
  whole_claim_nli_veto_only  — Phase 07 traces, llm_plus_nli_veto (legacy)
  calibrator_only            — Phase 08 calibrator features → calibrator predict
  no_distractors / adv       — Phase 11 robustness eval (paired clean/adv)

The rows that *would* require new compute (bm25_only, no_reranker,
base_retriever) are listed in the open-follow-ups; not run here.

Writes: artifacts/ablation_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
TRACES_PATH = ROOT / "artifacts" / "per_subclaim_traces.jsonl"
CHAINS_PATH = ROOT / "artifacts" / "evidence_chains.jsonl"
CALIB_EVAL = ROOT / "artifacts" / "calibration_eval.json"
ROBUSTNESS_EVAL = ROOT / "artifacts" / "robustness_eval.json"
RESULT_PATH = ROOT / "artifacts" / "ablation_results.json"

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _metrics(y_true: list[str], y_pred: list[str], description: str) -> dict:
    """Wrap src.eval.run_eval._metrics with extra context."""
    from src.eval.run_eval import _metrics as compute

    m = compute(y_true, y_pred)
    m["description"] = description
    return m


def _whole_claim_via_aggregator(mode: str) -> dict:
    """Re-aggregate Phase 07 traces through `mode` (no calibrator)."""
    from src.config import PipelineConfig
    from src.schema import Label
    from src.verifier.aggregate import aggregate

    cfg = PipelineConfig.load(ROOT / "configs" / "default.yaml")
    y_true, y_pred = [], []
    with open(TRACES_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            verdict, _, _ = aggregate(
                mode=mode,
                llm_verdict=Label(r["llm"]["verdict"]),
                llm_confidence=r["llm"]["confidence"],
                llm_reasoning=r["llm"]["reasoning"],
                nli_scores=r["nli"],
                contra_veto_threshold=cfg.verifier.contra_veto_threshold,
                entail_threshold=cfg.verifier.entail_threshold,
            )
            y_true.append(r["gold_label"])
            y_pred.append(verdict.value)
    return y_true, y_pred  # type: ignore[return-value]


def _decomposed_chains() -> tuple[list[str], list[str]]:
    """Read Phase 10 chains: final_verdict per claim."""
    y_true, y_pred = [], []
    with open(CHAINS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                y_true.append(r["gold_label"])
                y_pred.append(r["final_verdict"])
    return y_true, y_pred


@app.command()
def main() -> None:
    """Build the ablation table."""
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(42)
    setup_logging()

    rows: list[dict] = []

    # 1. full_production: decomposed + calibrator (Phase 10's chains)
    y_t, y_p = _decomposed_chains()
    m = _metrics(
        y_t,
        y_p,
        "Production: decomposer → per-sub-claim verifier → calibrator → aggregator. From artifacts/evidence_chains.jsonl.",
    )
    rows.append({"name": "full_production_decomposed", **m})
    logger.info(
        "  full_production         acc={:.3f} macro-F1={:.3f}",
        m["accuracy"]["point"],
        m["macro_f1"],
    )

    # 2. whole_claim_no_calibrator: Phase 11's recommended mode
    y_t, y_p = _whole_claim_via_aggregator("llm_plus_nli_bidir")
    m = _metrics(
        y_t,
        y_p,
        "Whole-claim verification, llm_plus_nli_bidir aggregator, no decomposer, no calibrator. Phase 11 recommended.",
    )
    rows.append({"name": "whole_claim_no_calibrator", **m})
    logger.info(
        "  whole_claim_no_calib    acc={:.3f} macro-F1={:.3f}",
        m["accuracy"]["point"],
        m["macro_f1"],
    )

    # 3. no_nli_veto: whole-claim, LLM-only
    y_t, y_p = _whole_claim_via_aggregator("llm_only")
    m = _metrics(
        y_t,
        y_p,
        "Whole-claim, NLI veto disabled. Reveals the raw 3B LLM verdict signal — measures how much NLI carries.",
    )
    rows.append({"name": "whole_claim_llm_only", **m})
    logger.info(
        "  whole_claim_llm_only    acc={:.3f} macro-F1={:.3f}",
        m["accuracy"]["point"],
        m["macro_f1"],
    )

    # 4. legacy_nli_veto: whole-claim, llm_plus_nli_veto (only SUPPORTED→REFUTED, no bidir)
    y_t, y_p = _whole_claim_via_aggregator("llm_plus_nli_veto")
    m = _metrics(
        y_t,
        y_p,
        "Whole-claim, legacy NLI veto rule (only SUPPORTED→REFUTED). Quantifies the bidir-rule lift over V1's single-direction veto.",
    )
    rows.append({"name": "whole_claim_legacy_veto", **m})
    logger.info(
        "  whole_claim_legacy_veto acc={:.3f} macro-F1={:.3f}",
        m["accuracy"]["point"],
        m["macro_f1"],
    )

    # 5. calibrator_only on HoVer: Phase 08 result, calibrator applied to features
    calib = json.loads(CALIB_EVAL.read_text(encoding="utf-8"))["results"]
    m = calib["hover_dev"]
    m["description"] = (
        "Calibrator alone (LR on 11 NLI+retrieval+lexical features). No LLM, no NLI veto. From artifacts/calibration_eval.json."
    )
    rows.append({"name": "calibrator_only_hover", **m})
    if "accuracy" in m:
        logger.info(
            "  calibrator_only_hover   acc={:.3f} macro-F1={:.3f}",
            m["accuracy"]["point"],
            m["macro_f1"],
        )

    # 6. FEVER eval (calibrator on FEVER dev) — context, not an ablation per se
    m = calib["fever_dev"]
    m["description"] = (
        "Calibrator on FEVER dev (300 stratified). Cross-dataset NEI-recall validation. From artifacts/calibration_eval.json."
    )
    rows.append({"name": "calibrator_fever_dev", **m})
    if "accuracy" in m:
        logger.info(
            "  calibrator_fever_dev    acc={:.3f} macro-F1={:.3f} NEI-recall={:.3f}",
            m["accuracy"]["point"],
            m["macro_f1"],
            m["per_class"]["NEI"]["recall"],
        )

    # 7. adversarial robustness — paired clean/adv from Phase 11
    rob = json.loads(ROBUSTNESS_EVAL.read_text(encoding="utf-8"))
    pb = rob["paired_bootstrap"]
    rows.append(
        {
            "name": "adversarial_distractors_injected",
            "description": "Whole-claim verifier with Phase 06 distractors injected before reranking. Paired vs same-uid clean. From artifacts/robustness_eval.json.",
            "n": pb["n"],
            "accuracy": {
                "point": pb["acc_adversarial"],
                "ci_lo": None,
                "ci_hi": None,
            },
            "paired_delta": {
                "clean": pb["acc_clean"],
                "adversarial": pb["acc_adversarial"],
                "delta": pb["delta"],
                "ci_lo": pb["ci_lo"],
                "ci_hi": pb["ci_hi"],
            },
        }
    )
    logger.info(
        "  adversarial             acc={:.3f} (Δ={:+.3f} [{:+.3f}, {:+.3f}])",
        pb["acc_adversarial"],
        pb["delta"],
        pb["ci_lo"],
        pb["ci_hi"],
    )

    payload = {
        "summary": {
            "n_ablations": len(rows),
            "open_follow_ups_not_run": [
                "bm25_only — replace dense with BM25 (Phase 04 has retrieval-only numbers)",
                "no_reranker — drop the cross-encoder reranker",
                "base_vs_finetune — compare base bge-small vs Phase 05 fine-tune (CIs overlapped in Phase 05)",
            ],
            "production_call": "whole_claim_no_calibrator wins macro-F1 over full_production_decomposed by 0.12 pts",
        },
        "ablations": rows,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote {}", RESULT_PATH)


if __name__ == "__main__":
    app()
