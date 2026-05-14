"""Re-aggregate per_subclaim_traces.jsonl under multiple verifier modes.

The Phase 07 LLM pass cached LLM verdict + NLI scores for every claim. This
script reads those traces and re-applies different aggregation rules to
compare modes WITHOUT re-running the LLM (which would cost ~1.5 hours).

Currently compares: `llm_only`, `llm_plus_nli_veto`, `llm_plus_nli_bidir`.

Run: python -m src.eval.reaggregate_traces
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger

from src.eval.verifier_eval import _metrics_from_traces
from src.schema import Label
from src.verifier.aggregate import aggregate

ROOT = Path(__file__).resolve().parents[2]
TRACES_PATH = ROOT / "artifacts" / "per_subclaim_traces.jsonl"
RESULT_PATH = ROOT / "artifacts" / "verifier_eval_phase07.json"

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _reaggregate_row(row: dict, mode: str, contra_thr: float, entail_thr: float) -> dict:
    """Apply `mode`'s aggregator to the cached LLM+NLI signals."""
    verdict, conf, reason = aggregate(
        mode=mode,
        llm_verdict=Label(row["llm"]["verdict"]),
        llm_confidence=row["llm"]["confidence"],
        llm_reasoning=row["llm"]["reasoning"],
        nli_scores=row["nli"],
        contra_veto_threshold=contra_thr,
        entail_threshold=entail_thr,
    )
    return {"verdict": verdict.value, "confidence": conf, "reasoning": reason}


@app.command()
def main(
    contra_threshold: float = typer.Option(0.95),
    entail_threshold: float = typer.Option(0.7),
) -> None:
    """Re-aggregate cached traces under three modes; rewrite the eval JSON."""
    if not TRACES_PATH.exists():
        raise FileNotFoundError(f"{TRACES_PATH} not found — run verifier_eval.py first")

    traces: list[dict] = []
    with open(TRACES_PATH, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                traces.append(json.loads(line))
    logger.info("loaded {} traces from {}", len(traces), TRACES_PATH)

    modes = ["llm_only", "llm_plus_nli_veto", "llm_plus_nli_bidir"]
    for tr in traces:
        for mode in modes:
            tr.setdefault("mode_results", {})[mode] = _reaggregate_row(
                tr, mode, contra_threshold, entail_threshold
            )

    results = {mode: _metrics_from_traces(traces, mode) for mode in modes}

    # Preserve the existing summary block if present
    summary: dict = {}
    if RESULT_PATH.exists():
        prior = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        summary = prior.get("summary", {})
    summary["n_claims"] = len(traces)
    summary["contra_veto_threshold"] = contra_threshold
    summary["entail_threshold"] = entail_threshold

    payload = {"summary": summary, "results": results}
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("=== summary (n={}) ===", len(traces))
    for mode, m in results.items():
        if not m or "accuracy" not in m:
            continue
        acc = m["accuracy"]
        logger.info(
            "  {:25s}  acc={:.3f} [{:.3f}, {:.3f}]  macro_f1={:.3f}",
            mode,
            acc["point"],
            acc["ci_lo"],
            acc["ci_hi"],
            m["macro_f1"],
        )
    logger.info("wrote {}", RESULT_PATH)


if __name__ == "__main__":
    app()
