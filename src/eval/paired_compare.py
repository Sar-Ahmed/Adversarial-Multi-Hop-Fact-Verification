"""Phase 16 — paired-bootstrap Δ between v1 (v3.0 production) and v2 (soft-prompt) traces.

Reads two `per_subclaim_traces*.jsonl` files (same uids, same passages, same NLI;
only the LLM prompt and verdict differ). Re-aggregates each row through the
production aggregation mode (`llm_plus_nli_bidir`) and computes:

- Per-variant accuracy + bootstrap 95% CI.
- Paired Δ (v2 − v1) + paired 95% CI by resampling the same indices both ways.
- Per-class confusion matrices.
- Verdict-shift table (v1 verdict → v2 verdict counts).

Writes: artifacts/softprompt_comparison.json

The decision rule from docs/PHASE_16_soft_prompt.md:
  - Ship v3.1 if Δ ≥ +5 pts and the paired CI doesn't include 0.
  - Scope out if Δ < +3 pts or CI overlaps 0.
  - Iterate prompt if Δ is +3-5 pts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
V1_TRACES = ROOT / "artifacts" / "per_subclaim_traces.jsonl"
V2_TRACES_DEFAULT = ROOT / "artifacts" / "per_subclaim_traces_softprompt.jsonl"
OUT_PATH = ROOT / "artifacts" / "softprompt_comparison.json"

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _load_traces(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["uid"]] = row
    return out


def _final_verdict(row: dict, cfg) -> str:  # noqa: ANN001
    """Re-aggregate one trace row through llm_plus_nli_bidir."""
    from src.schema import Label
    from src.verifier.aggregate import aggregate

    verdict, _conf, _reason = aggregate(
        mode="llm_plus_nli_bidir",
        llm_verdict=Label(row["llm"]["verdict"]),
        llm_confidence=row["llm"]["confidence"],
        llm_reasoning=row["llm"]["reasoning"],
        nli_scores=row["nli"],
        contra_veto_threshold=cfg.verifier.contra_veto_threshold,
        entail_threshold=cfg.verifier.entail_threshold,
    )
    return verdict.value


def _accuracy(predicted: list[str], gold: list[str]) -> float:
    if not predicted:
        return 0.0
    return sum(1 for p, g in zip(predicted, gold) if p == g) / len(predicted)


def _bootstrap_paired(
    v1_correct: list[int],
    v2_correct: list[int],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap: resample shared indices, compute Δ each time."""
    rng = random.Random(seed)
    n = len(v1_correct)
    if n == 0:
        return {"v1_point": 0.0, "v2_point": 0.0, "delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}

    v1_point = sum(v1_correct) / n
    v2_point = sum(v2_correct) / n
    deltas: list[float] = []
    v1_means: list[float] = []
    v2_means: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v1 = sum(v1_correct[i] for i in idx) / n
        v2 = sum(v2_correct[i] for i in idx) / n
        deltas.append(v2 - v1)
        v1_means.append(v1)
        v2_means.append(v2)
    deltas.sort()
    v1_means.sort()
    v2_means.sort()
    return {
        "v1_point": round(v1_point, 4),
        "v1_ci": [round(v1_means[int(0.025 * n_boot)], 4), round(v1_means[int(0.975 * n_boot)], 4)],
        "v2_point": round(v2_point, 4),
        "v2_ci": [round(v2_means[int(0.025 * n_boot)], 4), round(v2_means[int(0.975 * n_boot)], 4)],
        "delta": round(v2_point - v1_point, 4),
        "delta_ci": [
            round(deltas[int(0.025 * n_boot)], 4),
            round(deltas[int(0.975 * n_boot)], 4),
        ],
        "n": n,
        "n_boot": n_boot,
    }


def _confusion(predicted: list[str], gold: list[str]) -> dict:
    labels = ["SUPPORTED", "REFUTED", "NEI"]
    cm = {g: dict.fromkeys(labels, 0) for g in labels}
    for g, p in zip(gold, predicted):
        if g in cm and p in labels:
            cm[g][p] += 1
    return cm


def _shift_table(v1_pred: list[str], v2_pred: list[str]) -> dict:
    labels = ["SUPPORTED", "REFUTED", "NEI"]
    table = {v: dict.fromkeys(labels, 0) for v in labels}
    for a, b in zip(v1_pred, v2_pred):
        if a in table and b in labels:
            table[a][b] += 1
    return table


@app.command()
def main(
    v2_traces: Path = typer.Option(V2_TRACES_DEFAULT, help="Path to v2 traces jsonl."),
    seed: int = typer.Option(42),
    n_boot: int = typer.Option(1000),
) -> None:
    from src.config import PipelineConfig
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(seed)
    setup_logging()

    cfg = PipelineConfig.load(ROOT / "configs" / "default.yaml")

    if not V1_TRACES.exists():
        raise FileNotFoundError(f"{V1_TRACES} not found")
    if not v2_traces.exists():
        raise FileNotFoundError(f"{v2_traces} not found")

    v1 = _load_traces(V1_TRACES)
    v2 = _load_traces(v2_traces)
    shared = sorted(set(v1.keys()) & set(v2.keys()))
    logger.info("v1: {} rows, v2: {} rows, shared: {}", len(v1), len(v2), len(shared))
    if not shared:
        raise RuntimeError("no overlapping uids between v1 and v2 traces")

    gold: list[str] = []
    v1_pred: list[str] = []
    v2_pred: list[str] = []
    for uid in shared:
        gold.append(v1[uid]["gold_label"])
        v1_pred.append(_final_verdict(v1[uid], cfg))
        v2_pred.append(_final_verdict(v2[uid], cfg))

    v1_correct = [1 if p == g else 0 for p, g in zip(v1_pred, gold)]
    v2_correct = [1 if p == g else 0 for p, g in zip(v2_pred, gold)]

    paired = _bootstrap_paired(v1_correct, v2_correct, n_boot=n_boot, seed=seed)

    delta = paired["delta"]
    ci_lo, ci_hi = paired["delta_ci"]
    if delta >= 0.05 and ci_lo > 0:
        decision = "ship_v31"
    elif delta < 0.03 or ci_lo <= 0 <= ci_hi:
        decision = "scope_out"
    else:
        decision = "iterate_prompt"

    payload = {
        "summary": {
            "n_shared": len(shared),
            "v1_traces": str(V1_TRACES.name),
            "v2_traces": str(v2_traces.name),
            "n_boot": n_boot,
            "seed": seed,
            "decision_rule": (
                "Δ≥0.05 with paired CI>0 → ship; "
                "Δ<0.03 or CI crosses 0 → scope_out; "
                "else → iterate_prompt"
            ),
            "decision": decision,
        },
        "paired": paired,
        "v1_confusion": _confusion(v1_pred, gold),
        "v2_confusion": _confusion(v2_pred, gold),
        "verdict_shift_v1_to_v2": _shift_table(v1_pred, v2_pred),
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("=== paired comparison ===")
    logger.info("  v1 acc: {} {}", paired["v1_point"], paired["v1_ci"])
    logger.info("  v2 acc: {} {}", paired["v2_point"], paired["v2_ci"])
    logger.info("  Δ:      {} (CI {})", paired["delta"], paired["delta_ci"])
    logger.info("  decision: {}", decision)
    logger.info("wrote {}", OUT_PATH)


if __name__ == "__main__":
    app()
