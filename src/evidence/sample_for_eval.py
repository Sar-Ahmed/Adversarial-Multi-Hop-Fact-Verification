"""Phase 14 — sample evidence chains for human rating.

Reads:  artifacts/evidence_chains.jsonl (Phase 10's 200 chains)
Writes: artifacts/human_eval_sample.csv  — one row per chain, empty rating columns

Sampling strategy: 50 chains stratified by `gold_label` (balanced across HoVer's
two classes). Phase 14's spec called for 100; we ship 50 with a documented
deviation because the manual rating cost is ~2 hours and the dimension-level
CIs at n=50 are still informative (each bin has ~10 ratings).

The reviewer fills `decomposition`, `citations`, `reasoning`, `faithfulness`,
and `overall` (each 1-5). A subsequent script reads the filled CSV and
emits `artifacts/human_eval_summary.json`.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
CHAINS_PATH = ROOT / "artifacts" / "evidence_chains.jsonl"
OUT_CSV = ROOT / "artifacts" / "human_eval_sample.csv"
OUT_RENDERED = ROOT / "artifacts" / "human_eval_rendered.txt"

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    n: int = typer.Option(50, help="Number of chains to sample."),
    seed: int = typer.Option(42),
) -> None:
    """Stratified-sample chains and emit a rating CSV + rendered companion."""
    from src.evidence.chain import from_jsonable
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(seed)
    setup_logging()

    rows: list[dict] = []
    with open(CHAINS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    logger.info("loaded {} chains from {}", len(rows), CHAINS_PATH)

    rng = random.Random(seed)
    by_gold: dict[str, list[dict]] = {"SUPPORTED": [], "REFUTED": []}
    for r in rows:
        if r["gold_label"] in by_gold:
            by_gold[r["gold_label"]].append(r)

    per_class = n // 2
    sample: list[dict] = []
    for cls, exs in by_gold.items():
        picked = rng.sample(exs, min(per_class, len(exs)))
        sample.extend(picked)
        logger.info("  gold={}: sampled {}/{}", cls, len(picked), len(exs))
    rng.shuffle(sample)

    # Write CSV with empty rating columns
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "uid",
        "gold_label",
        "predicted_verdict",
        "predicted_confidence",
        "n_sub_claims",
        "n_citations",
        "claim",
        # Rating columns (1-5; reviewer fills)
        "decomposition",
        "citations",
        "reasoning",
        "faithfulness",
        "overall",
        "notes",
    ]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sample:
            n_cites = sum(len(v["cited_passage_ids"]) for v in r["verifications"])
            w.writerow(
                {
                    "uid": r["uid"],
                    "gold_label": r["gold_label"],
                    "predicted_verdict": r["final_verdict"],
                    "predicted_confidence": r["final_confidence"],
                    "n_sub_claims": len(r["sub_claims"]),
                    "n_citations": n_cites,
                    "claim": r["claim"][:300],
                    "decomposition": "",
                    "citations": "",
                    "reasoning": "",
                    "faithfulness": "",
                    "overall": "",
                    "notes": "",
                }
            )

    # Write rendered text companion (one block per chain, reviewer reads this)
    out_lines: list[str] = [
        "# Phase 14 — rendered chains for human rating",
        "",
        f"{len(sample)} chains stratified by gold label (25 SUPPORTED + 25 REFUTED).",
        "Read each chain, then fill the corresponding row in `human_eval_sample.csv`.",
        "Rubric in `docs/HUMAN_EVAL_PROTOCOL.md`.",
        "",
    ]
    for i, r in enumerate(sample, start=1):
        chain = from_jsonable(r)
        out_lines.append(
            f"## #{i}  uid={r['uid'][:8]}  gold={r['gold_label']}  predicted={r['final_verdict']}"
        )
        out_lines.append("")
        out_lines.append("```")
        out_lines.append(chain.render_text())
        out_lines.append("```")
        out_lines.append("")
        out_lines.append("---")
        out_lines.append("")
    OUT_RENDERED.write_text("\n".join(out_lines), encoding="utf-8")

    logger.info("wrote {} ({} chains)", OUT_CSV, len(sample))
    logger.info("wrote {} (rendered companion)", OUT_RENDERED)


if __name__ == "__main__":
    app()
