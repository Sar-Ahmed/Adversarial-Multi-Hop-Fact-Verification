"""Phase 13 — sample 50 failures and emit a markdown template for tagging.

Reads:  artifacts/per_subclaim_traces.jsonl (Phase 07 — whole-claim mode is
        the Phase 11 recommended production)
Writes: artifacts/failures_for_tagging.md (one block per failure, with auto-
        computed metadata and an empty `category:` field for the reviewer)

The reviewer fills `category:` with one of the failure-mode tags from the
taxonomy and (optionally) `notes:`. A subsequent run of
`finalize_tagged.py` reads the markdown back and writes the structured
`failures_tagged.json` for the final report.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
TRACES_PATH = ROOT / "artifacts" / "per_subclaim_traces.jsonl"
OUT_PATH = ROOT / "artifacts" / "failures_for_tagging.md"

# Failure-mode taxonomy. Reviewer picks one per failure.
TAXONOMY = [
    "refuted_as_supported",
    "supported_as_refuted",
    "nei_miscalibration",  # pred=NEI but gold is SUP/REF (with gold retrieved)
    "retrieval_miss",  # gold passage NOT in retrieved top-10
    "entity_confusion",  # right topic, wrong entity
    "negation_blindness",  # missed an explicit negation in claim or passage
    "partial_match_as_full",  # some facts supported, others not
    "temporal_error",  # wrong year / sequel / era
    "decomposition_error",  # n/a in whole-claim mode but kept for completeness
    "other",
]

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _load_traces_with_bidir_pred() -> list[dict]:
    """Read Phase 07 traces, re-aggregate through llm_plus_nli_bidir, return
    rows with the predicted verdict attached."""
    from src.config import PipelineConfig
    from src.schema import Label
    from src.verifier.aggregate import aggregate

    cfg = PipelineConfig.load(ROOT / "configs" / "default.yaml")
    rows: list[dict] = []
    with open(TRACES_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            verdict, conf, _reason = aggregate(
                mode="llm_plus_nli_bidir",
                llm_verdict=Label(r["llm"]["verdict"]),
                llm_confidence=r["llm"]["confidence"],
                llm_reasoning=r["llm"]["reasoning"],
                nli_scores=r["nli"],
                contra_veto_threshold=cfg.verifier.contra_veto_threshold,
                entail_threshold=cfg.verifier.entail_threshold,
            )
            r["pred_verdict"] = verdict.value
            r["pred_confidence"] = conf
            rows.append(r)
    return rows


def _gold_in_top10(trace: dict, hover_gold_titles_by_uid: dict[str, set[str]]) -> bool:
    """Does any HoVer gold title appear among the trace's top-10 passage titles?"""
    titles = hover_gold_titles_by_uid.get(trace["uid"], set())
    if not titles:
        return False
    retrieved_titles = {pid.split("::")[0] for pid in trace["passage_doc_ids"]}
    return bool(titles & retrieved_titles)


def _stratified_sample(failures: list[dict], n: int, seed: int) -> list[dict]:
    """Stratify by (gold, pred) transition class, aiming for ~equal coverage."""
    rng = random.Random(seed)
    by_transition: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in failures:
        by_transition[(f["gold_label"], f["pred_verdict"])].append(f)

    # Order strata by size to fill smaller buckets first
    strata = sorted(by_transition.items(), key=lambda kv: len(kv[1]))
    per_stratum_min = max(2, n // (2 * len(strata)))  # at least 2 per non-empty stratum

    sample: list[dict] = []
    remaining_strata: list[list[dict]] = []
    for _, exs in strata:
        take = min(per_stratum_min, len(exs))
        sample.extend(rng.sample(exs, take))
        if len(exs) > take:
            remaining_strata.append([e for e in exs if e not in sample])

    # Fill the rest proportionally from remaining
    while len(sample) < n and any(remaining_strata):
        for bucket in list(remaining_strata):
            if not bucket:
                remaining_strata.remove(bucket)
                continue
            picked = rng.choice(bucket)
            sample.append(picked)
            bucket.remove(picked)
            if len(sample) >= n:
                break

    return sample[:n]


@app.command()
def main(
    n: int = typer.Option(50, help="Number of failures to sample."),
    seed: int = typer.Option(42),
) -> None:
    """Sample failures and write artifacts/failures_for_tagging.md."""
    from src.data.load import load_hover
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(seed)
    setup_logging()

    rows = _load_traces_with_bidir_pred()
    failures = [r for r in rows if r["pred_verdict"] != r["gold_label"]]
    logger.info(
        "{} failures out of {} ({:.1%})", len(failures), len(rows), len(failures) / len(rows)
    )

    # Build gold-title lookup from HoVer dev for retrieval-miss check
    hover = load_hover()
    gold_by_uid: dict[str, set[str]] = {
        ex.uid: set(ex.supporting_titles) for ex in hover["validation"] if ex.supporting_facts
    }

    for f in failures:
        f["gold_in_top10"] = _gold_in_top10(f, gold_by_uid)

    sample = _stratified_sample(failures, n, seed)
    logger.info(
        "sampled {} failures across {} (gold, pred) strata",
        len(sample),
        len({(f["gold_label"], f["pred_verdict"]) for f in sample}),
    )

    # Confusion matrix on FULL failures (for context in the doc)
    cm: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in failures:
        cm[f["gold_label"]][f["pred_verdict"]] += 1

    # Write markdown for manual tagging
    lines = [
        "# Phase 13 — failure tagging template",
        "",
        f"**Sampled {len(sample)} failures** (whole-claim mode, `llm_plus_nli_bidir`) ",
        f"stratified across (gold, pred) transitions. Seed {seed}. ",
        f"Total failures: {len(failures)} / {len(rows)} = {len(failures)/len(rows):.1%}.",
        "",
        "## Confusion matrix on ALL failures (rows=gold, cols=pred)",
        "",
        "| gold \\ pred | SUPPORTED | REFUTED | NEI |",
        "|---|---|---|---|",
    ]
    for g in ("SUPPORTED", "REFUTED", "NEI"):
        s = cm.get(g, {}).get("SUPPORTED", 0)
        r = cm.get(g, {}).get("REFUTED", 0)
        ne = cm.get(g, {}).get("NEI", 0)
        lines.append(f"| **{g}** | {s} | {r} | {ne} |")

    lines.extend(
        [
            "",
            "## Taxonomy",
            "",
            *(f"- `{t}`" for t in TAXONOMY),
            "",
            "## How to tag",
            "",
            "Fill `category:` with one tag and `notes:` with a one-line explanation. ",
            "Auto-fields (gold, pred, gold_in_top10, llm reasoning) are pre-populated.",
            "",
            "---",
            "",
        ]
    )

    for i, f in enumerate(sample, start=1):
        retrieved_titles = sorted({pid.split("::")[0] for pid in f["passage_doc_ids"][:5]})
        gold_titles = sorted(gold_by_uid.get(f["uid"], set()))
        lines.extend(
            [
                f"### #{i}  uid={f['uid'][:8]}",
                "",
                f"- **gold**: `{f['gold_label']}`",
                f"- **predicted**: `{f['pred_verdict']}` (conf={f['pred_confidence']:.2f})",
                f"- **gold_in_top10**: `{f['gold_in_top10']}`",
                f"- **num_hops**: {f['num_hops']}",
                f"- **NLI**: max_contra={f['nli']['max_contra']:.3f}, max_entail={f['nli']['max_entail']:.3f}",
                "",
                f"**Claim:** {f['claim']}",
                "",
                f"**Gold titles:** {gold_titles or '(none)'}",
                "",
                f"**Top-5 retrieved titles:** {retrieved_titles}",
                "",
                f"**LLM verdict:** `{f['llm']['verdict']}`",
                "",
                f"**LLM reasoning:** {f['llm']['reasoning']}",
                "",
                "```",
                "category: <fill in>",
                "notes: <optional>",
                "```",
                "",
                "---",
                "",
            ]
        )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote {}", OUT_PATH)
    logger.info("now fill in `category:` for each of the {} blocks", len(sample))


if __name__ == "__main__":
    app()
