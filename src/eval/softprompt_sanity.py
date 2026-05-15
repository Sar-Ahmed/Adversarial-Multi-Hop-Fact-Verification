"""Phase 16 sanity check — load the LLM once, run both prompt variants on the
same N HoVer dev claims, and report parse-ability + verdict distribution.

Faster than the full verifier_eval because it skips NLI scoring and skips
retrieval cache writes — just LLM calls on cached top-K passages from the v3.0
trace file. ~20 min on CPU for N=10 claims (LLM does both v1 and v2 per claim).

Pass criteria (recorded in PHASE_16_soft_prompt.md sanity-check section):
  - ≤ 2/N parse failures on v2
  - v2 NEI rate < v1 NEI rate (any drop)
  - v2 accuracy ≥ v1 accuracy − 0.10

Reads:  artifacts/per_subclaim_traces.jsonl (v3.0 traces — provides claim+passage_doc_ids)
Writes: artifacts/softprompt_sanity.json — per-claim verdicts for both variants + summary
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
V1_TRACES = ROOT / "artifacts" / "per_subclaim_traces.jsonl"
OUT_PATH = ROOT / "artifacts" / "softprompt_sanity.json"

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _load_passage_by_id(corpus_parquet: Path) -> dict[str, dict]:
    """Build a doc_id → {title, sent_idx, text} map from the corpus parquet."""
    import pandas as pd

    df = pd.read_parquet(corpus_parquet, columns=["doc_id", "title", "sent_idx", "text"])
    return {
        row["doc_id"]: {"title": row["title"], "sent_idx": int(row["sent_idx"]), "text": row["text"]}
        for row in df.to_dict("records")
    }


@app.command()
def main(
    n: int = typer.Option(10, help="Number of claims to test (re-used from v3.0 traces)."),
    seed: int = typer.Option(42),
) -> None:
    from src.config import PipelineConfig
    from src.schema import Passage
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed
    from src.verifier.llm import LLMVerifier

    set_global_seed(seed)
    setup_logging()

    cfg = PipelineConfig.load(ROOT / "configs" / "default.yaml")
    if not cfg.verifier.llm_path:
        raise RuntimeError("verifier.llm_path is null in configs/default.yaml")

    # Load v3.0 traces — these give us (claim, passage_doc_ids, v1 verdict, gold)
    rows: list[dict] = []
    with open(V1_TRACES, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    logger.info("loaded {} v3.0 trace rows from {}", len(rows), V1_TRACES.name)

    # Take first N (deterministic — the file's row order is the eval order)
    sample = rows[:n]
    logger.info("running v1 + v2 verifier on first {} claims", len(sample))

    # Map doc_ids back to text by reading the corpus
    corpus_path = ROOT / "artifacts" / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus not built at {corpus_path}; run `make corpus`")
    logger.info("loading corpus passages from {}", corpus_path)
    passage_text = _load_passage_by_id(corpus_path)

    # Load LLM once with v1 prompt; for v2 we'll swap the variant attribute.
    v1_verifier = LLMVerifier(llm_path=cfg.verifier.llm_path, prompt_variant="v1")
    # Reuse same llama-cpp handle for v2 by reassigning prompt_variant
    # (LocalLLM.chat() takes messages — the variant only changes the system msg).

    per_row: list[dict] = []
    t0 = time.time()
    for i, r in enumerate(sample):
        claim = r["claim"]
        doc_ids = r["passage_doc_ids"]
        passages = []
        for d in doc_ids:
            info = passage_text.get(d)
            if not info:
                continue
            passages.append(
                Passage(
                    doc_id=d,
                    title=info["title"],
                    sent_idx=info["sent_idx"],
                    text=info["text"],
                    score=0.0,
                )
            )

        # v1
        v1_verifier.prompt_variant = "v1"
        v1_label, v1_conf, v1_reason = v1_verifier.verify(claim, passages)

        # v2
        v1_verifier.prompt_variant = "v2"
        v2_label, v2_conf, v2_reason = v1_verifier.verify(claim, passages)

        per_row.append(
            {
                "uid": r["uid"],
                "claim": claim[:140],
                "gold": r["gold_label"],
                "v3_cached_v1_verdict": r["llm"]["verdict"],  # from frozen v3.0 trace
                "fresh_v1_verdict": v1_label.value,
                "fresh_v1_reason": v1_reason[:200],
                "v2_verdict": v2_label.value,
                "v2_reason": v2_reason[:200],
                "v2_parse_failed": v2_reason.startswith("parse failure"),
            }
        )
        logger.info(
            "[{}/{}] gold={} v1={} v2={} (Δ={})",
            i + 1,
            len(sample),
            r["gold_label"],
            v1_label.value,
            v2_label.value,
            "same" if v1_label is v2_label else "shift",
        )

    elapsed = time.time() - t0

    # Summary
    def _verdict_counts(key: str) -> dict:
        out = {"SUPPORTED": 0, "REFUTED": 0, "NEI": 0}
        for row in per_row:
            out[row[key]] = out.get(row[key], 0) + 1
        return out

    def _accuracy(key: str) -> float:
        return sum(1 for r in per_row if r[key] == r["gold"]) / len(per_row)

    summary = {
        "n": len(per_row),
        "elapsed_s": round(elapsed, 1),
        "v1_verdict_counts": _verdict_counts("fresh_v1_verdict"),
        "v2_verdict_counts": _verdict_counts("v2_verdict"),
        "v3_cached_v1_verdict_counts": _verdict_counts("v3_cached_v1_verdict"),
        "v1_accuracy": round(_accuracy("fresh_v1_verdict"), 3),
        "v2_accuracy": round(_accuracy("v2_verdict"), 3),
        "v3_cached_v1_accuracy": round(_accuracy("v3_cached_v1_verdict"), 3),
        "v2_parse_failures": sum(1 for r in per_row if r["v2_parse_failed"]),
        "v1_v2_verdict_shifts": sum(
            1 for r in per_row if r["fresh_v1_verdict"] != r["v2_verdict"]
        ),
    }
    payload = {"summary": summary, "per_row": per_row}
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("=== summary ===")
    logger.info("  n={}, elapsed={}s", summary["n"], summary["elapsed_s"])
    logger.info("  v1 counts:   {}", summary["v1_verdict_counts"])
    logger.info("  v2 counts:   {}", summary["v2_verdict_counts"])
    logger.info("  v1 acc={}, v2 acc={}", summary["v1_accuracy"], summary["v2_accuracy"])
    logger.info("  v2 parse failures: {}", summary["v2_parse_failures"])
    logger.info("  v1→v2 shifts: {}", summary["v1_v2_verdict_shifts"])
    logger.info("wrote {}", OUT_PATH)


if __name__ == "__main__":
    app()
