"""Phase 10 — run the full pipeline over HoVer dev and save every
EvidenceChain as JSONL plus 10 stratified rendered examples.

Reads:  cfg + Phase 01 corpus + Phase 03 Qwen + Phase 04 reranker + Phase 08
        calibrator (if checkpoint present)
Writes:
  artifacts/evidence_chains.jsonl
  artifacts/evidence_chain_render_examples.txt

The eval has resume capability — a system reboot mid-run reads the existing
JSONL, skips done UIDs, and appends new ones (same pattern as Phase 07).

Run: python -m src.eval.build_chains --n 200
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
CHAINS_PATH = ROOT / "artifacts" / "evidence_chains.jsonl"
EXAMPLES_PATH = ROOT / "artifacts" / "evidence_chain_render_examples.txt"

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    n: int = typer.Option(200, help="HoVer-dev claims (0 = all)."),
    seed: int = typer.Option(42),
) -> None:
    """Build evidence chains for the HoVer-dev eval slice."""
    from src.config import PipelineConfig
    from src.data.load import load_hover
    from src.evidence.chain import to_jsonable, validate
    from src.pipeline import build_pipeline
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(seed)
    setup_logging()

    cfg = PipelineConfig.load(ROOT / "configs" / "default.yaml")

    splits = load_hover()
    dev = [ex for ex in splits["validation"] if ex.supporting_facts]
    rng = random.Random(seed)
    if n and len(dev) > n:
        dev = rng.sample(dev, n)
    logger.info("building chains for {} HoVer-dev claims (seed={})", len(dev), seed)

    # Resume support: read existing JSONL, build seen-uid set, skip those.
    already: set[str] = set()
    CHAINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CHAINS_PATH.exists():
        with open(CHAINS_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                already.add(json.loads(line)["uid"])
        logger.info("resuming: {} chains already on disk", len(already))

    todo = [ex for ex in dev if ex.uid not in already]
    logger.info("{} claims still to build ({} done)", len(todo), len(already))

    pipeline = build_pipeline(cfg)

    invalid_count = 0
    t0 = time.time()
    with open(CHAINS_PATH, "a", encoding="utf-8") as fh:
        for i, ex in enumerate(todo):
            if i % 5 == 0 and i > 0:
                rate = i / max(time.time() - t0, 1e-6)
                eta = (len(todo) - i) / max(rate, 1e-6)
                logger.info(
                    "[{}/{} remaining] ({:.2f} cl/s, ETA {:.0f}s)",
                    i,
                    len(todo),
                    rate,
                    eta,
                )

            chain = pipeline.verify(ex.claim)
            validation = validate(chain)
            if not validation.is_valid:
                invalid_count += 1
                logger.warning("uid={} INVALID: {}", ex.uid[:8], validation.errors[:3])

            row = {
                "uid": ex.uid,
                "gold_label": ex.label,
                "num_hops": ex.num_hops,
                "validation_errors": list(validation.errors),
                **to_jsonable(chain),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

    elapsed = time.time() - t0
    logger.info("=== chain build done in {:.1f}s ===", elapsed)
    logger.info("invalid chains: {} / {}", invalid_count, len(todo))

    # Write stratified rendered examples (3 SUPPORTED, 3 REFUTED, 4 NEI-ish)
    _write_rendered_examples()


def _write_rendered_examples() -> None:
    """Stratified-sample 10 chains and write rendered text for the final report."""
    from src.evidence.chain import from_jsonable

    rows: list[dict] = []
    with open(CHAINS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    by_verdict: dict[str, list[dict]] = {"SUPPORTED": [], "REFUTED": [], "NEI": []}
    for r in rows:
        v = r["final_verdict"]
        if v in by_verdict:
            by_verdict[v].append(r)

    sample: list[dict] = []
    sample.extend(by_verdict["SUPPORTED"][:3])
    sample.extend(by_verdict["REFUTED"][:3])
    sample.extend(by_verdict["NEI"][:4])

    out_lines: list[str] = []
    for r in sample:
        out_lines.append(
            f"=== uid={r['uid'][:8]}  gold={r['gold_label']}  predicted={r['final_verdict']} ==="
        )
        chain = from_jsonable(r)
        out_lines.append(chain.render_text())
        out_lines.append("")
        out_lines.append("---")
        out_lines.append("")

    EXAMPLES_PATH.write_text("\n".join(out_lines), encoding="utf-8")
    logger.info("wrote {} rendered examples to {}", len(sample), EXAMPLES_PATH)


if __name__ == "__main__":
    app()
