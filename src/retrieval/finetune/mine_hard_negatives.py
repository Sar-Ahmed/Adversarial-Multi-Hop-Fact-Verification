"""Hard-negative mining for the Phase 05 retriever fine-tune.

For each sampled (claim, gold passages) in HoVer train + FEVER train:

  1. Run BM25 over the focused corpus (177k passages), get top-50.
  2. Filter out hits whose doc_id matches any gold passage or whose title is
     in the gold-title set (both directions of the gold relation).
  3. Skip the BM25 top-10 (V1's trick — top-10 are often near-paraphrases
     of the gold passage and pollute the negative pool with would-be positives).
  4. Sample 4 from positions 11-50 with seed=42 → 4 hard negatives per claim.
  5. Pick one random gold passage from the corpus as the positive.
  6. Emit 4 triplets per claim: (claim, positive_text, negative_text).

Output: artifacts/hard_negatives_v3.jsonl

The mix-of-datasets policy is V3's hypothesis-fix for V1's negative result:
V1 mined only from FEVER and the resulting fine-tune dropped HoVer R@10 by
~2.5 points. We mix to keep HoVer coverage intact.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = ROOT / "artifacts" / "hard_negatives_v3.jsonl"

# Sampling policy — both datasets contribute, weighted toward HoVer (the
# eval target distribution).
DEFAULT_HOVER_CLAIMS = 6000
DEFAULT_FEVER_CLAIMS = 4000
NEGATIVES_PER_CLAIM = 4
BM25_TOP_K = 50
SKIP_TOP_K = 10  # paraphrase-risk filter

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _mine_for_claim(  # noqa: PLR0913
    claim: str,
    gold_doc_ids: set[str],
    gold_titles: set[str],
    bm25_results,  # noqa: ANN001
    rng: random.Random,
    negatives_per_claim: int,
    skip_top: int,
) -> list[str] | None:
    """Return up to `negatives_per_claim` hard-negative passage texts, or None
    if the candidate pool is too small after filtering."""
    candidates = []
    for i, p in enumerate(bm25_results):
        if i < skip_top:
            continue
        if p.doc_id in gold_doc_ids or p.title in gold_titles:
            continue
        candidates.append(p.text)
    if len(candidates) < negatives_per_claim:
        return None
    return rng.sample(candidates, negatives_per_claim)


def _gold_passage_text(  # noqa: ANN001
    df, gold_doc_ids: set[str], rng: random.Random
) -> str | None:
    """Pick one random gold passage's text from the corpus parquet."""
    matched = df[df["doc_id"].isin(list(gold_doc_ids))]
    if len(matched) == 0:
        return None
    row = matched.sample(n=1, random_state=rng.randint(0, 2**31 - 1)).iloc[0]
    return str(row["text"])


@app.command()
def main(
    n_hover: int = typer.Option(DEFAULT_HOVER_CLAIMS, help="Sample size from HoVer train."),
    n_fever: int = typer.Option(DEFAULT_FEVER_CLAIMS, help="Sample size from FEVER train."),
    seed: int = typer.Option(42),
) -> None:
    """Mine hard-negative triplets and write artifacts/hard_negatives_v3.jsonl."""
    import pandas as pd

    from src.config import PipelineConfig
    from src.data.load import load_fever, load_hover
    from src.retrieval.bm25 import BM25Retriever
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(seed)
    setup_logging()

    cfg = PipelineConfig.load(ROOT / "configs" / "default.yaml")
    rng = random.Random(seed)

    # Build the candidate claim pool.
    logger.info("loading HoVer + FEVER train splits...")
    hover = load_hover()
    fever = load_fever()
    hover_train = [ex for ex in hover["train"] if ex.supporting_facts]
    fever_train = [ex for ex in fever["train"] if ex.evidence_facts]
    logger.info(
        "available: hover/train={} (with-gold), fever/train={} (with-gold)",
        len(hover_train),
        len(fever_train),
    )

    if n_hover and n_hover < len(hover_train):
        hover_train = rng.sample(hover_train, n_hover)
    if n_fever and n_fever < len(fever_train):
        fever_train = rng.sample(fever_train, n_fever)
    logger.info("sampled: hover={}, fever={}", len(hover_train), len(fever_train))

    # Normalize to a common (claim, gold_doc_ids, gold_titles) tuple.
    examples: list[tuple[str, set[str], set[str], str]] = []
    for ex in hover_train:
        gold_ids = {f"{t}::{s}" for t, s in ex.supporting_facts}
        gold_titles = set(ex.supporting_titles)
        examples.append((ex.claim, gold_ids, gold_titles, "hover"))
    for ex in fever_train:
        gold_ids = {f"{t}::{s}" for t, s in ex.evidence_facts}
        gold_titles = set(ex.evidence_titles)
        examples.append((ex.claim, gold_ids, gold_titles, "fever"))
    rng.shuffle(examples)

    bm25 = BM25Retriever(cfg.corpus, top_k=BM25_TOP_K)
    df = pd.read_parquet(cfg.corpus.parquet_path)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_emitted = 0
    n_skipped_no_pos = 0
    n_skipped_pool = 0
    t0 = time.time()
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for i, (claim, gold_ids, gold_titles, source) in enumerate(examples):
            if i % 500 == 0 and i > 0:
                rate = i / max(time.time() - t0, 1e-6)
                eta_s = (len(examples) - i) / max(rate, 1e-6)
                logger.info(
                    "[{}/{}] emitted={} skip_pool={} skip_no_pos={} ({:.1f} ex/s, ETA {:.0f}s)",
                    i,
                    len(examples),
                    n_emitted,
                    n_skipped_pool,
                    n_skipped_no_pos,
                    rate,
                    eta_s,
                )

            positive = _gold_passage_text(df, gold_ids, rng)
            if positive is None:
                n_skipped_no_pos += 1
                continue

            bm25_hits = bm25.retrieve(claim, top_k=BM25_TOP_K)
            negatives = _mine_for_claim(
                claim,
                gold_ids,
                gold_titles,
                bm25_hits,
                rng,
                NEGATIVES_PER_CLAIM,
                SKIP_TOP_K,
            )
            if negatives is None:
                n_skipped_pool += 1
                continue

            for neg in negatives:
                fh.write(
                    json.dumps(
                        {
                            "claim": claim,
                            "positive": positive,
                            "negative": neg,
                            "source": source,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_emitted += 1

    elapsed = time.time() - t0
    logger.info(
        "=== done in {:.0f}s ===  emitted={}  skip_pool={}  skip_no_pos={}",
        elapsed,
        n_emitted,
        n_skipped_pool,
        n_skipped_no_pos,
    )
    logger.info("wrote {}", OUT_PATH)


if __name__ == "__main__":
    app()
