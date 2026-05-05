"""Debug why retrieval recall is much lower than V1 reported.

Hypothesis: HoVer gold doc_ids don't match the corpus doc_ids — likely a title
encoding mismatch between HoVer's `supporting_facts` (passed through
`fever_encode`) and the FEVER wiki dump's `id` field.

Checks:
 1. Does ANY of the HoVer dev gold doc_ids exist in the corpus?
 2. For one example, what does retrieve() return vs what does gold expect?
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    """Run the diagnostic checks."""
    import pandas as pd

    from src.config import PipelineConfig
    from src.data.load import load_hover

    cfg = PipelineConfig.load(Path("configs/default.yaml"))
    df = pd.read_parquet(cfg.corpus.parquet_path)
    corpus_doc_ids = set(df["doc_id"].tolist())
    print(f"corpus has {len(corpus_doc_ids)} doc_ids", flush=True)
    print(f"  sample: {next(iter(corpus_doc_ids))!r}", flush=True)

    splits = load_hover()
    dev = [ex for ex in splits["validation"] if ex.supporting_facts]
    print(f"hover/validation has {len(dev)} examples with gold facts", flush=True)

    # Sample a few examples
    for i in range(5):
        ex = dev[i]
        print(f"\n--- example {i} ---", flush=True)
        print(f"  claim: {ex.claim[:90]}", flush=True)
        print(f"  num_hops: {ex.num_hops}", flush=True)
        gold_doc_ids = {f"{title}::{sid}" for title, sid in ex.supporting_facts}
        print(f"  gold doc_ids ({len(gold_doc_ids)}):", flush=True)
        for g in list(gold_doc_ids)[:5]:
            in_corpus = g in corpus_doc_ids
            print(f"    {g!r}  in_corpus={in_corpus}", flush=True)
            if not in_corpus:
                # Look for partial matches (same title, any sent_idx)
                title = g.split("::")[0]
                candidates = [d for d in corpus_doc_ids if d.startswith(title + "::")][:3]
                if candidates:
                    print(f"      title found at: {candidates}", flush=True)
                else:
                    # Try without title encoding
                    print(f"      title NOT in corpus; checking partial title match", flush=True)
                    title_words = title.split("_")
                    if title_words:
                        first = title_words[0]
                        partial = [
                            d
                            for d in list(corpus_doc_ids)[:10000]
                            if d.startswith(first + "_") or d == first + "::0"
                        ][:3]
                        print(f"      first-word matches: {partial}", flush=True)

    # Aggregate stats
    total_gold = 0
    matched_gold = 0
    for ex in dev[:500]:
        for title, sid in ex.supporting_facts:
            total_gold += 1
            if f"{title}::{sid}" in corpus_doc_ids:
                matched_gold += 1
    print(
        f"\n=== aggregate ({500} examples) ===\ngold_in_corpus = {matched_gold} / {total_gold}"
        f" ({matched_gold/total_gold:.1%})",
        flush=True,
    )


if __name__ == "__main__":
    main()
