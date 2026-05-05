"""Build the focused Wikipedia corpus for claim verification.

Strategy (Phase 01):
  1. Collect FEVER-encoded gold titles from HoVer train+dev and FEVER train+dev.
  2. Stream the FEVER wiki-pages dump and emit one parquet row per
     (title, sent_idx) for every article whose title is in the gold set.
  3. Save artifacts/corpus.parquet (zstd) + artifacts/corpus_stats.json.

We use the FEVER wiki-pages zip directly (~10 GB) rather than the HF
`fever/wiki_pages` config because the latter is unreliable across HF
`datasets` versions. The zip is downloaded once and cached under data/raw/.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.data.load import load_fever, load_hover

WIKI_URL = "https://fever.ai/download/fever/wiki-pages.zip"

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
WIKI_ZIP = RAW_DIR / "wiki-pages.zip"
GOLD_TITLES = ARTIFACTS / "gold_titles.txt"
CORPUS_OUT = ARTIFACTS / "corpus.parquet"
STATS_OUT = ARTIFACTS / "corpus_stats.json"

# Skip very short sentences — they're typically section headers or noise.
MIN_TEXT_CHARS = 10


def collect_gold_titles() -> set[str]:
    """Union of gold titles from HoVer + FEVER train+dev (FEVER-encoded)."""
    titles: set[str] = set()

    logger.info("loading HoVer to collect gold titles...")
    hover = load_hover()
    for split, examples in hover.items():
        for ex in examples:
            titles.update(ex.supporting_titles)
        logger.info(
            "  hover/{}: {} examples, running gold-title count = {}",
            split,
            len(examples),
            len(titles),
        )

    logger.info("loading FEVER to collect gold titles...")
    fever = load_fever()
    for split, examples in fever.items():
        for ex in examples:
            titles.update(ex.evidence_titles)
        logger.info(
            "  fever/{}: {} examples, running gold-title count = {}",
            split,
            len(examples),
            len(titles),
        )

    logger.info("total unique gold titles: {}", len(titles))
    return titles


def download_wiki_zip(dest: Path = WIKI_ZIP) -> Path:
    """Idempotent download of the FEVER wiki-pages zip (~10 GB)."""
    if dest.exists() and dest.stat().st_size > 1_000_000_000:
        logger.info("wiki zip already present: {} ({:.2f} GB)", dest, dest.stat().st_size / 1e9)
        return dest

    logger.warning("downloading FEVER wiki-pages zip (~10 GB) — this will take a while")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:  # noqa: S310
        total = int(r.headers.get("Content-Length", 0))
        chunk_size = 1 << 20  # 1 MB
        with tqdm(total=total, unit="B", unit_scale=True, desc="wiki-pages.zip") as pbar:
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


def parse_lines_field(lines_field: str) -> list[tuple[int, str]]:
    """FEVER stores article text as `\\n`-separated lines, each
    `<sent_idx>\\t<sentence>\\t<entity>\\t<link>...`. Return [(sent_idx, text)]."""
    out: list[tuple[int, str]] = []
    if not lines_field:
        return out
    for raw in lines_field.split("\n"):
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) < 2:
            continue
        try:
            sid = int(parts[0])
        except ValueError:
            continue
        text = parts[1].strip()
        if len(text) < MIN_TEXT_CHARS:
            continue
        out.append((sid, text))
    return out


def stream_wiki_to_corpus(zip_path: Path, gold_titles: set[str]) -> list[dict]:
    """Stream the wiki zip, yielding rows for articles whose title is in `gold_titles`."""
    rows: list[dict] = []
    matched_titles: set[str] = set()

    with zipfile.ZipFile(zip_path, "r") as zf:
        # The zip contains many JSONL files (wiki-001.jsonl, wiki-002.jsonl, ...).
        jsonl_names = sorted(n for n in zf.namelist() if n.endswith(".jsonl"))
        logger.info("wiki zip has {} jsonl shards", len(jsonl_names))

        for shard_name in tqdm(jsonl_names, desc="wiki shards"):
            with zf.open(shard_name) as fh:
                # Some FEVER shards contain stray non-UTF8 bytes (e.g., 0xb0 degree
                # signs from a Latin-1 source). errors='replace' substitutes U+FFFD
                # for the offending byte rather than crashing the whole pipeline.
                buf = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
                for line in buf:
                    if not line.strip():
                        continue
                    try:
                        article = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    title = article.get("id", "")
                    if title not in gold_titles:
                        continue
                    matched_titles.add(title)
                    for sid, text in parse_lines_field(article.get("lines", "")):
                        rows.append(
                            {
                                "doc_id": f"{title}::{sid}",
                                "title": title,
                                "sent_idx": sid,
                                "text": text,
                            }
                        )
    coverage = len(matched_titles) / max(len(gold_titles), 1)
    logger.info(
        "matched {}/{} gold titles ({:.1%}) → {} passages",
        len(matched_titles),
        len(gold_titles),
        coverage,
        len(rows),
    )
    return rows


def build() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    gold = collect_gold_titles()
    GOLD_TITLES.write_text("\n".join(sorted(gold)), encoding="utf-8")
    logger.info("wrote {} gold titles to {}", len(gold), GOLD_TITLES)

    zip_path = download_wiki_zip()
    rows = stream_wiki_to_corpus(zip_path, gold)

    if not rows:
        raise RuntimeError("Corpus is empty — gold-title matching failed.")

    df = pd.DataFrame(rows)
    df.to_parquet(CORPUS_OUT, compression="zstd", index=False)
    logger.info("wrote {} ({:.2f} MB)", CORPUS_OUT, CORPUS_OUT.stat().st_size / 1e6)

    stats = {
        "n_passages": int(len(df)),
        "n_unique_titles": int(df["title"].nunique()),
        "avg_text_chars": float(df["text"].str.len().mean()),
        "p50_text_chars": float(df["text"].str.len().median()),
        "n_gold_titles_requested": len(gold),
        "title_coverage": float(df["title"].nunique() / max(len(gold), 1)),
    }
    STATS_OUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("stats: {}", stats)


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(42)
    setup_logging()
    build()
