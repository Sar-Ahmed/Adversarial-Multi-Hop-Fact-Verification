"""HoVer + FEVER loaders backed by `datasets`.

Each loader returns `{split_name: list[Example]}` with labels normalized to the
project taxonomy (SUPPORTED / REFUTED / NEI).

Schemas verified against `datasets==2.20.0`:

HoVer (`hover`):
    splits: train, validation, test
    columns: id, uid, claim, supporting_facts, label, num_hops, hpqa_id
    supporting_facts: list[dict{"key": title (human-readable), "value": sent_idx}]
    label: int — 0=SUPPORTED, 1=NOT_SUPPORTED (we map to REFUTED)
    titles need fever_encode() before joining FEVER wiki dump.

FEVER (`fever`, config `v1.0`):
    splits: train, labelled_dev, paper_dev, paper_test, unlabelled_dev, unlabelled_test
    columns: id, label, claim, evidence_annotation_id, evidence_id,
             evidence_wiki_url, evidence_sentence_id
    Schema is **flat** — one row per (claim, evidence-sentence) pair. Multiple
    rows share the same `id` for a single claim with multiple evidence sentences.
    `evidence_wiki_url` is already in FEVER-encoded form; do NOT re-encode.
    NEI rows have empty/None evidence fields.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.data.types import FeverExample, HoverExample
from src.data.wiki import fever_encode

# HoVer label encoding: 0 -> SUPPORTED, 1 -> NOT_SUPPORTED.
_HOVER_LABEL_MAP: dict[Any, str] = {
    0: "SUPPORTED",
    1: "REFUTED",
    "SUPPORTED": "SUPPORTED",
    "NOT_SUPPORTED": "REFUTED",
}

_FEVER_LABEL_MAP: dict[str, str] = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NEI",
}


def _try_load(candidates: list[tuple[str, str | None]]) -> Any:
    """Try a list of (dataset_name, config) candidates and return the first hit."""
    from datasets import load_dataset

    last_err: Exception | None = None
    for name, config in candidates:
        try:
            logger.info("trying load_dataset({!r}, {!r})", name, config)
            return (
                load_dataset(name, config, trust_remote_code=True)
                if config is not None
                else load_dataset(name, trust_remote_code=True)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("  failed: {}", str(e)[:200])
            last_err = e
    raise RuntimeError(
        f"None of the dataset candidates loaded: {candidates}. Last error: {last_err}"
    )


def _to_hover_example(row: dict[str, Any]) -> HoverExample:
    label_raw = row.get("label")
    label = _HOVER_LABEL_MAP.get(label_raw, str(label_raw))

    facts: list[tuple[str, int]] = []
    sf = row.get("supporting_facts") or []
    for entry in sf:
        if not isinstance(entry, dict):
            continue
        title_raw = entry.get("key") or entry.get("title")
        sid_raw = entry.get("value") if "value" in entry else entry.get("sent_id")
        if title_raw is None or sid_raw is None:
            continue
        # HoVer titles are human-readable; FEVER wiki dump uses encoded form.
        facts.append((fever_encode(str(title_raw)), int(sid_raw)))

    facts_dedup = tuple(dict.fromkeys(facts))
    titles_unique = tuple(dict.fromkeys(t for t, _ in facts_dedup))
    return HoverExample(
        uid=str(row.get("uid") or row.get("id") or ""),
        claim=str(row.get("claim", "")),
        label=label,
        num_hops=int(row.get("num_hops", 0) or 0),
        supporting_titles=titles_unique,
        supporting_facts=facts_dedup,
    )


def _aggregate_fever_split(rows: Any) -> list[FeverExample]:
    """Group FEVER rows by claim id, since the HF schema is one-row-per-evidence."""
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        cid = int(row["id"])
        if cid not in by_id:
            by_id[cid] = {
                "id": cid,
                "claim": str(row.get("claim", "")),
                "label": _FEVER_LABEL_MAP.get(str(row.get("label")), str(row.get("label"))),
                "facts": [],
            }
        title = row.get("evidence_wiki_url")
        sid = row.get("evidence_sentence_id")
        # NEI rows or unlabeled rows have empty title/sid; skip them.
        if title in (None, "", -1) or sid in (None, -1):
            continue
        by_id[cid]["facts"].append((str(title), int(sid)))

    examples: list[FeverExample] = []
    for cid, g in by_id.items():
        facts_dedup = tuple(dict.fromkeys(g["facts"]))
        titles_unique = tuple(dict.fromkeys(t for t, _ in facts_dedup))
        examples.append(
            FeverExample(
                id=cid,
                claim=g["claim"],
                label=g["label"],
                evidence_titles=titles_unique,
                evidence_facts=facts_dedup,
            )
        )
    return examples


def load_hover() -> dict[str, list[HoverExample]]:
    """Load HoVer train + validation + test splits."""
    ds = _try_load(
        [
            ("hover", None),
            ("hover-team/hover", None),
        ]
    )
    out: dict[str, list[HoverExample]] = {}
    for split in ds:
        out[split] = [_to_hover_example(row) for row in ds[split]]
        logger.info("hover/{}: {} examples", split, len(out[split]))
    return out


def load_fever() -> dict[str, list[FeverExample]]:
    """Load FEVER labeled splits (train, labelled_dev, paper_dev/test).

    Excludes unlabelled splits and the wiki_pages config (corpus lives in
    build_corpus.py).
    """
    ds = _try_load(
        [
            ("fever", "v1.0"),
            ("fever/fever", "v1.0"),
        ]
    )
    out: dict[str, list[FeverExample]] = {}
    for split in ds:
        if "wiki" in split.lower() or "unlabelled" in split.lower():
            continue
        # FEVER's HF schema is flat: one row per evidence sentence. Aggregate by claim id.
        examples = _aggregate_fever_split(ds[split])
        out[split] = examples
        logger.info(
            "fever/{}: {} unique claims (from {} evidence-rows)",
            split,
            len(examples),
            len(ds[split]),
        )
    return out
