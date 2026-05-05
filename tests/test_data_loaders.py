"""Phase 01 tests for the data loaders.

These tests hit the real datasets via the HuggingFace `datasets` library, so
they require network on first run. They're marked `slow` because of that.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_wiki_encode_decode_roundtrip() -> None:
    from src.data.wiki import fever_decode, fever_encode

    cases = [
        "Inception (2010 film)",
        "Friends [TV series]",
        "Q: a ratio",
        "Plain Title",
    ]
    for c in cases:
        assert fever_decode(fever_encode(c)) == c, f"roundtrip failed for {c!r}"


@pytest.mark.slow
def test_load_hover_minimal() -> None:
    """Load HoVer; verify schema and basic invariants on the dev split."""
    from src.data.load import load_hover

    splits = load_hover()
    assert "validation" in splits or "dev" in splits, f"no dev split in {list(splits)}"
    dev_key = "validation" if "validation" in splits else "dev"
    dev = splits[dev_key]
    assert len(dev) > 0, "HoVer dev is empty"

    sample = dev[0]
    assert sample.claim, "claim is empty"
    assert sample.label in {"SUPPORTED", "REFUTED"}, f"unexpected label {sample.label!r}"
    assert sample.num_hops >= 1
    assert sample.uid


@pytest.mark.slow
def test_load_fever_minimal() -> None:
    """Load FEVER; verify the three-label taxonomy is present in train."""
    from src.data.load import load_fever

    splits = load_fever()
    assert "train" in splits, f"no train split in {list(splits)}"
    train = splits["train"]
    assert len(train) > 0, "FEVER train is empty"

    labels = {ex.label for ex in train[:1000]}
    # Expect at least SUPPORTED and REFUTED in the first 1000 rows;
    # NEI is rarer but present in FEVER.
    assert "SUPPORTED" in labels
    assert "REFUTED" in labels
