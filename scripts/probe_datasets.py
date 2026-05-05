"""Phase 01 probe: load HoVer + FEVER, print schema and sample row.

Run with: python -m scripts.probe_datasets

Sanity-checks the loaders before kicking off the 10 GB wiki download.
"""

from __future__ import annotations

import json
import sys

from src.utils.logging import setup_logging
from src.utils.seed import set_global_seed


def main() -> int:
    """Load HoVer + FEVER and print sample rows; returns 0 on success, 1 on failure."""
    set_global_seed(42)
    setup_logging()

    from src.data.load import load_fever, load_hover

    print("=== HoVer ===", flush=True)
    try:
        hover = load_hover()
        for split, examples in hover.items():
            print(f"  {split}: {len(examples)} examples", flush=True)
        first = next(iter(hover.values()))[0]
        print("  sample:", flush=True)
        print(
            json.dumps(
                {
                    "uid": first.uid,
                    "claim": first.claim[:120],
                    "label": first.label,
                    "num_hops": first.num_hops,
                    "supporting_titles": list(first.supporting_titles)[:5],
                    "supporting_facts": [list(f) for f in first.supporting_facts[:5]],
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  HoVer load FAILED: {e}", flush=True)
        return 1

    print("\n=== FEVER ===", flush=True)
    try:
        fever = load_fever()
        for split, examples in fever.items():
            print(f"  {split}: {len(examples)} examples", flush=True)
        first = next(iter(fever.values()))[0]
        print("  sample:", flush=True)
        print(
            json.dumps(
                {
                    "id": first.id,
                    "claim": first.claim[:120],
                    "label": first.label,
                    "evidence_titles": list(first.evidence_titles)[:5],
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  FEVER load FAILED: {e}", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
