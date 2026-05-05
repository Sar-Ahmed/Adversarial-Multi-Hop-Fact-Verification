"""Debug the FEVER dataset structure — what splits are available, what does
a row look like, and which config has train labels with evidence?

Run: python -m scripts.debug_fever_schema
"""

from __future__ import annotations

import json
from typing import Any


def try_load(name: str, config: str | None) -> dict[str, Any] | None:
    """Attempt to load a dataset; return splits + handle on success, None on failure."""
    from datasets import load_dataset

    try:
        ds = (
            load_dataset(name, config, trust_remote_code=True)
            if config
            else load_dataset(name, trust_remote_code=True)
        )
        return {"splits": list(ds.keys()), "ds": ds}
    except Exception as e:  # noqa: BLE001
        print(f"  load_dataset({name!r}, {config!r}) FAILED: {str(e)[:300]}")
        return None


def main() -> None:
    """Try every plausible FEVER dataset name+config and print first-row schema."""
    candidates = [
        ("fever", "v1.0"),
        ("fever", "v2.0"),
        ("fever/fever", "v1.0"),
        ("fever-team/fever", None),
        ("copenlu/fever_gold_evidence", None),
        ("mwong/fever-evidence-related", None),
    ]

    for name, config in candidates:
        print(f"\n=== {name} ({config}) ===", flush=True)
        result = try_load(name, config)
        if result is None:
            continue
        for split in result["splits"]:
            ds = result["ds"][split]
            n = len(ds)
            print(f"  split {split!r}: n={n}", flush=True)
            if n > 0:
                row = ds[0]
                # Print a compact view of the first row's structure
                view: dict[str, Any] = {}
                for k, v in row.items():
                    if isinstance(v, list):
                        view[k] = f"<list len={len(v)} sample={v[:1]}>" if v else "<list len=0>"
                    elif isinstance(v, dict):
                        view[k] = f"<dict keys={list(v.keys())}>"
                    elif isinstance(v, str):
                        view[k] = v[:80] + ("..." if len(v) > 80 else "")
                    else:
                        view[k] = v
                print("    first row:", flush=True)
                print(f"    {json.dumps(view, default=str, indent=2)[:1500]}", flush=True)


if __name__ == "__main__":
    main()
