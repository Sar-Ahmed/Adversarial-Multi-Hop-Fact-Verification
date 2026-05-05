"""Inspect HoVer raw row schema to make sure our loader gets supporting_titles."""

from __future__ import annotations


def main() -> None:
    """Inspect raw HoVer row schema."""
    from datasets import load_dataset

    ds = load_dataset("hover", trust_remote_code=True)
    print("splits:", list(ds.keys()), flush=True)
    for split_name in ds:
        split = ds[split_name]
        print(f"\n=== {split_name}: n={len(split)} ===", flush=True)
        row = split[0]
        print(f"  keys: {list(row.keys())}", flush=True)
        for k, v in row.items():
            tv = type(v).__name__
            print(f"  {k}: type={tv}, repr={repr(v)[:300]}", flush=True)
        break


if __name__ == "__main__":
    main()
