"""Print the raw structure of a FEVER labelled_dev row to fix evidence parsing."""

from __future__ import annotations


def main() -> None:
    """Inspect raw FEVER row schemas across splits."""
    from datasets import load_dataset

    ds = load_dataset("fever", "v1.0", trust_remote_code=True)
    print("splits:", list(ds.keys()), flush=True)

    for split_name in ("train", "labelled_dev"):
        if split_name not in ds:
            continue
        split = ds[split_name]
        print(f"\n=== {split_name}: n={len(split)} ===", flush=True)
        # Find a row that has evidence (i.e. SUPPORTED or REFUTED, not NEI)
        for i in range(min(20, len(split))):
            row = split[i]
            label = row.get("label")
            ev = row.get("evidence")
            print(f"\nrow {i}: label={label}", flush=True)
            print(f"  keys: {list(row.keys())}", flush=True)
            print(f"  evidence type: {type(ev).__name__}", flush=True)
            print(
                f"  evidence repr: {repr(ev)[:600]}",
                flush=True,
            )
            if isinstance(ev, dict):
                print(f"  evidence dict keys: {list(ev.keys())}", flush=True)
                for k, v in ev.items():
                    print(f"    {k}: type={type(v).__name__}, sample={repr(v)[:200]}", flush=True)
            if i >= 4:
                break


if __name__ == "__main__":
    main()
