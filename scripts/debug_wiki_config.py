"""Probe `fever/wiki_pages` to see if it's loadable and what it looks like.

If HF caches it cleanly, we can read the corpus without manually downloading
the 10 GB zip. If not, we fall back to direct download in build_corpus.py.
"""

from __future__ import annotations


def main() -> None:
    """Probe HF's fever/wiki_pages config via streaming."""
    from datasets import load_dataset

    print("trying load_dataset('fever', 'wiki_pages', streaming=True)...", flush=True)
    try:
        ds = load_dataset("fever", "wiki_pages", trust_remote_code=True, streaming=True)
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {str(e)[:600]}", flush=True)
        return

    print(f"  splits: {list(ds.keys())}", flush=True)
    for split_name in ds:
        stream = ds[split_name]
        print(f"\n=== {split_name} (streamed) ===", flush=True)
        for i, row in enumerate(stream):
            if i == 0:
                print(f"  keys: {list(row.keys())}", flush=True)
                for k, v in row.items():
                    tv = type(v).__name__
                    rv = repr(v)
                    print(f"  {k}: type={tv}, len={len(v) if hasattr(v, '__len__') else 'NA'}, sample={rv[:200]}", flush=True)
            if i >= 2:
                break


if __name__ == "__main__":
    main()
