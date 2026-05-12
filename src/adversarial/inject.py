"""Adversarial distractor injection into retrieval results.

Loads pre-mined distractors from `artifacts/distractors_v3.json` and mixes
them into the dense-retrieval top-K before reranking, so the reranker has
to filter them out under realistic conditions. Phase 11 robustness eval
toggles this on/off to measure the accuracy delta.

Three modes:
  - `mix`              — insert distractors at random positions in the candidate list
  - `replace_bottom`   — replace the K lowest-scored candidates with distractors
  - `replace_random`   — replace K random candidates with distractors
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from src.schema import Passage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISTRACTORS_PATH = ROOT / "artifacts" / "distractors_v3.json"

InjectMode = Literal["mix", "replace_bottom", "replace_random"]


class DistractorStore:
    """Lazy loader for the mined distractors JSON.

    Indexed by HoVer claim `uid`. Callers look up per-claim distractors at
    eval time and pass them into `inject_distractors`.
    """

    def __init__(self, path: str | Path = DEFAULT_DISTRACTORS_PATH) -> None:
        self.path = Path(path)
        self._cache: dict[str, list[Passage]] | None = None

    def _load(self) -> dict[str, list[Passage]]:
        if self._cache is not None:
            return self._cache
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found — run `python -m src.adversarial.mine` first"
            )
        data = json.loads(self.path.read_text(encoding="utf-8"))
        cache: dict[str, list[Passage]] = {}
        for uid, rec in data["results"].items():
            cache[uid] = [
                Passage(
                    doc_id=d["doc_id"],
                    title=d["title"],
                    sent_idx=int(d["sent_idx"]),
                    text=d["text"],
                    # Stash the cosine sim in `score` so downstream code can
                    # see it; reranker overwrites this with its own score.
                    score=float(d["cos"]),
                )
                for d in rec["distractors"]
            ]
        self._cache = cache
        return cache

    def for_claim(self, uid: str) -> list[Passage]:
        return self._load().get(uid, [])


def inject_distractors(
    candidates: list[Passage],
    distractors: list[Passage],
    mode: InjectMode = "mix",
    seed: int | None = 42,
) -> list[Passage]:
    """Inject `distractors` into `candidates` per `mode`.

    Returns a new list; does not mutate the input. Deduplicates by doc_id so
    a distractor that happens to overlap a real retrieval hit isn't doubled.
    """
    if not distractors:
        return list(candidates)
    if mode not in ("mix", "replace_bottom", "replace_random"):
        raise ValueError(f"unknown inject mode: {mode!r}")

    rng = random.Random(seed)
    existing_ids = {p.doc_id for p in candidates}
    new_distractors = [d for d in distractors if d.doc_id not in existing_ids]

    if not new_distractors:
        return list(candidates)

    out = list(candidates)
    if mode == "replace_bottom":
        # Drop the K lowest-scored, append distractors at the bottom.
        if not out:
            return list(new_distractors)
        # Stable-sort by score descending; replace the tail.
        out_sorted = sorted(out, key=lambda p: -p.score)
        keep = out_sorted[: max(0, len(out_sorted) - len(new_distractors))]
        return keep + list(new_distractors)

    if mode == "replace_random":
        if not out:
            return list(new_distractors)
        n_replace = min(len(new_distractors), len(out))
        replace_idx = sorted(rng.sample(range(len(out)), n_replace))
        new_list = list(out)
        for j, idx in enumerate(replace_idx):
            new_list[idx] = new_distractors[j]
        # Append leftover distractors if there were more than we could replace.
        if len(new_distractors) > n_replace:
            new_list.extend(new_distractors[n_replace:])
        return new_list

    # mode == "mix"  — insert at random positions, do not remove anything.
    new_list = list(out)
    for d in new_distractors:
        pos = rng.randint(0, len(new_list))
        new_list.insert(pos, d)
    return new_list
