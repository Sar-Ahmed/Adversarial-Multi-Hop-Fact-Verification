"""Phase 11 — adversarial robustness eval.

Re-runs the whole-claim verifier path against HoVer-dev with the pre-mined
adversarial distractors injected between retriever and reranker (the same
hook Phase 06 introduced). Computes the **paired bootstrap delta**:

  Δ = acc(clean) − acc(adversarial)

paired across the same example UIDs. Spec target: Δ ≤ 0.05.

Whole-claim mode is used because (a) Phase 11's headline picks whole-claim
as the production aggregator (decomposed mode is worse on this 3B verifier),
and (b) whole-claim re-runs are ~10× faster than decomposed re-runs.

The clean-side traces are read from artifacts/per_subclaim_traces.jsonl
(Phase 07). The adversarial-side traces are computed here and saved to
artifacts/adversarial_traces.jsonl with the same resume pattern.

Reads:  artifacts/per_subclaim_traces.jsonl, artifacts/distractors_v3.json
Writes: artifacts/adversarial_traces.jsonl, artifacts/robustness_eval.json

Run: python -m src.eval.robustness --n 50
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
CLEAN_TRACES_PATH = ROOT / "artifacts" / "per_subclaim_traces.jsonl"
ADV_TRACES_PATH = ROOT / "artifacts" / "adversarial_traces.jsonl"
RESULT_PATH = ROOT / "artifacts" / "robustness_eval.json"
DISTRACTORS_PATH = ROOT / "artifacts" / "distractors_v3.json"

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _load_clean_traces() -> list[dict]:
    rows: list[dict] = []
    with open(CLEAN_TRACES_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_distractors() -> dict[str, list[dict]]:
    data = json.loads(DISTRACTORS_PATH.read_text(encoding="utf-8"))
    return {uid: rec["distractors"] for uid, rec in data["results"].items()}


def _load_existing_adv_uids() -> set[str]:
    if not ADV_TRACES_PATH.exists():
        return set()
    uids: set[str] = set()
    with open(ADV_TRACES_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                uids.add(json.loads(line)["uid"])
    return uids


def _paired_bootstrap_delta(
    pairs: list[tuple[int, int]], n_resamples: int = 1000, seed: int = 42
) -> dict:
    """Each pair is (clean_correct, adv_correct) ∈ {0,1} × {0,1}.
    Returns the paired-bootstrap point + 95% CI on delta = clean - adv."""
    import numpy as np

    if not pairs:
        return {"n": 0, "delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    arr = np.asarray(pairs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, len(arr), size=len(arr))
        sample = arr[idx]
        deltas[i] = sample[:, 0].mean() - sample[:, 1].mean()
    point = float(arr[:, 0].mean() - arr[:, 1].mean())
    return {
        "n": int(len(arr)),
        "acc_clean": round(float(arr[:, 0].mean()), 4),
        "acc_adversarial": round(float(arr[:, 1].mean()), 4),
        "delta": round(point, 4),
        "ci_lo": round(float(np.percentile(deltas, 2.5)), 4),
        "ci_hi": round(float(np.percentile(deltas, 97.5)), 4),
    }


@app.command()
def main(
    n: int = typer.Option(50, help="Number of HoVer-dev claims to re-run adversarially."),
    seed: int = typer.Option(42),
) -> None:
    """Run adversarial whole-claim verifier on a subset and compute the delta."""
    from src.adversarial.inject import inject_distractors
    from src.config import PipelineConfig
    from src.reranker.cross_encoder import CrossEncoderReranker
    from src.retrieval.dense import DenseRetriever
    from src.schema import Passage
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed
    from src.verifier.ensemble import EnsembleVerifier

    set_global_seed(seed)
    setup_logging()

    cfg = PipelineConfig.load(ROOT / "configs" / "default.yaml")

    clean = _load_clean_traces()
    distractors_by_uid = _load_distractors()

    # Sample the same UIDs we'll evaluate (no resampling between calls).
    rng = random.Random(seed)
    candidate_uids = [c["uid"] for c in clean if c["uid"] in distractors_by_uid]
    if n and len(candidate_uids) > n:
        candidate_uids = rng.sample(candidate_uids, n)

    already_adv = _load_existing_adv_uids()
    todo_uids = [u for u in candidate_uids if u not in already_adv]
    logger.info(
        "adversarial eval: {} target uids, {} already done, {} to compute",
        len(candidate_uids),
        len(already_adv),
        len(todo_uids),
    )

    clean_by_uid = {c["uid"]: c for c in clean}

    retriever = DenseRetriever(cfg.retriever, cfg.corpus)
    reranker = CrossEncoderReranker(cfg.reranker)
    verifier = EnsembleVerifier(cfg.verifier)  # no calibrator in this whole-claim eval

    ADV_TRACES_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(ADV_TRACES_PATH, "a", encoding="utf-8") as fh:
        for i, uid in enumerate(todo_uids):
            if i % 5 == 0 and i > 0:
                rate = i / max(time.time() - t0, 1e-6)
                eta = (len(todo_uids) - i) / max(rate, 1e-6)
                logger.info(
                    "[{}/{}] ({:.2f} cl/s, ETA {:.0f}s)",
                    i,
                    len(todo_uids),
                    rate,
                    eta,
                )
            c = clean_by_uid[uid]
            candidates = retriever.retrieve(c["claim"], top_k=cfg.retriever.top_k)
            distractors = [
                Passage(
                    doc_id=d["doc_id"],
                    title=d["title"],
                    sent_idx=int(d["sent_idx"]),
                    text=d["text"],
                    score=float(d.get("cos", 0.85)),
                )
                for d in distractors_by_uid[uid]
            ]
            mixed = inject_distractors(
                candidates,
                distractors,
                mode=cfg.adversarial.inject_mode,
                seed=cfg.eval.seed,
            )
            top = reranker.rerank(c["claim"], mixed, top_k=cfg.reranker.top_k)
            verdict, conf, reason = verifier.verify(c["claim"], top)
            fh.write(
                json.dumps(
                    {
                        "uid": uid,
                        "claim": c["claim"],
                        "gold_label": c["gold_label"],
                        "adversarial_verdict": verdict.value,
                        "adversarial_confidence": conf,
                        "adversarial_reasoning": reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()

    # Read back adversarial traces (now complete for `candidate_uids`).
    adv_by_uid: dict[str, dict] = {}
    with open(ADV_TRACES_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                adv_by_uid[row["uid"]] = row

    # Build paired (clean_correct, adv_correct) for every uid in our sample.
    # Phase 07 traces cached `llm_only` and `llm_plus_nli_veto`; re-aggregate
    # through bidir mode inline so we measure the production rule.
    from src.schema import Label
    from src.verifier.aggregate import aggregate as _agg

    pairs: list[tuple[int, int]] = []
    for uid in candidate_uids:
        c = clean_by_uid[uid]
        a = adv_by_uid.get(uid)
        if a is None:
            continue
        gold = c["gold_label"]
        clean_verdict, _, _ = _agg(
            mode="llm_plus_nli_bidir",
            llm_verdict=Label(c["llm"]["verdict"]),
            llm_confidence=c["llm"]["confidence"],
            llm_reasoning=c["llm"]["reasoning"],
            nli_scores=c["nli"],
            contra_veto_threshold=cfg.verifier.contra_veto_threshold,
            entail_threshold=cfg.verifier.entail_threshold,
        )
        clean_pred = clean_verdict.value
        adv_pred = a["adversarial_verdict"]
        pairs.append((1 if clean_pred == gold else 0, 1 if adv_pred == gold else 0))

    result = _paired_bootstrap_delta(pairs, n_resamples=1000, seed=seed)
    payload = {
        "summary": {
            "n_target": len(candidate_uids),
            "n_evaluated": len(pairs),
            "seed": seed,
            "mode": "whole_claim_llm_plus_nli_bidir",
            "inject_mode": cfg.adversarial.inject_mode,
            "spec_target": "delta <= 0.05",
            "passes_spec": result["delta"] <= 0.05,
        },
        "paired_bootstrap": result,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("=== robustness eval ===")
    logger.info(
        "  n={}, acc_clean={:.3f}, acc_adversarial={:.3f}",
        result["n"],
        result["acc_clean"],
        result["acc_adversarial"],
    )
    logger.info(
        "  Δ = {:.3f} [{:.3f}, {:.3f}]  spec target: ≤ 0.05  passes={}",
        result["delta"],
        result["ci_lo"],
        result["ci_hi"],
        payload["summary"]["passes_spec"],
    )
    logger.info("wrote {}", RESULT_PATH)


if __name__ == "__main__":
    app()
