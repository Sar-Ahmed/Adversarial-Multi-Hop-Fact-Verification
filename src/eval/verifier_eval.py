"""Phase 07 verifier eval — compare `llm_only` vs `llm_plus_nli_veto`.

We run the full retrieval+rerank+verifier pipeline once per claim
(skipping decomposition — Phase 07 measures the verifier itself, not the
end-to-end multi-hop pipeline; that's Phase 11's job), capture the LLM
verdict + NLI scores in a trace, and then compute metrics for *both*
aggregation modes offline from the same trace. Two birds, one (slow)
pipeline pass.

Reads:  configs/default.yaml + Phase 01 corpus + Phase 04 reranker + Qwen GGUF
Writes:
  artifacts/per_subclaim_traces.jsonl   one trace per HoVer-dev claim
  artifacts/verifier_eval_phase07.json  per-mode metrics with bootstrap CIs

Run:    python -m src.eval.verifier_eval --n 200
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
TRACES_PATH = ROOT / "artifacts" / "per_subclaim_traces.jsonl"
RESULT_PATH = ROOT / "artifacts" / "verifier_eval_phase07.json"


def _paths_for_suffix(suffix: str) -> tuple[Path, Path]:
    """Return (traces, result) paths for an output suffix. Empty = v3.0 paths."""
    if not suffix:
        return TRACES_PATH, RESULT_PATH
    return (
        ROOT / "artifacts" / f"per_subclaim_traces_{suffix}.jsonl",
        ROOT / "artifacts" / f"verifier_eval_{suffix}.json",
    )

app = typer.Typer(add_completion=False, no_args_is_help=False)

_LABEL_ORDER = ["SUPPORTED", "REFUTED", "NEI"]


def _metrics_from_traces(traces: list[dict], mode: str) -> dict:
    """Compute accuracy + per-class F1 with bootstrap CIs from cached traces."""
    from sklearn.metrics import classification_report

    from src.eval.metrics import bootstrap_ci

    y_true: list[str] = []
    y_pred: list[str] = []
    correct: list[float] = []
    for tr in traces:
        gold = tr["gold_label"]
        if gold not in _LABEL_ORDER:
            continue
        pred = tr["mode_results"][mode]["verdict"]
        y_true.append(gold)
        y_pred.append(pred)
        correct.append(1.0 if pred == gold else 0.0)

    if not y_true:
        return {"n": 0}

    acc_point, acc_lo, acc_hi = bootstrap_ci(correct)
    report = classification_report(
        y_true,
        y_pred,
        labels=_LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    per_class: dict[str, dict] = {}
    for cls in _LABEL_ORDER:
        cls_correct = [
            1.0 if (y == cls and p == cls) else 0.0 for y, p in zip(y_true, y_pred, strict=True)
        ]
        # precision / recall / f1 from sklearn (no CI — those are per-class
        # rates and bootstrapping each rate is overkill at n=200)
        per_class[cls] = {
            "precision": round(float(report[cls]["precision"]), 4),
            "recall": round(float(report[cls]["recall"]), 4),
            "f1": round(float(report[cls]["f1-score"]), 4),
            "support": int(report[cls]["support"]),
            "matched": int(sum(cls_correct)),
        }

    return {
        "n": len(y_true),
        "accuracy": {
            "point": round(acc_point, 4),
            "ci_lo": round(acc_lo, 4),
            "ci_hi": round(acc_hi, 4),
        },
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "per_class": per_class,
        "confusion_matrix": _confusion(y_true, y_pred),
    }


def _confusion(y_true: list[str], y_pred: list[str]) -> dict:
    cm: dict[str, dict[str, int]] = {t: {p: 0 for p in _LABEL_ORDER} for t in _LABEL_ORDER}
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t][p] += 1
    return cm


@app.command()
def main(
    n: int = typer.Option(200, help="Number of HoVer-dev claims (0 = all)."),
    seed: int = typer.Option(42),
    prompt_variant: str = typer.Option(
        "v1",
        "--prompt-variant",
        help="Verifier prompt variant: v1 (v3.0 production) or v2 (Phase 16 soft-prompt).",
    ),
    out_suffix: str = typer.Option(
        "",
        "--out-suffix",
        help="Suffix for traces+result paths (e.g. 'softprompt'). Empty = v3.0 paths.",
    ),
) -> None:
    """Run the verifier eval and write traces + metrics."""
    from src.config import PipelineConfig
    from src.data.load import load_hover
    from src.reranker.cross_encoder import CrossEncoderReranker
    from src.retrieval.dense import DenseRetriever
    from src.schema import Label
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed
    from src.verifier.aggregate import aggregate
    from src.verifier.ensemble import EnsembleVerifier
    from src.verifier.prompts import prompt_hash

    set_global_seed(seed)
    setup_logging()

    cfg = PipelineConfig.load(ROOT / "configs" / "default.yaml")
    if not cfg.verifier.llm_path:
        raise RuntimeError(
            "configs/default.yaml has verifier.llm_path: null — Phase 07 needs the GGUF"
        )

    traces_path, result_path = _paths_for_suffix(out_suffix)
    logger.info(
        "prompt_variant={} hash={} → traces={}", prompt_variant, prompt_hash(prompt_variant), traces_path.name
    )

    splits = load_hover()
    dev = [ex for ex in splits["validation"] if ex.supporting_facts]
    rng = random.Random(seed)
    if n and len(dev) > n:
        dev = rng.sample(dev, n)
    logger.info("verifier eval on {} HoVer-dev claims (seed={})", len(dev), seed)

    retriever = DenseRetriever(cfg.retriever, cfg.corpus)
    reranker = CrossEncoderReranker(cfg.reranker)
    # Use llm_plus_nli_veto so verify_with_trace always has NLI scores cached;
    # we recompute llm_only by re-aggregating the trace offline.
    verifier = EnsembleVerifier(cfg.verifier, prompt_variant=prompt_variant)

    # Resume from an existing traces file if present so a system restart mid-eval
    # doesn't waste the work already done. Each line is one (claim) row keyed
    # by `uid`; we skip uids already seen and append new ones.
    traces: list[dict] = []
    already_seen: set[str] = set()
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    if traces_path.exists():
        with open(traces_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                traces.append(row)
                already_seen.add(row["uid"])
        logger.info("resuming: loaded {} existing trace rows from {}", len(traces), traces_path)

    todo = [ex for ex in dev if ex.uid not in already_seen]
    logger.info("{} claims still to verify (already done: {})", len(todo), len(already_seen))

    t0 = time.time()
    # Append-mode so partial work survives an interruption.
    with open(traces_path, "a", encoding="utf-8") as fh:
        for i, ex in enumerate(todo):
            if i % 10 == 0 and i > 0:
                rate = i / max(time.time() - t0, 1e-6)
                eta = (len(todo) - i) / max(rate, 1e-6)
                logger.info(
                    "[{}/{} of remaining; {}/{} overall] ({:.2f} cl/s, ETA {:.0f}s)",
                    i,
                    len(todo),
                    i + len(already_seen),
                    len(dev),
                    rate,
                    eta,
                )

            candidates = retriever.retrieve(ex.claim, top_k=cfg.retriever.top_k)
            top = reranker.rerank(ex.claim, candidates, top_k=cfg.reranker.top_k)
            trace = verifier.verify_with_trace(ex.claim, top)

            # Also re-aggregate as llm_only for comparison (free — no extra LLM call)
            llm_only_verdict, llm_only_conf, llm_only_reason = aggregate(
                mode="llm_only",
                llm_verdict=Label(trace["llm"]["verdict"]),
                llm_confidence=trace["llm"]["confidence"],
                llm_reasoning=trace["llm"]["reasoning"],
                nli_scores={},
            )

            row = {
                "uid": ex.uid,
                "claim": ex.claim,
                "gold_label": ex.label,
                "num_hops": ex.num_hops,
                "n_passages": len(top),
                "passage_doc_ids": [p.doc_id for p in top],
                "llm": trace["llm"],
                "nli": trace["nli"],
                "mode_results": {
                    "llm_only": {
                        "verdict": llm_only_verdict.value,
                        "confidence": llm_only_conf,
                        "reasoning": llm_only_reason,
                    },
                    "llm_plus_nli_veto": trace["final"],
                },
            }
            traces.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

    elapsed = time.time() - t0
    logger.info("=== eval done in {:.1f}s ===", elapsed)
    logger.info("traces: {} rows → {}", len(traces), traces_path)

    # Compute metrics for each mode
    results = {
        "llm_only": _metrics_from_traces(traces, "llm_only"),
        "llm_plus_nli_veto": _metrics_from_traces(traces, "llm_plus_nli_veto"),
    }
    payload = {
        "summary": {
            "n_claims": len(dev),
            "seed": seed,
            "verifier_llm": cfg.verifier.llm_path,
            "nli_model": cfg.verifier.nli_model,
            "prompt_variant": prompt_variant,
            "prompt_hash": prompt_hash(prompt_variant),
            "contra_veto_threshold": cfg.verifier.contra_veto_threshold,
            "entail_threshold": cfg.verifier.entail_threshold,
            "elapsed_s": round(elapsed, 1),
        },
        "results": results,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("=== summary ===")
    for mode, m in results.items():
        if not m:
            continue
        acc = m["accuracy"]
        logger.info(
            "  {:18s}  acc={:.3f} [{:.3f}, {:.3f}]  macro_f1={:.3f}",
            mode,
            acc["point"],
            acc["ci_lo"],
            acc["ci_hi"],
            m["macro_f1"],
        )
    logger.info("wrote {}", result_path)


if __name__ == "__main__":
    app()
