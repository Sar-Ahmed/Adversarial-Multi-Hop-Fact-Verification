"""Evaluate the decomposer on the 30-claim hand-built eval set.

Reads:  artifacts/decomposer_eval_gold.json (claims + category tags)
Writes: artifacts/decomposer_eval.json     (per-claim decompositions + summary)

Run: python -m src.decomposer.eval_decomposer
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "artifacts" / "decomposer_eval_gold.json"
RESULT_PATH = ROOT / "artifacts" / "decomposer_eval.json"


def run() -> None:
    """Run the decomposer over the gold claim set and write the eval JSON."""
    from src.config import PipelineConfig
    from src.decomposer.decomposer import Decomposer
    from src.decomposer.prompts import prompt_hash

    cfg = PipelineConfig.load(ROOT / "configs" / "default.yaml")
    if not cfg.decomposer.llm_path:
        raise RuntimeError(
            "configs/default.yaml has decomposer.llm_path: null — set it to the GGUF path first"
        )

    decomposer = Decomposer(
        llm_path=cfg.decomposer.llm_path,
        n_ctx=cfg.decomposer.n_ctx,
        max_tokens=cfg.decomposer.max_tokens,
        temperature=cfg.decomposer.temperature,
        seed=cfg.eval.seed,
    )

    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    results: list[dict] = []
    fallback_count = 0
    sub_counts: list[int] = []
    elapsed_per: list[float] = []

    for i, ex in enumerate(gold):
        t0 = time.time()
        sub_claims = decomposer.decompose(ex["claim"])
        elapsed = time.time() - t0
        elapsed_per.append(elapsed)

        decomp_payload = [
            {
                "id": s.id,
                "text": s.text,
                "depends_on": list(s.depends_on),
                "reasoning_type": s.reasoning_type.value,
            }
            for s in sub_claims
        ]
        # Read the explicit signal from the decomposer, not a text-equality heuristic
        # (which can't tell apart "parser failed and we wrapped the input" from
        # "model correctly emitted one sub-claim because the input was atomic").
        is_fb = decomposer.last_call_used_fallback
        if is_fb:
            fallback_count += 1
        sub_counts.append(len(sub_claims))

        results.append(
            {
                "id": ex["id"],
                "claim": ex["claim"],
                "category": ex.get("category", "unknown"),
                "n_sub_claims": len(sub_claims),
                "is_fallback": is_fb,
                "elapsed_s": round(elapsed, 2),
                "decomposition": decomp_payload,
            }
        )
        logger.info(
            "[{}/{}] {!r} → {} sub-claims ({:.1f}s){}",
            i + 1,
            len(gold),
            ex["claim"][:60],
            len(sub_claims),
            elapsed,
            " [FALLBACK]" if is_fb else "",
        )

    summary = {
        "model": "Qwen/Qwen2.5-3B-Instruct-GGUF Q4_K_M",
        "model_path": cfg.decomposer.llm_path,
        "prompt_hash": prompt_hash(),
        "seed": cfg.eval.seed,
        "max_tokens": cfg.decomposer.max_tokens,
        "temperature": cfg.decomposer.temperature,
        "n_claims": len(gold),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(gold), 4),
        "avg_sub_claims": round(sum(sub_counts) / len(sub_counts), 2),
        "min_sub_claims": min(sub_counts),
        "max_sub_claims": max(sub_counts),
        "avg_elapsed_s": round(sum(elapsed_per) / len(elapsed_per), 2),
        "total_elapsed_s": round(sum(elapsed_per), 2),
    }

    out = {"summary": summary, "results": results}
    RESULT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("=== summary ===")
    for k, v in summary.items():
        logger.info("  {}: {}", k, v)
    logger.info("wrote {}", RESULT_PATH)


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(42)
    setup_logging()
    run()
