"""Print the raw model output for known fallback claims to see what's wrong."""

from __future__ import annotations

from src.config import PipelineConfig
from src.decomposer.llm import LocalLLM
from src.decomposer.prompts import build_messages


def main() -> None:
    """Run a few claims through the LLM and print the raw assistant message."""
    cfg = PipelineConfig.load("configs/default.yaml")
    llm = LocalLLM(model_path=cfg.decomposer.llm_path, n_ctx=cfg.decomposer.n_ctx, seed=42)

    fallback_claims = [
        "The Eiffel Tower is located in Paris.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Tom Cruise has never won an Academy Award for Best Actor.",
        "The 2008 Olympic Games were held in Beijing.",
        "Mount Everest is taller than K2.",
    ]

    for c in fallback_claims:
        print(f"\n=== claim: {c!r} ===", flush=True)
        msgs = build_messages(c, retry=False)
        raw = llm.chat(messages=msgs, max_tokens=512, temperature=0.0, stop=["\n\n\n", "Claim:"])
        print(f"--- raw ({len(raw)} chars) ---", flush=True)
        print(raw, flush=True)
        print("--- end raw ---", flush=True)


if __name__ == "__main__":
    from src.utils.logging import setup_logging

    setup_logging(json_sink=False)
    main()
