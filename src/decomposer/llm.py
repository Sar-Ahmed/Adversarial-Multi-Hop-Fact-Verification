"""Thin wrapper around llama_cpp.Llama for chat completions + logprobs.

Lazy-loads the model on first call (cold start ~10 s for Qwen 3B Q4 on CPU).
Used by the decomposer (Phase 03) and the verifier LLM (Phase 07). The
logprobs path is consumed by the calibrator (Phase 08) — exposed here so
both phases share one wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


class LocalLLM:
    """Lazy llama-cpp wrapper. One instance is reused across many calls."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        seed: int = 42,
        n_threads: int | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.seed = seed
        self.n_threads = n_threads
        self._llm: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"GGUF model not found at {self.model_path}. "
                "Run `python -m scripts.download_qwen` to fetch it."
            )
        from llama_cpp import Llama

        logger.info("loading llama_cpp model: {}", self.model_path)
        kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "seed": self.seed,
            "verbose": False,
            "logits_all": False,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads
        self._llm = Llama(**kwargs)
        logger.info("llama_cpp model loaded")

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """Return the assistant's response text from a chat-completion call."""
        self._ensure_loaded()
        out = self._llm.create_chat_completion(  # type: ignore[union-attr]
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        return out["choices"][0]["message"]["content"]

    def chat_with_logprobs(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_logprobs: int = 5,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return the full chat-completion dict with logprobs.

        Phase 06/08 consume `out["choices"][0]["logprobs"]["content"]` to
        derive verdict-token confidence and feed the NEI calibrator.
        """
        self._ensure_loaded()
        return self._llm.create_chat_completion(  # type: ignore[union-attr]
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            stop=stop or [],
        )
