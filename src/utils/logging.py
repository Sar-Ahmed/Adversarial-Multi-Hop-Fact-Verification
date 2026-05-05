"""loguru sink config: stderr human-readable + JSON sink to artifacts/logs/.

Idempotent — safe to call from every entry point. Replaces the print-debug
pattern used in V1 and the bare `logging` module used in V2.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logging(
    log_dir: Path | str = "artifacts/logs",
    json_sink: bool = True,
    level: str = "INFO",
) -> None:
    """Configure loguru with a human-readable stderr sink and optional JSON file sink.

    Args:
        log_dir: directory for the JSON sink (created if missing).
        json_sink: enable the structured JSON file sink.
        level: minimum log level (one of TRACE/DEBUG/INFO/WARNING/ERROR/CRITICAL).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    if json_sink:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path / "run_{time:YYYYMMDD_HHmmss}.jsonl",
            level=level,
            serialize=True,
            rotation="100 MB",
        )

    _CONFIGURED = True
