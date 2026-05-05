"""CLI entry point: `python -m src.cli verify "<claim>"`.

Phase 02 commands:
    verify   — run the pipeline on a single claim, print the rendered chain
    smoke    — run the 5-example smoke test (`pytest -m smoke`)
"""

from __future__ import annotations

import json as _json
from pathlib import Path

import typer

from src.utils.logging import setup_logging
from src.utils.seed import set_global_seed

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def verify(
    claim: str = typer.Argument(..., help="The claim to verify."),  # noqa: B008
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to PipelineConfig YAML.",
    ),
    json_output: bool = typer.Option(  # noqa: B008
        False, "--json", help="Emit JSON instead of human-readable rendering."
    ),
    seed: int = typer.Option(42, help="Global RNG seed."),  # noqa: B008
) -> None:
    """Run the full pipeline on a single claim."""
    set_global_seed(seed)
    setup_logging()

    from src.config import PipelineConfig
    from src.pipeline import build_pipeline

    cfg = PipelineConfig.load(config_path)
    pipeline = build_pipeline(cfg)
    chain = pipeline.verify(claim)

    if json_output:
        out = {
            "claim": chain.claim,
            "final_verdict": chain.final_verdict.value,
            "final_confidence": chain.final_confidence,
            "sub_claims": [
                {
                    "id": sc.id,
                    "text": sc.text,
                    "depends_on": list(sc.depends_on),
                    "reasoning_type": sc.reasoning_type.value,
                }
                for sc in chain.sub_claims
            ],
            "verifications": [
                {
                    "sub_claim_id": v.sub_claim_id,
                    "verdict": v.verdict.value,
                    "confidence": v.confidence,
                    "cited_passage_ids": list(v.cited_passage_ids),
                    "reasoning": v.reasoning,
                }
                for v in chain.verifications
            ],
        }
        typer.echo(_json.dumps(out, indent=2))
    else:
        typer.echo(chain.render_text())


@app.command()
def smoke() -> None:
    """Run the integration smoke test (delegates to pytest)."""
    import subprocess
    import sys

    rc = subprocess.call([sys.executable, "-m", "pytest", "tests/", "-m", "smoke", "-v"])
    raise typer.Exit(code=rc)


if __name__ == "__main__":
    app()
