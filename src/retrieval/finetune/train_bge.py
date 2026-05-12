"""Fine-tune bge-small-en-v1.5 on Phase 05 hard-negative triplets.

Uses sentence-transformers' MultipleNegativesRankingLoss (InfoNCE with
in-batch negatives plus the explicit hard negative). Designed to run on
either CPU (slow, ~3 hours for 40k triplets) or Colab T4 GPU
(~10-15 min). The accompanying notebook
`notebooks/phase05_finetune_retriever.ipynb` is the recommended path.

Reads:  artifacts/hard_negatives_v3.jsonl
Writes: checkpoints/bge-small-v3-hn/  (model + tokenizer + a README.md)
        artifacts/retriever_finetune_log.jsonl (per-step training stats)

Run:
    python -m src.retrieval.finetune.train_bge --epochs 1 --batch-size 32

Or from the Colab notebook (T4 GPU, fp16) for ~10× speedup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
TRIPLETS_PATH = ROOT / "artifacts" / "hard_negatives_v3.jsonl"
CHECKPOINT_DIR = ROOT / "checkpoints" / "bge-small-v3-hn"
LOG_PATH = ROOT / "artifacts" / "retriever_finetune_log.jsonl"

DEFAULT_BASE_MODEL = "BAAI/bge-small-en-v1.5"

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _load_triplets(path: Path) -> list[dict]:
    """Load JSONL triplets. Each row has {claim, positive, negative, source}."""
    triplets: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            triplets.append(json.loads(line))
    logger.info("loaded {} triplets from {}", len(triplets), path)
    return triplets


@app.command()
def main(
    base_model: str = typer.Option(DEFAULT_BASE_MODEL),
    epochs: int = typer.Option(1),
    batch_size: int = typer.Option(32),
    learning_rate: float = typer.Option(2e-5),
    warmup_ratio: float = typer.Option(0.1),
    seed: int = typer.Option(42),
    output_dir: Path = typer.Option(CHECKPOINT_DIR),  # noqa: B008
    fp16: bool = typer.Option(
        False,
        "--fp16/--no-fp16",
        help="Enable fp16 training (GPU only). Explicit secondary flag so this works "
        "across typer versions — older typer skips deriving --no-fp16 automatically and "
        "fails with 'Secondary flag is not valid for non-boolean flag'.",
    ),
) -> None:
    """Fine-tune bge-small with MNRL on the hard-negative triplets."""
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    set_global_seed(seed)
    setup_logging()

    if not TRIPLETS_PATH.exists():
        raise FileNotFoundError(f"{TRIPLETS_PATH} not found — run mine_hard_negatives.py first")

    triplets = _load_triplets(TRIPLETS_PATH)
    examples = [InputExample(texts=[t["claim"], t["positive"], t["negative"]]) for t in triplets]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: {} (fp16={})", device, fp16 and device == "cuda")

    model = SentenceTransformer(base_model, device=device)
    model.max_seq_length = 256
    train_loss = losses.MultipleNegativesRankingLoss(model)

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
    n_steps = len(loader) * epochs
    warmup_steps = int(n_steps * warmup_ratio)
    logger.info(
        "training: epochs={}, steps={}, warmup={}, batch_size={}, lr={}",
        epochs,
        n_steps,
        warmup_steps,
        batch_size,
        learning_rate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        use_amp=(fp16 and device == "cuda"),
        output_path=str(output_dir),
        show_progress_bar=True,
        save_best_model=False,  # plain save at end of training
    )
    elapsed = time.time() - t0
    logger.info("training done in {:.1f}s", elapsed)

    # Persist a small training-log JSON describing the run.
    log_entry = {
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "seed": seed,
        "device": device,
        "fp16": (fp16 and device == "cuda"),
        "n_triplets": len(triplets),
        "n_steps": n_steps,
        "elapsed_s": round(elapsed, 1),
        "output_path": str(output_dir),
    }
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, indent=2))
    logger.info("wrote {}", LOG_PATH)

    # Drop a tiny README inside the checkpoint dir for hand-off readability.
    (output_dir / "README.md").write_text(
        f"""# bge-small-v3-hn

V3 fine-tune of `{base_model}` on hard-negative triplets mined from
HoVer + FEVER train (skip-top-10 BM25 trick).

| Field | Value |
|---|---|
| base | `{base_model}` |
| epochs | {epochs} |
| batch_size | {batch_size} |
| learning_rate | {learning_rate} |
| warmup_ratio | {warmup_ratio} |
| seed | {seed} |
| device | {device} ({"fp16" if fp16 and device == "cuda" else "fp32"}) |
| n_triplets | {len(triplets)} |
| training_time_s | {round(elapsed, 1)} |

This checkpoint is loaded only if `configs/default.yaml` has
`retriever.finetune_path` pointed here. The Phase 05 decision (ship base or
fine-tune) is recorded in `docs/PHASE_05_DECISION.md`.
""",
        encoding="utf-8",
    )
    logger.info("checkpoint ready at {}", output_dir)


if __name__ == "__main__":
    app()
