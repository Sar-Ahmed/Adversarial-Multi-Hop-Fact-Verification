# Phase 05 — Hard-Negative Mining and Retriever Fine-tune

**Goal.** Mine hard-negative triplets from FEVER + HoVer training data, fine-tune `bge-small-en-v1.5` with `MultipleNegativesRankingLoss`, and ship whichever of {base, fine-tune} has higher HoVer dev R@10. Spec requirement #2 (second half) plus the explicit "fine-tuned retriever checkpoint with hard negative training logs" deliverable.

**Effort.** 2–3 days. (~30–45 min training time on Colab T4.)
**Compute.** **Colab T4 GPU strongly recommended** for the training step. Hard-negative mining and evaluation are CPU.
**Depends on.** Phase 04 (we need the baseline R@10 to compare against).

## Why this exists

V1's hard-negative fine-tune **dropped** HoVer R@10 by ~2.5 points (0.895 → ~0.870). They reverted to the base model and documented honestly. V3's hypothesis is that V1's miner sourced hard negatives only from FEVER, so the fine-tune optimized for FEVER-style retrieval at HoVer's expense. We mine from a *mix* of FEVER and HoVer training data and validate against HoVer dev.

This phase has a binding decision rule at the end: **ship the model that wins on HoVer dev R@10. If base wins, ship base and write the postmortem.**

## Inputs

- HoVer + FEVER training splits (loaded via Phase 01 loaders).
- BM25 from Phase 04 (used as the hard-negative source).
- `BAAI/bge-small-en-v1.5` base.
- HoVer dev for validation (held out — never touched during training).

## Deliverables

- `src/retrieval/finetune/mine_hard_negatives.py` — produces `artifacts/hard_negatives_v3.jsonl` (one record per triplet: `claim`, `positive_text`, `negative_text`).
- `src/retrieval/finetune/train_bge.py` — sentence-transformers fine-tune script.
- `notebooks/phase05_finetune_retriever.ipynb` — Colab runner mirroring `train_bge.py` for GPU.
- `checkpoints/bge-small-v3-hn/` — final checkpoint (model.safetensors, config, tokenizer, README with hyperparameters).
- `artifacts/retriever_finetune_log.jsonl` — per-step training loss, learning rate, eval metrics.
- `artifacts/retriever_eval_finetune.json` — finetune R@5/10/20 with CIs vs the Phase 04 baseline.
- `docs/PHASE_05_DECISION.md` — short writeup: which checkpoint won, by how much, and the production config update.

## Technical approach

- **Hard-negative mining.**
  - For each (claim, gold_doc_id) in HoVer + FEVER training:
    1. Run BM25 over the corpus, top-50 results.
    2. Drop any result whose `doc_id == gold_doc_id` or whose `title == gold_title`.
    3. **Skip the top-10** results (BGE / V1 trick — top-10 are often near-paraphrases that confuse training).
    4. Sample 4 from the remaining (positions 11–50), with a fixed random seed.
  - Output 4 triplets per claim → ~30–40k triplets total (with the FEVER+HoVer mix).
- **Training.**
  - sentence-transformers `MultipleNegativesRankingLoss` (InfoNCE with in-batch negatives + the explicit hard negative).
  - Hyperparameters: 1 epoch, batch_size=32, lr=2e-5, warmup_ratio=0.1, seed=42, fp16 on GPU.
  - Eval callback: every 500 steps, run R@10 on a 500-example HoVer dev subset to get a training-time signal. Save the best-by-R@10 checkpoint, not the last.
  - Save final checkpoint + training log JSON.
- **Validation.**
  - Re-encode the corpus with the fine-tune (~30 min on Colab T4 or ~45 min CPU).
  - Save fine-tune corpus embeddings to `artifacts/corpus_embeddings_ft.npy` and FAISS index to `artifacts/corpus_ft.faiss`.
  - Run `retrieval_eval.py` (Phase 04) with `retriever.kind: dense_finetuned` config.
  - Compare to baseline.

## Implementation steps

1. Implement `mine_hard_negatives.py`. Verify output: spot-check 5 triplets to ensure negatives are not paraphrases.
2. Implement `train_bge.py`. Test on 500 triplets locally to sanity-check the loop runs.
3. Translate to `notebooks/phase05_finetune_retriever.ipynb`. Mount Drive, upload triplets, run training.
4. Download checkpoint to `checkpoints/bge-small-v3-hn/`.
5. Re-encode corpus with the fine-tune. Build new FAISS index.
6. Run `retrieval_eval.py` against fine-tune.
7. **Decision step.** Compare R@10 base vs fine-tune.
   - If fine-tune R@10 > base R@10 (with CIs not overlapping): set `PipelineConfig.retriever.finetune_path` to the new checkpoint. Smoke test.
   - If fine-tune R@10 ≤ base R@10: keep `finetune_path: null` (use base in production), document the negative result, and note in `docs/PHASE_05_DECISION.md` that the checkpoint is preserved for ablation only.
8. Either way, commit the checkpoint to local storage (not git LFS) and reference it from the eval JSON.

## Exit criteria

- [ ] `artifacts/hard_negatives_v3.jsonl` exists with at least 25k triplets sourced from a mix of HoVer + FEVER.
- [ ] `checkpoints/bge-small-v3-hn/` contains a loadable sentence-transformers model.
- [ ] `artifacts/retriever_finetune_log.jsonl` shows decreasing loss and a saved best-by-eval-R@10 checkpoint.
- [ ] `artifacts/retriever_eval_finetune.json` reports fine-tune R@5/10/20 with CIs against base.
- [ ] `docs/PHASE_05_DECISION.md` exists and states which model won and what production config now references.
- [ ] `make smoke` passes against whichever model is now in production.

## Risks and gotchas

- The biggest risk is that V3 also fails to beat base — the dataset distribution mismatch is real. Mitigations:
  - Mix HoVer training data into the negative pool (not just FEVER).
  - Eval on HoVer dev *during* training and save best-by-eval, not last.
  - If still fails: do a 2-epoch run with lower LR (1e-5) before declaring defeat.
- sentence-transformers v3.x changed the trainer API; pin v3.4.1 to match what the script expects.
- Colab disk is ephemeral — save the checkpoint to Drive and `gdown` it back. Don't trust `/content/`.
- Re-encoding 200k passages takes time; budget for it.

## What NOT to do

- Do not ship the fine-tune just because the loss went down. Loss going down is necessary, not sufficient. The validation gate is HoVer dev R@10 vs the base.
- Do not fine-tune for more than 1 epoch initially. Overfitting on the hard negatives is a real failure mode, especially with small models.
- Do not skip the postmortem if the fine-tune loses. That document is a deliverable.

## Outcome (filled at end of phase)

> Append: triplet count, training wall time on Colab, base vs fine-tune R@10 with CIs, decision (ship which), prod config diff.
