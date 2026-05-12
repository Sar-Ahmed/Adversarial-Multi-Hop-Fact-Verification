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

## Outcome (Phase 05 closed 2026-05-13)

**Decision: ship `base`. Fine-tune preserved on disk for Phase 12 ablation. Full writeup in [docs/PHASE_05_DECISION.md](PHASE_05_DECISION.md).**

**Mining (local, CPU).**

- 6,000 HoVer + 4,000 FEVER train claims sampled (seed=42)
- 9,968 claims yielded valid triplets (32 skipped for no recoverable gold)
- **39,872 hard-negative triplets** → `artifacts/hard_negatives_v3.jsonl` (18 MB, force-committed)
- 5,995 HoVer + 3,973 FEVER unique claims → HoVer-weighted mix as designed
- Mining wall time: ~25 hours (CPU). Slower than estimated because the gold-positive lookup uses `df[df.doc_id.isin(...)]` which is O(N=177k) per claim. Logged for follow-up; would be ~100× faster with a doc_id→row dict.

**Training (Colab T4, fp16).**

- Model: `BAAI/bge-small-en-v1.5` base, `MultipleNegativesRankingLoss`
- Hyperparams: 1 epoch, batch_size=64, lr=2e-5, warmup_ratio=0.1, seed=42, max_seq_length=256
- 623 steps, **~7.4 min** wall time on T4 (actual training: 221 s; remainder was the wandb prompt before the cell was restarted with `WANDB_DISABLED=true`)
- Re-encoded 177k passages → `corpus_embeddings_ft.npy` + `corpus_ft.faiss` on T4 in ~3 min (`max_seq_length=256`, `batch_size=256`)

**Comparison eval (HoVer dev n=200, seed=42, 95% bootstrap CI).**

| Metric | Base | Fine-tune | Δ | CI overlap |
|---|---|---|---|---|
| R@5 | 0.452 [0.412, 0.493] | 0.463 [0.424, 0.506] | +0.011 | ⚠ yes |
| **R@10** | **0.524** [0.484, 0.565] | **0.533** [0.491, 0.574] | +0.009 | ⚠ yes |
| R@20 | 0.573 [0.534, 0.613] | 0.588 [0.547, 0.632] | +0.015 | ⚠ yes |
| H@5 | 0.890 [0.845, 0.930] | 0.910 [0.870, 0.950] | +0.020 | ⚠ yes |
| **H@10** | **0.925** [0.890, 0.960] | **0.935** [0.900, 0.970] | +0.010 | ⚠ yes |
| H@20 | 0.950 [0.920, 0.980] | 0.960 [0.935, 0.985] | +0.010 | ⚠ yes |

**Decision.** Fine-tune wins on all 6 metrics by 1–2 points (directional consistency is real — 1.6% chance under the null) but every CI overlaps. Under our binding rule (non-overlapping CI required), production stays on **base**. Fine-tune kept for Phase 12 ablation.

**Production config diff.** `configs/default.yaml` unchanged — `retriever.finetune_path: null` stays. `DenseRetriever` continues to load `BAAI/bge-small-en-v1.5` against `corpus.faiss`.

**Hypothesis update.** V1's negative result was "HoVer recall drops when mining only from FEVER". V3's fix (HoVer-weighted mix) prevented the regression but didn't produce a significant lift. The mix-fix is half-confirmed: no regression, but also no headline gain.

**Smoke test.** `pytest -m smoke` was not re-run because nothing in `build_pipeline()` changed — config still points at base. Will rerun once Phase 06 lands. (If the eval had flipped the decision, the smoke would be mandatory.)

**Three Colab-specific bugs found and fixed during this phase.**

1. **Stale clone / missing `%cd`.** Colab `%cd` only persists for the current cell; restart loses it. Added an explicit hard-reset clone block in the notebook.
2. **`typer-slim 0.24` vs `typer 0.12.5` clash.** Colab preinstalls `typer-slim` newer than our pin, breaking the `--fp16/--no-fp16` flag derivation with `TypeError: Secondary flag is not valid for non-boolean flag`. Fixes: rewrote training + re-encoding cells as inline Python (no typer CLI); added explicit `--fp16/--no-fp16` to `train_bge.py` for future CLI users; dropped `typer`/`pydantic` pins from `requirements-colab.txt`.
3. **`sentence_transformers.fit()` prompts for wandb.** Colab preinstalls wandb; the fit method prompts interactively, blocking the cell. Set `WANDB_DISABLED=true` + `WANDB_MODE=disabled` env vars before the import.

**Open follow-ups.**

- Phase 12 ablation: include base vs fine-tune as one of the ablation rows.
- If Phase 13 error analysis flags retrieval as the bottleneck: rerun `retrieval_eval_finetune.py --n 1000` (CIs tighten by ~2.2×) before committing to a fine-tune flip.
- Mining script: replace `df.isin(gold)` with `dict[doc_id, row_idx]` lookup for ~100× speedup.
- `checkpoints/bge-small-v3-hn/model.safetensors` (133 MB) exceeds GitHub's 100 MB single-file cap. We omit it from git; the Colab notebook is the canonical reproduction path. Document this in `PHASE_05_DECISION.md`.
