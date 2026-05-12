# Phase 05 Decision — Ship `base` retriever, keep fine-tune for ablation

**Date:** 2026-05-13
**Eval:** `artifacts/retriever_eval_finetune.json` (HoVer dev, n=200, seed=42, 95% bootstrap CIs n=1000)

## TL;DR

The HoVer + FEVER hard-negative fine-tune (`checkpoints/bge-small-v3-hn`) is **consistently 1–2 points better** than the `BAAI/bge-small-en-v1.5` base across all six measured metrics, but **every CI overlaps** with the base under bootstrap. Under the binding Phase 05 decision rule, that's not enough to flip production. **Production stays on `bge-small-en-v1.5` base; the fine-tune is preserved for Phase 12 ablation and a possible larger-n re-eval.**

## Headline numbers

| Metric | Base | Fine-tune | Δ | CI overlap |
|---|---|---|---|---|
| R@5 | 0.452 [0.412, 0.493] | 0.463 [0.424, 0.506] | +0.011 | yes |
| **R@10** | **0.524** [0.484, 0.565] | **0.533** [0.491, 0.574] | **+0.009** | yes |
| R@20 | 0.573 [0.534, 0.613] | 0.588 [0.547, 0.632] | +0.015 | yes |
| H@5 | 0.890 [0.845, 0.930] | 0.910 [0.870, 0.950] | +0.020 | yes |
| **H@10** | **0.925** [0.890, 0.960] | **0.935** [0.900, 0.970] | **+0.010** | yes |
| H@20 | 0.950 [0.920, 0.980] | 0.960 [0.935, 0.985] | +0.010 | yes |

Notation: `point [ci_lo, ci_hi]` on 200 HoVer-dev examples. R@K = per-passage recall; H@K = hit-rate.

## Decision rule (as documented in `docs/PHASE_05_retriever_finetune.md`)

> **Ship the model that wins on HoVer dev R@10. If fine-tune R@10 CI is above base R@10 CI: ship fine-tune. Else: ship base and document the negative result.**

Applied:
- Fine-tune R@10 CI: [0.491, 0.574]
- Base R@10 CI: [0.484, 0.565]
- These overlap. The fine-tune does *not* win under the rule.
- **Decision: ship `base`.**

## Why not pick the fine-tune anyway (it's better on every metric)?

Directional consistency across all 6 metrics is suggestive — the probability of all 6 paired comparisons going the same direction under the null is (1/2)^6 ≈ 1.6%. That's a real signal, but:

1. **The rule is the rule.** Phase 05's contract was non-overlapping CIs, and we wrote it that way deliberately to avoid shipping fine-tunes that look better but aren't. Loosening the rule now to favour a result we like is exactly the bias we wanted to defend against.
2. **Cost of a wrong call is asymmetric.** Shipping a fine-tune that turns out to be noise burdens every subsequent phase with extra checkpoint + index management. Shipping `base` when the fine-tune is real costs us up to ~1 point of recall — recoverable later.
3. **The fine-tune is preserved.** `checkpoints/bge-small-v3-hn/` and the `_ft` corpus artifacts stay on disk and in git. Phase 12 will run the ablation properly. If Phase 13 error analysis points to retrieval as the bottleneck, we can revisit at n=1000 (tighter CIs) before any architecture changes.

## What the fine-tune did and didn't fix

V1's hypothesis was that mining hard negatives only from FEVER caused HoVer recall to drop 2.5 points. V3 mixed sources (5,995 HoVer + 3,973 FEVER claims → 39,872 triplets) and explicitly held HoVer dev as the validation target.

- **V3 didn't drop HoVer recall.** This alone is progress over V1. The mix-of-datasets fix works.
- **V3 didn't significantly improve HoVer recall either.** The +1.6% directional uplift is within noise at n=200.

So V3's hypothesis is *half-confirmed*: HoVer-mixing prevents regression, but it doesn't generate the kind of headline lift that would justify fine-tuning at all.

## Production config

`configs/default.yaml` keeps `retriever.finetune_path: null`. `DenseRetriever` continues to load `BAAI/bge-small-en-v1.5` as the encoder against `artifacts/corpus.faiss` (the Phase 01 baseline FAISS index). No change to `Pipeline.build_pipeline()` behaviour.

## Artifacts preserved (for Phase 12 ablation + future re-eval)

| Path | Bytes | Purpose |
|---|---|---|
| `checkpoints/bge-small-v3-hn/model.safetensors` | 133 MB | Fine-tuned encoder weights |
| `checkpoints/bge-small-v3-hn/{tokenizer.json,vocab.txt,*.json}` | <1 MB | Tokenizer + sentence-transformers config |
| `artifacts/corpus_embeddings_ft.npy` | 272 MB | Float32 [177317, 384] embeddings, L2-normalised |
| `artifacts/corpus_ft.faiss` | 272 MB | FAISS IndexFlatIP over the fine-tune embeddings |
| `artifacts/corpus_encoding_meta_ft.json` | <1 KB | Encoding metadata |
| `artifacts/retriever_finetune_log.jsonl` | <1 KB | Training hyperparameters + wall time |
| `artifacts/hard_negatives_v3.jsonl` | 18 MB | 39,872 triplets used for training |
| `artifacts/retriever_eval_finetune.json` | <2 KB | This decision's source-of-truth eval |

The 272 MB corpus artifacts are gitignored; they're regenerable from the checkpoint via `python -m src.data.encode_corpus --model checkpoints/bge-small-v3-hn --suffix _ft`. The checkpoint itself is in git LFS-style (committed without LFS because 133 MB safetensors is under GitHub's 100 MB hard limit per file — actually it's slightly over, so it'll need force-add with care or LFS).

> **Disk note:** the 133 MB `model.safetensors` does exceed GitHub's 100 MB single-file cap. We'll either commit via Git LFS or omit it from git and rely on the user re-running the Colab notebook to regenerate. Decision: omit from git; the notebook is the canonical reproduction path.

## How to revisit later

If Phase 13 error analysis or Phase 12 ablation suggests retrieval is the headline bottleneck, re-run the comparison at n=1000:

```bash
python -m src.eval.retrieval_eval_finetune --n 1000 --seed 42
```

With 5× more examples the CIs tighten by ~√5 ≈ 2.2×. If the fine-tune's CI lower bound moves above the base's CI upper bound, flip `retriever.finetune_path` in `configs/default.yaml` and ship.
