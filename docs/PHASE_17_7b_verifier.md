# Phase 17 — 7B Verifier Sweep (V3.1, Path B)

**Goal.** Swap the production 3B Qwen verifier for Qwen 2.5-7B-Instruct, re-run the verifier on HoVer dev n=200, paired-bootstrap Δ vs v3.0, decide ship-or-scope-out. Expected lift: +10–15 pts (per [`docs/HANDOFF.md`](HANDOFF.md) and [`docs/ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md)).

**Effort.** 1 h setup + ~30 min Colab T4 run + 1 h analysis = ~2.5 h.
**Compute.** Colab T4 (mandatory — 7B Q4 on CPU is ~10 min/inference, prohibitive for 200 claims).
**Depends on.** V3.0 cached traces (`per_subclaim_traces.jsonl`) for paired comparison; Phase 16's `--prompt-variant` and `--out-suffix` plumbing (we reuse `v1` prompt with the bigger model).

## Why this exists

[Phase 16](PHASE_16_soft_prompt.md) ruled out prompt rewording as the fix for the 3B verifier's NEI miscalibration. The remaining hypothesis is **model capacity**: the 3B is just too small to commit to a verdict on multi-hop claims. The handoff doc names a 7B sweep as the most-likely-to-work next experiment.

The Phase 13 error analysis (56% nei_miscalibration on 50 stratified failures) and Phase 14 human eval (reasoning rated 2.58/5; cited passages contain answers the model denies) both implicate the verifier. The Phase 16 sanity result rules out the prompt as the lever. The model size is the next lever.

## Why the cached infrastructure makes this cheap

**The 7B sweep does not need to re-run retrieval or NLI.** Phase 07's traces cache per-claim `passage_doc_ids` and `nli.{max_contra,max_entail,...}`. NLI scores are independent of the LLM verifier — they're just DeBERTa-NLI over `(claim, passage)` pairs. Only the LLM call needs to change.

This means the Colab notebook just iterates the cached v3.0 traces, swaps in a 7B LLMVerifier, and writes new `(llm.verdict, llm.reasoning)` while copying `nli` from v3.0. Local paired-bootstrap comparison then uses the existing `src/eval/paired_compare.py` (from Phase 16).

## Inputs

- `artifacts/per_subclaim_traces.jsonl` — v3.0 baseline (committed; provides claim/gold/passage_doc_ids/cached NLI).
- `artifacts/corpus.parquet` — passage texts by doc_id (gitignored; uploaded to Colab).
- `BAAI/bge-reranker-base` — not needed; we reuse the cached top-K passages.
- `Qwen/Qwen2.5-7B-Instruct` — pulled from HuggingFace at notebook runtime.

## Deliverables

- [`notebooks/phase17_7b_verifier_sweep.ipynb`](../notebooks/phase17_7b_verifier_sweep.ipynb) — Colab T4 notebook. Pulls the repo, downloads Qwen 7B, runs verifier on n=200 cached claims, writes `per_subclaim_traces_7b.jsonl`, zips for download.
- `artifacts/per_subclaim_traces_7b.jsonl` — new traces. Same schema as v3.0; only `llm.*` differs (and `model_size: "7b"` in the row).
- `artifacts/sevenb_comparison.json` — paired Δ via `src/eval/paired_compare.py --v2-traces artifacts/per_subclaim_traces_7b.jsonl`.
- Phase 17 outcome appended to this doc with decision (ship as v3.1 / scope-out / iterate).
- If ship: `configs/default.yaml` updated with the 7B model path; `FINAL_REPORT.md` updated with new headline numbers; tag `v3.1`.

## Sanity check (n=5 in the same Colab session, ~5 min)

Before the full n=200, the notebook first runs on 5 claims and checks:
- ≤ 1/5 parse failures (5/5 = 80%+ parse rate)
- v2 verdict distribution shifts vs v1's 5/5 NEI seen in Phase 16 sanity (if it doesn't, scope-out immediately — 7B isn't enough either)

If sanity passes, the full n=200 runs in the same session.

## Decision rule

Same rule as Phase 16:
- **Ship as v3.1** if paired Δ accuracy ≥ +5 pts with non-overlapping 95% CI vs v3.0.
- **Iterate** if Δ is +3-5 pts with overlapping CI (e.g., try a different 7B variant or add CoT prompting).
- **Scope out** if Δ < +3 pts. At that point the bottleneck isn't the LLM at all — it's NLI signals plus a 50-50 prior on multi-hop, which suggests an entirely different fix path (Path C, entity-aware retrieval, or even Phase 03 decomposition changes).

## Technical approach

The notebook is structured to mirror Phase 05's pattern (anonymous HTTPS clone → minimal inline deps → inline execution to dodge typer CLI clashes → zip+download). Key cells:

1. **GPU + repo check.** `torch.cuda.is_available()`, `git clone https://github.com/.../Adversarial-Multi-Hop-Fact-Verification.git`.
2. **Install.** vLLM (or HF transformers + bitsandbytes for Q4) — vLLM is preferred for batched inference but transformers is the fallback if vLLM hits a Python/CUDA version mismatch in the Colab default image.
3. **Upload corpus.parquet.** ~12 MB; same flow as Phase 05's cell 13.
4. **Load 7B model.** Qwen 2.5-7B-Instruct. Memory check: 7B in fp16 needs ~14 GB; T4 has 16 GB. Quantize to Q4/NF4 via bitsandbytes if OOM.
5. **Verify + cache.** Iterate `per_subclaim_traces.jsonl`, re-build passages from `corpus.parquet`, call the 7B with the v1 prompt template, write new traces.
6. **Sanity inline.** First-5 verdict counts + parse-failure rate; abort if criteria fail.
7. **Full run.** Same loop, n=200 with progress logging.
8. **Zip + download.** `per_subclaim_traces_7b.jsonl` (gold included so paired_compare.py works offline).

The verifier prompt stays at `v1` for this experiment — we want the *only* changing variable to be model size. Once the 7B is wired in, a future iteration can also test `v2` prompt on 7B if Path A's negative result was 3B-specific.

## Local re-aggregation (after the Colab run)

```bash
# Drop the downloaded traces into artifacts/
python -m src.eval.paired_compare \
    --v2-traces artifacts/per_subclaim_traces_7b.jsonl \
    > artifacts/sevenb_comparison.json
```

Output JSON has `paired.delta`, `paired.delta_ci`, `decision`. Same fields as Phase 16's softprompt_comparison.json (the script is generic).

## Risks and gotchas

- **VRAM ceiling.** 7B fp16 ≈ 14 GB; T4 has 16 GB. If we hit OOM, fall back to 4-bit quantization via `bitsandbytes`. Document the quantization in the trace summary so the comparison isn't apples-to-oranges with the 3B Q4_K_M.
- **Token budget.** 7B at fp16 generates ~30 tok/sec on T4. 200 claims × ~80 output tokens = ~10-15 min generation, plus prefill (~30 sec/claim × 200 = ~100 min worst case). Total: 30 min - 2 h. vLLM with continuous batching cuts this to ~10-20 min.
- **Parse failures.** Larger models sometimes output verbose JSON-with-prose. The existing `_parse_verdict` in `src/verifier/llm.py` extracts the first JSON object; that should still work, but watch the parse-failure rate in the sanity check.
- **The 7B is also fine-tuned by Alibaba, so its formatting prior is strong.** It should follow the same JSON format the 3B follows. If not, document and iterate prompt — but the prompt change in that case is *format-anchoring*, not the soft-prompt change that failed in Phase 16.
- **Apples-to-apples passages.** The 7B sees the same top-K passages as the 3B did in v3.0 (cached doc_ids). This is the right paired setup — we measure ONLY the LLM-component lift.

## What NOT to do

- **Do not run retrieval or NLI again in Colab.** Both are cached in v3.0 traces. Re-running adds compute cost and a new source of variance.
- **Do not switch the prompt variant for this experiment.** Phase 16's variant infrastructure is there if a future iteration wants to test prompt-vs-7B interactions; for THIS phase the only variable is model size.
- **Do not edit `configs/default.yaml` to point at the 7B until the decision is made.** Run, measure, then decide.
- **Do not skip the n=5 sanity in Colab.** If the 7B also can't shift verdicts, knowing that in 5 minutes is much cheaper than after the full n=200 run.

## Exit criteria

- [ ] [`notebooks/phase17_7b_verifier_sweep.ipynb`](../notebooks/phase17_7b_verifier_sweep.ipynb) committed.
- [ ] Colab session executes n=5 sanity successfully (≤ 1 parse failure; verdict distribution different from v3.0's 5/5 NEI).
- [ ] Full n=200 run completes; `artifacts/per_subclaim_traces_7b.jsonl` committed.
- [ ] `python -m src.eval.paired_compare --v2-traces artifacts/per_subclaim_traces_7b.jsonl` produces `artifacts/sevenb_comparison.json` with `decision` ∈ `{ship_v31, scope_out, iterate_prompt}`.
- [ ] This phase doc's Outcome section has the paired Δ + 95% CI + decision.
- [ ] If ship: `configs/default.yaml`, `FINAL_REPORT.md` headline numbers, `HANDOFF.md` fix-path table, and the `v3.1` git tag all reflect the new verifier.
- [ ] If scope-out: section added to [`docs/SCOPED_OUT.md`](SCOPED_OUT.md) with the data; HANDOFF.md updated to point next-iteration effort at Path C (entity-aware retrieval) instead.

## Outcome (filled at end of phase)

> Append: paired Δ accuracy + 95% CI, per-class CM, sanity-run distribution, parse-failure rate, model variant (fp16 vs 4-bit), Colab wall time, decision, link to artifacts JSON.
