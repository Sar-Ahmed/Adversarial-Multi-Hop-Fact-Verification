# Phase 07 — Cross-Document Verifier (LLM + NLI Ensemble)

**Goal.** Replace `VerifierStub` with a real ensemble: Qwen 3B Q4 LLM verdict + DeBERTa NLI cross-encoder veto, optionally per-sub-claim CoT. Spec requirement #3.

**Effort.** 2 days CPU; +1 day if the optional Colab 7B sweep is run.
**Compute.** CPU baseline. **Colab T4/A100 optional** for the 7B sweep (only if the CPU 3B accuracy stays below 55% after the NLI veto).
**Depends on.** Phase 03 (decomposer), Phase 04 (retrieval+rerank), Phase 06 (distractors for eval-mode robustness).

## Why this exists

V1 settled on LLM + NLI veto as the production config because the LLM alone has a 74%-SUPPORTED bias (76% of failures are REFUTED→SUPPORTED). The NLI veto rule (`if LLM=SUPPORTED ∧ NLI_contra_prob ≥ 0.98 then flip to REFUTED`) added ~4–6 points of accuracy. V3 keeps this but tests one new variant V1 didn't: per-sub-claim CoT *post-NLI-veto*. V1's Phase 5 CoT failed because the 3B model emits ~42% NEI per sub-claim and any-NEI→NEI aggregation collapses; if the NLI veto fires before aggregation, the picture might be different.

## Inputs

- `src/decomposer/` from Phase 03 (sub-claims).
- `src/retrieval/` + `src/reranker/` from Phase 04 (top-10 passages).
- Qwen2.5-3B-Instruct Q4_K_M GGUF.
- `cross-encoder/nli-deberta-v3-base`.

## Deliverables

- `src/verifier/llm.py` — `LLMVerifier.verify(claim, passages) -> (Label, float, str)` returning (verdict, confidence-as-logprob-derived, reasoning text).
- `src/verifier/nli.py` — `NLIVerifier.score(claim, passages) -> dict[passage_id, {entail, contra, neutral}]`.
- `src/verifier/aggregate.py` — combines LLM + NLI signals + retrieval features into a single verdict per sub-claim. Implements three modes: `llm_only`, `llm_plus_nli_veto`, `per_subclaim_cot`.
- `src/verifier/prompts.py` — verifier system prompt, claim-passages format, JSON schema for the output.
- `notebooks/phase07_verifier_sweep.ipynb` — optional Colab 7B FP16 sweep using vLLM.
- `artifacts/verifier_eval_phase07.json` — accuracy + macro-F1 for each mode on HoVer dev with 95% CIs.
- `artifacts/per_subclaim_traces.jsonl` — per-(claim, sub-claim) (verdict, llm_reasoning, NLI scores, retrieval scores) for downstream analysis.

## Technical approach

- **LLM verifier (per claim or per sub-claim).**
  - Prompt: system message + claim + numbered passages (`[1] ... [10] ...`) + instruction "Respond with JSON: {verdict: SUPPORTED|REFUTED|NEI, confidence: 0..1, reasoning: ...}".
  - Greedy decoding (`temperature=0`), `max_tokens=256`, with logprobs.
  - Parse JSON; on parse failure return `Label.NEI` with confidence 0 and a `parse_failure` flag.
- **NLI verifier (per (claim, passage) pair).**
  - Score each of the top-10 passages against the claim with `cross-encoder/nli-deberta-v3-base`.
  - Aggregate over passages: `max_contra`, `max_entail`, `max_neutral`.
- **Aggregation modes.**
  - `llm_only`: take LLM verdict.
  - `llm_plus_nli_veto`:
    - If `LLM=SUPPORTED` and `max_contra ≥ 0.95`: flip to REFUTED.
    - If `LLM=SUPPORTED` and `max_entail < 0.5`: downgrade confidence by 0.2.
    - Else: keep LLM verdict.
  - `per_subclaim_cot`:
    - Run LLM + NLI veto for each sub-claim independently.
    - Aggregate: if any sub-claim is REFUTED, final is REFUTED. Else if all SUPPORTED, final is SUPPORTED. Else NEI.
    - **Variant A** (the V1 failure mode): aggregate without NEI absorption.
    - **Variant B** (V3 hypothesis): a sub-claim verdict of NEI is *abstained* — overall verdict computed from the non-NEI sub-claims if at least one exists.
- **Final NEI signal** is *not* set by this phase; Phase 08 owns the NEI calibrator that converts (LLM verdict, NLI features, retrieval features) → calibrated 3-way label. Phase 07 emits raw verdicts + features.

## Implementation steps

1. Implement `LLMVerifier` with prompt, parsing, retry-once.
2. Implement `NLIVerifier` with batched cross-encoder scoring.
3. Implement `aggregate.py` with the three modes; pure functions, no model loading.
4. Wire into `pipeline.py`; default mode `llm_plus_nli_veto`. Add `verifier.mode` to `PipelineConfig`.
5. Run the three modes on HoVer dev (full ~4k or n=200 subset for fast iteration) → save eval JSONs.
6. If accuracy of best mode < 0.55: open `notebooks/phase07_verifier_sweep.ipynb`, run Qwen 7B FP16 via vLLM on Colab (~1k claims, 1–2 h on T4). Compare. If 7B helps materially (>3 points), document and consider for production *only* if the latency budget tolerates it (it likely won't on CPU).
7. Save `per_subclaim_traces.jsonl` for Phase 08 / 13 consumption.

## Exit criteria

- [ ] `make smoke` passes with the real ensemble verifier.
- [ ] `llm_plus_nli_veto` accuracy ≥ 0.55 on HoVer dev (V1 hit ~0.575–0.60 on test split; we want at least V1's number).
- [ ] Macro-F1 ≥ 0.40 on HoVer dev.
- [ ] Per-mode comparison saved with bootstrap CIs.
- [ ] If `per_subclaim_cot` (variant B) beats the simple ensemble by >2 points: document the result and switch the default mode. Else: document the negative result and keep the simple ensemble.
- [ ] `per_subclaim_traces.jsonl` populated for every dev example (Phase 08 will consume this).

## Risks and gotchas

- The 3B model's bias is known and cannot be fully prompted away. Don't spend more than half a day on prompt engineering; the NLI veto is the right lever.
- DeBERTa NLI's contradiction probability is well-calibrated for clean entailment but less so for "passage doesn't address claim" (the NEI confusion). Phase 08 fixes this; don't try to fix it here.
- Logprob extraction from llama-cpp can stall under certain settings — set `logits_all=False` and only request top-k logprobs at the verdict token.
- vLLM on Colab requires specific torch/cuda combinations — pin them in `requirements-colab.txt`. The notebook should self-contain pip installs.

## What NOT to do

- Do not mix the calibrator into the verifier. Phase 08 is a separate, learned post-processor.
- Do not hard-code the NLI veto threshold. Read from `PipelineConfig.verifier.contra_veto_threshold` (default 0.95).
- Do not silently fall back to neutral NLI scores if the NLI model fails to load (V2's bug). Raise.

## Outcome (filled at end of phase)

> Append: per-mode accuracy + macro-F1 + 95% CIs, whether variant B post-NLI CoT helped, whether the 7B sweep was run, final default mode chosen.
