# Phase 03 — Claim Decomposer

**Goal.** Replace `DecomposerStub` with a real Qwen2.5-3B-Instruct Q4_K_M decomposer that produces a list of `SubClaim` objects with valid `depends_on` edges. Spec requirement #1.

**Effort.** 1–2 days.
**Compute.** CPU (llama-cpp Q4 GGUF runs at ~10–30 tok/s on a modern laptop CPU).
**Depends on.** Phase 02.

## Why this exists

Compound claims like *"The director of Inception also directed a Batman movie that grossed over $1 billion"* must be split into atomic sub-claims that the verifier can check independently. V1 demonstrated that a few-shot prompt against Qwen 3B Q4 with regex JSON extraction and a wrap-in-single-claim fallback achieves a 4.07/5 human quality rating and 0% fallback rate on n=200. We replicate that and bring V2's better dataclass schema along.

## Inputs

- `src/schema.py` from Phase 02 (`SubClaim`, `ReasoningType`).
- Qwen2.5-3B-Instruct-GGUF Q4_K_M from `scripts/fetch_models.sh`.
- 30 manually-decomposed claims (from V1's `decomposer_eval.json` if available, or hand-write fresh) for evaluation.

## Deliverables

- `src/decomposer/decomposer.py` — `Decomposer.decompose(claim) -> list[SubClaim]`.
- `src/decomposer/prompts.py` — system prompt + 5–7 few-shot examples covering compound, dependent, comparison, negation, and temporal patterns.
- `src/decomposer/llm.py` — thin wrapper around `llama_cpp.Llama` with `chat()` and `chat_with_logprobs()` methods.
- `src/decomposer/eval_decomposer.py` — runs decomposer on the 30-claim eval set, dumps `artifacts/decomposer_eval.json`.
- `tests/test_decomposer.py` — unit tests for parser, fallback path, and end-to-end against 3 fixed claims.

## Technical approach

- **Backbone.** Qwen2.5-3B-Instruct Q4_K_M via `llama_cpp.Llama(model_path, n_ctx=4096, logits_all=False)`.
- **Prompt.** System message defines task + JSON schema + reasoning_type taxonomy. Few-shot block of 5–7 examples covering:
  - Compound: "X did Y and Z" → 2 independent sub-claims.
  - Dependent: "the director of X also did Y" → 2 sub-claims with `depends_on=[id_of_first]`.
  - Comparison: "X grossed more than Y" → 3 sub-claims (X gross, Y gross, comparison).
  - Negation: "X never did Y" → preserves negation in sub-claim text.
  - Temporal: "X did Y in 2008" → tags `reasoning_type=temporal`.
- **Output format.** Greedy decoding (temperature=0), `max_tokens=512`, JSON list of objects with `{id, text, depends_on, reasoning_type}`.
- **Parsing.** Regex-extract the first JSON array between `[` and `]`. If parse fails, retry once with a stricter "respond with JSON only" prompt. If retry fails, fall back to single-SubClaim wrapping the entire claim.
- **Validation post-parse.** Check `depends_on` ids reference earlier sub-claims (DAG, no cycles). If not, drop the offending edge with a warning log.

## Implementation steps

1. Add Qwen GGUF to `scripts/fetch_models.sh`. Verify `llama_cpp.Llama` loads it (cold start ~10 s).
2. Write `prompts.py` with examples lifted/adapted from V1's `decomposer/examples.py`.
3. Implement `Decomposer.decompose()`:
   - Build chat messages.
   - Call `llm.chat(...)`.
   - Parse JSON.
   - Validate DAG.
   - Construct `SubClaim` instances (with **`id=...`**, not `sub_claim_id` — explicit regression test in Phase 02 already covers this).
4. Implement retry-once and single-claim fallback.
5. Build the 30-claim eval set with hand-written gold decompositions in `artifacts/decomposer_eval_gold.json`.
6. Write `eval_decomposer.py` — emits per-claim parse success, fallback rate, sub-claim count distribution. (Quality rating is manual / Phase 14-style; defer narrative writeup.)
7. Update `pipeline.py` to swap `DecomposerStub` for `Decomposer`. Re-run `make smoke`.

## Exit criteria

- [ ] `make smoke` still passes (5 claims now run through the real decomposer).
- [ ] Fallback rate ≤ 5% on the 30-claim eval set.
- [ ] Average sub-claim count per claim is between 1.5 and 4.5 (sanity bounds).
- [ ] No invalid `depends_on` edges in the output (Phase 02 schema validation enforces this; should never trigger).
- [ ] `artifacts/decomposer_eval.json` committed with model name, prompt hash, seed, and per-claim outputs.

## Risks and gotchas

- llama-cpp parse failures often look like *truncated* JSON — set `max_tokens=512` and stop_sequence on `\n\n`.
- Few-shot prompts that include the negation example sometimes cause the model to over-negate downstream claims. V1 noted this; mitigate by ensuring the negation example is the third or fourth example, not the first.
- The Q4 quant produces deterministic-looking but actually slightly random output across machines (different ARM vs x86 BLAS); document seed + hardware for reproducibility.

## What NOT to do

- Do not introduce grammar-constrained decoding (Outlines / GBNF) here. V1 evaluated and rejected; regex + retry + fallback is enough for the observed parse-success rate.
- Do not chain the decomposer with the verifier yet — Phase 07 owns aggregation. Decomposer's job ends at producing valid `SubClaim`s.
- Do not hardcode the model path. Read from `PipelineConfig.decomposer.llm_path`.

## Outcome (Phase 03 closed 2026-05-06)

**Wall time.** ~3 hours code + ~10 min Qwen download + 30-claim eval (3.7 min after warm cache) + smoke test (69 s).

**Headline numbers (`artifacts/decomposer_eval.json`).**

| Metric | Value | Target | Status |
|---|---|---|---|
| Fallback rate | **0.0%** (0 / 30) | ≤ 5% | ✅ matches V1's 0% on n=200 |
| Avg sub-claims | **1.8** | 1.5–4.5 | ✅ |
| Min / max sub-claims | 1 / 3 | within bounds | ✅ |
| Avg latency / claim | 7.3 s (warm cache) | — | ~118 s on first run; KV cache reuse is huge |
| Total eval time | 219 s for 30 claims | — | acceptable |
| Prompt hash | `672f68862cc1` | — | logged for traceability |

**Smoke test.** `pytest -m smoke` — **7 passed in 69 s** (was 80 s in Phase 02 with the stub; Qwen decomposer adds <5 s amortised across 5 claims because the model loads once via the module-scoped fixture).

**Spec exit criteria.**

- [x] `make smoke` still passes (real decomposer in pipeline).
- [x] Fallback rate ≤ 5%: achieved 0%.
- [x] Avg sub-claim count ∈ [1.5, 4.5]: achieved 1.8.
- [x] No invalid `depends_on` edges (enforced by `SubClaim.__post_init__`; eval ran 30 claims with zero exceptions).
- [x] `artifacts/decomposer_eval.json` committed with model name, prompt hash, seed (42), per-claim outputs.

**Bug found and fixed during the eval.**

First eval run reported a fake 36.7% fallback rate. The cause was the eval's `_is_fallback` heuristic — it flagged any single-sub-claim output whose text equalled the input as a fallback. That conflated two different cases:
1. Real fallback: parser failed twice and the decomposer wrapped the input verbatim.
2. Correct atomic decomposition: model recognised the claim was already atomic and emitted exactly one sub-claim with the proper `reasoning_type`.

Fix: added an explicit `Decomposer.last_call_used_fallback` attribute set inside the fallback path; eval reads that instead of the heuristic. Real fallback rate after the fix: **0/30**. Lesson recorded for future phases — never infer behaviour from output shape when the producing component already knows the truth.

**Decomposition quality observations (qualitative, no formal target).**

- Compound (`X and Y`): always splits cleanly into independent sub-claims.
- Composition (`X who did Y also did Z`): emits 2 sub-claims with `depends_on=[0]` as designed.
- Comparison: model sometimes returns 1 atomic sub-claim with `reasoning_type=comparison` instead of the 3-step decomposition shown in the few-shot. Acceptable — the verifier (Phase 07) handles either shape; quality refinement waits for Phase 13 error analysis.
- Negation: polarity preserved every time.
- Atomic lookups: emitted as a single sub-claim with `reasoning_type=lookup`. Correct behaviour.

**Latency note.**

First call on the cold model takes ~120 s on CPU (Qwen 3B Q4, ~1500-token few-shot context). Subsequent calls average 7 s thanks to llama-cpp's automatic KV-cache reuse for the shared prefix. This is the practical reason every component that consumes the LLM should hold one `LocalLLM` instance and stream calls through it — instantiating a new one per call is a 17× slowdown.

**Open follow-ups.**

- Phase 07 will reuse `LocalLLM` for the verifier (different prompt, same instance pattern).
- Phase 13 error analysis will surface whether the comparison-pattern under-decomposition matters for end-to-end accuracy. If yes, revisit the comparison few-shot example.
- The 30-claim eval set is hand-curated; Phase 13 may motivate enlarging it.
