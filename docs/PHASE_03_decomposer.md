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

## Outcome (filled at end of phase)

> Append: actual fallback rate, prompt hash, per-claim avg sub-claim count, any regressions to the smoke test.
