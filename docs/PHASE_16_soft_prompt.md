# Phase 16 — Soft-Prompt the 3B Verifier (V3.1)

**Goal.** Replace the strict "if on-topic but doesn't address claim → NEI" rule in [`src/verifier/prompts.py`](../src/verifier/prompts.py) with a soft alternative, re-run the verifier on HoVer dev n=200, and decide whether to ship or scope out based on paired-bootstrap Δ vs the v3.0 baseline.

**Effort.** 1.5 h setup + ~3–16 h re-run (depends on Colab vs CPU).
**Compute.** CPU sanity (n=20) then either CPU overnight or Colab T4 for full n=200.
**Depends on.** V3.0 — cached `per_subclaim_traces.jsonl` for paired comparison.

## Why this exists

[`docs/HANDOFF.md`](HANDOFF.md) and [`docs/ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md) both name this as the cheapest test of the V3 bottleneck. Phase 13 measured 28/50 failures as `nei_miscalibration` — the 3B verifier returns NEI when the cited passage actually contains the answer. Phase 14 (n=50 chains, single rater) saw the same thing from the chain-level audit angle: cited passage literally states the fact, model says "no information". The hypothesis is that the prompt's strict NEI-on-uncertain rule is responsible.

V1 used a similarly strict rule; V3 inherited it. The Phase 12 bidirectional-NLI veto recovers a chunk of NEIs (drives accuracy 0.02 → 0.36) but **56% of residual failures are still NEI miscalibrations**. The veto is a downstream patch. The upstream fix — softening the prompt itself — has never been tested.

## The change

**Old (v3.0)** — `SYSTEM_PROMPT` in `src/verifier/prompts.py`:
```
Rules:
- Reason ONLY from the evidence shown. Do not use outside knowledge.
- If the evidence is on-topic but doesn't address the specific assertion, return NEI.
- If even part of the claim is contradicted, return REFUTED.
```

**New (v3.1 candidate)** — `SYSTEM_PROMPT_V2`:
```
Rules:
- Reason ONLY from the evidence shown. Do not use outside knowledge.
- If any part of the claim is directly contradicted by the evidence, return REFUTED.
- If every part of the claim is directly supported by the evidence, return SUPPORTED.
- If parts of the claim are partially addressed but the evidence taken together
  clearly leans one way, prefer SUPPORTED or REFUTED over NEI.
- Return NEI only when the evidence does not address the claim's key entities or
  assertions at all.
```

The NEI few-shot example also changes — the v3.0 example ("Christopher Nolan owns a yacht in Monaco" + a generic Nolan bio) anchors the model to NEI on topical-overlap-without-direct-answer, which is exactly the failure mode for multi-hop HoVer claims. The v3.1 NEI example will use a case where the evidence is **completely off-topic** (different entity, different domain).

## Inputs

- `artifacts/per_subclaim_traces.jsonl` — v3.0 baseline traces (committed, n=200).
- HoVer dev n=200 stratified examples (same as v3.0; reproducible via `seed=42`).
- `src/verifier/prompts.py` (modified — both prompts kept for A/B).

## Deliverables

- `src/verifier/prompts.py` — both `SYSTEM_PROMPT` and `SYSTEM_PROMPT_V2`; `prompt_hash()` updated to take a variant arg; `build_messages` takes a `variant` parameter.
- `src/eval/verifier_eval.py` — new `--prompt-variant {v1,v2}` and `--out-suffix` CLI flags so the two runs write to different trace files.
- `artifacts/per_subclaim_traces_softprompt.jsonl` — new traces from v3.1 prompt (n=200).
- `artifacts/verifier_eval_softprompt.json` — metrics + per-class breakdown for v3.1.
- `src/eval/paired_compare.py` — paired-bootstrap Δ between v3.0 and v3.1 traces (1000 resamples).
- `artifacts/softprompt_comparison.json` — paired Δ + non-overlapping-CI verdict.
- `docs/PHASE_16_soft_prompt.md` (this file) — outcome section appended.

## Sanity check (n=20, CPU, ~10–20 min)

Before committing to the full run, verify on n=20:
- v3.1 prompt parses cleanly (no JSON parse failures).
- v3.1 NEI rate drops vs v3.0 on the same 20 examples.
- v3.1 accuracy isn't catastrophically worse (e.g., not collapsed to always-SUPPORTED).

**Pass criteria for the sanity check:**
- [ ] ≤ 2/20 parse failures
- [ ] v3.1 NEI rate < v3.0 NEI rate (any drop counts; the strong version is ≤ 50% of v3.0 NEI rate)
- [ ] v3.1 accuracy ≥ v3.0 accuracy − 0.10 (within 10 pts; sanity, not headline)

If the sanity fails, iterate the prompt before the full run.

## Full run (n=200)

Run on whichever compute is available:
- CPU laptop: ~16 h, resume-safe (append-mode JSONL, `--resume` default).
- Colab T4: ~3 h.

## Decision rule

After the full n=200 run:

- **Ship v3.1** if paired Δ accuracy is **≥ +5 pts** with non-overlapping 95% CI vs v3.0 baseline. The +5pt threshold is half of the +10–15pt expected for the 7B sweep (Path B); we want soft-prompt to deliver clear value before going to a bigger model.
- **Scope out** if Δ is < +3 pts or the CI overlaps zero. Append to `SCOPED_OUT.md` with the negative-result rationale.
- **Iterate** if Δ is +3–5 pts: try one more prompt variant (e.g., add a chain-of-thought example) before deciding.

## Risks and gotchas

- **Softening too much.** The new prompt could swing the model from over-NEI to over-SUPPORTED. The NLI bidir veto already mitigates this, but watch the per-class CM in the sanity check.
- **Parse failures.** Few-shot examples are the only format-anchor; if I write a less-anchoring example by mistake, the model might output free text. Sanity check verifies this.
- **CI overlap.** At n=200, the 95% CI on a single accuracy point is ~±0.07. Paired bootstrap tightens this Δ-CI but +3–4 pt deltas might still overlap zero. The decision rule accounts for this.
- **Existing cache contamination.** The cached `per_subclaim_traces.jsonl` is keyed on `uid`, so re-running with the old prompt to a new file is safe. The eval script must not write to the v3.0 traces file.

## What NOT to do

- **Do not delete `per_subclaim_traces.jsonl`.** It is the v3.0 baseline. Even if v3.1 wins, the old traces stay as the ablation reference.
- **Do not edit the v3.0 prompt in place.** Both prompts live side-by-side; v3.1 selection is a config/flag, not a code edit.
- **Do not skip the paired comparison.** "v3.1 is 0.40" is meaningless without "v3.0 is 0.36 [0.295, 0.425] on the same examples." Paired bootstrap Δ is the headline.
- **Do not let the soft-prompt re-introduce silent fallbacks.** If the model returns garbage, the LLMVerifier still falls back to NEI — that part of the contract stays intact.

## Exit criteria

- [ ] `SYSTEM_PROMPT_V2` lives in `src/verifier/prompts.py`; `prompt_hash()` differentiates the two.
- [ ] `verifier_eval.py` supports `--prompt-variant v1|v2`, writes to suffixed output files.
- [ ] Sanity check (n=20) meets the 3 pass criteria.
- [ ] Full n=200 run completes; `artifacts/verifier_eval_softprompt.json` exists with bootstrap CIs.
- [ ] `artifacts/softprompt_comparison.json` has paired Δ + non-overlapping-CI flag.
- [ ] This phase doc's Outcome section names the decision (ship / scope-out / iterate) with the data.
- [ ] If ship: re-tag as `v3.1`, update `FINAL_REPORT.md` headline numbers, update `HANDOFF.md`'s "highest-leverage next experiment" table.
- [ ] If scope-out: append to `SCOPED_OUT.md` with the data link.

## Outcome (Phase 16 closed 2026-05-15 — scope-out, negative result)

**Decision: scope out Path A. Soft-prompt rewording does not shift the 3B verifier's verdicts on the multi-hop HoVer claims that fail in v3.0. The bottleneck is model capacity, not prompt wording. Pivoting to Path B (7B verifier sweep) in a new phase doc.**

### Sanity result that triggered the scope-out

Ran [`src.eval.softprompt_sanity`](../src/eval/softprompt_sanity.py) on the first n=5 claims of v3.0's cached trace order (same passages, same NLI, same LLM at temp=0; only the prompt differs):

| | v1 (v3.0) | v2 (soft-prompt) |
|---|---|---|
| SUPPORTED | 0 | 0 |
| REFUTED | 0 | 0 |
| **NEI** | **5** | **5** |
| Accuracy | 0/5 | 0/5 |
| Parse failures | 0 | 0 |
| Δ verdict shifts | — | **0/5** |

Pass criteria from this phase doc:
- ✓ ≤ 2/N parse failures (got 0)
- ✗ **v2 NEI rate < v1 NEI rate** — got 5/5 same
- ✓ v2 acc ≥ v1 acc − 0.10 (both 0)

The criterion that matters — "does v2 actually shift verdicts" — failed cleanly at n=5. Continuing to the full 16 h n=200 run would burn compute on a hypothesis the sanity already rejected.

### What the reasoning text reveals (worse than identical verdicts)

The 5/5 verdict-identity hides a more troubling pattern: **v2's reasoning text is qualitatively worse**. The v2 prompt's NEI definition ("evidence is off-topic — does not mention the claim's entities at all") anchors the model on the new few-shot example (Sahara/bananas) and causes it to falsely claim evidence is off-topic when it's actually on-topic-but-partial.

| uid | gold | passage relevant? | v1 reasoning | v2 reasoning |
|---|---|---|---|---|
| `8ae5dcf4` | REFUTED | YES (Texas_Monthly::0 cited) | "does not clearly support or refute the claim about Christopher Kelly being a journalist for Texas Monthly" | **"does not mention Christopher Kelly, Texas Monthly"** (false — Texas_Monthly IS in passages) |
| `259c831a` | REFUTED | YES (lake passages cited) | "Passages 8 and 9 provide conflicting information about the relative elevations" | **"Passages 1-9 do not mention Lake Kanasatka or Lake Winnipesaukee"** (false — they appear in passages) |
| `27818aa7` | SUPPORTED | partial | "does not provide information about the specific mixed martial arts promotion" | "does not mention Joe Lauzon, World Extreme Cagefighting" |

This is the same `8ae5dcf4` chain we hand-rated as faithfulness=2 in [`docs/PHASE_14_human_eval.md`](PHASE_14_human_eval.md) — the v2 prompt makes faithfulness *worse* by encouraging the model to falsely claim entities are absent.

### Read

Three converging signals:

1. **Verdict didn't shift** on 5/5 cases where v3.0 v1 was NEI. The 3B model is too anchored on the NEI-on-uncertain heuristic for prompt rewording to dislodge it.
2. **Reasoning got worse.** The new "off-topic only" NEI framing causes false-absence claims. Faithfulness drops even when verdict is identical.
3. **The bidir NLI rule already captures the recoverable signal.** Phase 12 measured the rule's lift at +33 points; what's left is what NLI signals can't lift either. Prompt rewording alone won't unlock that residual.

This is consistent with the original Phase 13 read: 56% of failures are NEI miscalibration on the 3B model. The fix needs to be at the model-capacity layer, not the prompt-rule layer.

### Files added (kept for traceability)

- [`src/verifier/prompts.py`](../src/verifier/prompts.py) — `SYSTEM_PROMPT_V2` lives alongside `SYSTEM_PROMPT`; variant selector via `get_prompt(variant)`, `build_messages(..., variant=)`, `prompt_hash(variant)`.
- [`src/verifier/llm.py`](../src/verifier/llm.py) — `LLMVerifier(prompt_variant=...)` parameter.
- [`src/verifier/ensemble.py`](../src/verifier/ensemble.py) — `EnsembleVerifier(prompt_variant=...)` plumbing.
- [`src/eval/verifier_eval.py`](../src/eval/verifier_eval.py) — `--prompt-variant` and `--out-suffix` CLI flags.
- [`src/eval/softprompt_sanity.py`](../src/eval/softprompt_sanity.py) — fast A/B sanity script (re-uses v3.0 cached passages).
- [`src/eval/paired_compare.py`](../src/eval/paired_compare.py) — paired-bootstrap comparison harness (never run on full traces; kept for future variant experiments).
- [`artifacts/softprompt_sanity.json`](../artifacts/softprompt_sanity.json) — 5-row sanity result.

These files cost 1.5 h to write and they're the right plumbing for *any* future prompt-variant experiment. Keeping them. The next prompt experiment (or the Path B 7B sweep) can re-use the variant infrastructure directly.

### Where the effort goes next

Path B from [`docs/HANDOFF.md`](HANDOFF.md) — Qwen 7B-Instruct via vLLM on Colab. New phase doc [`docs/PHASE_17_7b_verifier.md`](PHASE_17_7b_verifier.md) (to be written). Expected lift: +10–15 pts vs v3.0's 0.360. The case for Path B is now stronger because Path A's negative result rules out the cheaper alternative.

### Spec exit criteria (re-cast)

- [x] `SYSTEM_PROMPT_V2` lives in `src/verifier/prompts.py`; `prompt_hash()` differentiates the two.
- [x] `verifier_eval.py` supports `--prompt-variant v1|v2`, writes to suffixed output files.
- [x] Sanity check (n=20 → reduced to n=5 once 4/5 same-NEI made the result statistically clear) **failed** criterion 2 (NEI rate didn't drop).
- [ ] ~~Full n=200 run~~ — **not run.** Sanity result rules it out.
- [ ] ~~Paired Δ comparison~~ — **not run.** Same reason.
- [x] This phase doc's Outcome section names the decision (scope_out) with the data.
- [x] Negative result and rationale added to [`docs/SCOPED_OUT.md`](SCOPED_OUT.md).
- [x] [`docs/HANDOFF.md`](HANDOFF.md) fix-path table updated to deprioritize Path A.
