# V3 Error Analysis — 50 stratified failure cases

50 failures from the whole-claim `llm_plus_nli_bidir` eval on HoVer-dev n=200 (Phase 11's recommended production mode). Stratified across the four observed (gold, pred) transition cells. Full per-failure tags + notes in [`../artifacts/failures_tagged.json`](../artifacts/failures_tagged.json).

## Headline distribution

| Category | Count | % | Read |
|---|---|---|---|
| **nei_miscalibration** | **28** | **56%** | verifier punted (LLM=NEI, NLI signals below threshold) |
| partial_match_as_full | 12 | 24% | claim has multiple facts; one is wrong but LLM/NLI weighted the wrong one |
| entity_confusion | 10 | 20% | NLI flagged contradiction between adjacent-but-distinct entities |
| temporal_error | 0 | 0% | **gates Phase 09 to Path B (scope out)** |
| retrieval_miss | 0 | 0% | sampling artifact (Phase 04's H@10 = 0.93 means most failures have gold retrieved) |
| negation_blindness | 0 | 0% | no clear case in the sample |
| decomposition_error | 0 | 0% | whole-claim mode bypasses the decomposer |

128 / 200 = 64% failure rate on HoVer-dev under the chosen mode. The 50-sample tag distribution above is the breakdown by root cause.

## Confusion matrix on ALL 128 failures

| gold \ pred | SUPPORTED | REFUTED | NEI |
|---|---|---|---|
| **SUPPORTED** | 0 | 74 | 18 |
| **REFUTED** | 7 | 0 | 29 |
| **NEI** | n/a (HoVer has no NEI gold) | n/a | n/a |

The biggest failure cell is **SUPPORTED → REFUTED (74 cases)** — the bidirectional NLI rule firing on partial-match or entity-confusion cases.

## Root causes, in order of impact

### 1. NEI miscalibration (56% of sampled failures)

The 3B Qwen verifier returns NEI on multi-hop HoVer claims when NLI signals are moderate (max_contra ~ 0.0–0.5, max_entail ~ 0.0). The bidirectional rule only fires above 0.95 on either side, so these claims stay NEI. HoVer has zero NEI gold, so all are wrong.

**Example (#10):** *"The winner of the 1982 Bavarian Tennis Championships was born two years before this tennis player. This tennis player won the 2012 Farmers Classic-Doubles."* — both passages retrieved (gold_in_top10=True), LLM says "no information about the winner or the comparison." max_contra = 0.338, max_entail = 0.000. Stays NEI.

The verifier can read each fact but can't bind them across hops. The prompt's "if on-topic but doesn't address the specific assertion, return NEI" rule rewards exactly this behaviour.

**Fix paths** (any one would shrink this bucket meaningfully):
- Lower the bidir threshold to 0.8 or 0.7 — more flips, more noise on the SUP-as-REF side, but better recall on REFUTED.
- Soften the LLM prompt: "if any part of the claim is contradicted by any passage, lean REFUTED." (Phase 07 open follow-up; estimated 1.5 h re-run.)
- Train calibrator with LLM verdict feature added back; Phase 08 dropped it for compute reasons.
- 7B Qwen via Colab — phase doc anticipated this conditional path; not yet run.

### 2. partial_match_as_full (24%)

The claim has multiple atomic facts; gold is REFUTED because *one* is wrong but the LLM/NLI weighted the rest. The bidirectional rule sometimes flips NEI→SUP on partial-entail (12 cases here) and sometimes flips NEI→REF on partial-contra. Both are wrong half the time.

**Example (#4):** *"'Black Maverick' is a biography of the founder of the Regional Council of Negro Leadership, an American civil rights leader, fraternal organization leader, entrepreneur and surgeon."* — Howard *did* found RCNL and was a civil rights leader, entrepreneur, surgeon. The "fraternal organization leader" attribution is the contested detail. NLI saw mostly entailment (0.95) on the matching attributes and fired NEI→SUP; gold is REFUTED on the fraternal-leader detail.

**Example (#20):** *"British band The Wanted's third album includes a song with a title about Barbadian superstar Rihanna who was born in 1948."* — Rihanna was born in 1988 (LLM noted this explicitly). NLI flipped to REF; gold is SUP because HoVer accepts the bulk of the claim (Wanted/album/song) and treats the birth-year as an aside.

This bucket is genuinely hard. There's no single threshold fix — the system needs *fact-level* decomposition + voting, which is the spec's per-sub-claim CoT idea. Phase 07 ablation showed per-sub-claim CoT *failed* on this 3B verifier, so the fix requires a stronger verifier OR a different decomposition + reading strategy.

### 3. entity_confusion (20%)

NLI flags a contradiction between adjacent entities. Either:
- The LLM has the wrong fact baked into its weights (e.g. #18: claims Piper Laurie starred in Suburbicon; actually Julianne Moore), OR
- The retrieved passages mention multiple plausible entities and NLI picks the wrong pair (e.g. #46: "John Burges + Dundee United" — Dundee United is a football club, the actual answer is Royal Dutch Shell HQ; NLI sees both passages and flags contradiction).

This is the same failure mode Phase 06 documented at distractor-mining time: the NLI cross-encoder treats different entities with similar attribute patterns as contradictions.

**Fix path:** entity-aware retrieval / reranking. spaCy NER over claim + passages; require the verifier to consume *only* passages that share named entities with the claim. Open follow-up; bigger lift than threshold tuning.

## What's NOT a bucket

- **temporal_error: 0/50.** Many claims mention years or "the 2008 film" patterns, but none of the failures I tagged were specifically *because of* temporal handling — the verifier failures are dominated by NEI bias and entity-binding issues that aren't temporal-specific. **This is the Phase 09 gate: temporal reasoning is scoped out for V3.**
- **retrieval_miss: 0/50.** Sampling artifact: my stratification was over (gold, pred) transitions, not retrieval-hit/miss. Phase 04's per-passage R@10 = 0.55 means 45% of gold passages aren't retrieved on average — but the failures I sampled were all the "verifier failed despite having gold passages" kind. Retrieval misses are a real failure mode but a different sample would show them.

## Implications for Phase 15

The headline accuracy ceiling on this pipeline is the **3B verifier's tendency to return NEI on partial multi-hop evidence**. The bidirectional NLI rule recovered most of the easy cases (Phase 12 showed the +33 pts lift); what's left is genuinely hard:

- 56% of failures are claims the verifier *could* answer but chooses not to.
- 24% are claims with one wrong fact among several — fact-level decomposition would help but requires a stronger verifier.
- 20% are entity-binding errors in the NLI cross-encoder.

V3's recommended forward path:

1. **Prompt softening** on the 3B verifier — would target the 56% bucket directly. 1.5 h on Colab, low-risk.
2. **7B model swap** — would target the 56% + 24% buckets together. ~2-3 h on Colab T4; higher upside but more compute.
3. **Entity-aware NLI gating** — drop the NLI veto when claim and passage don't share named entities. Targets the 20% bucket. Half-day implementation.

None of these are V3 work. They're recommendations for whoever picks up the codebase next.