# Phase 09 — Temporal Reasoning (Gated by Phase 13)

**Goal.** Decide whether to implement explicit temporal reasoning. The decision is **data-driven** — gated by what Phase 13 error analysis says. This phase exists in the plan to make the decision visible; the actual work is conditional.

**Effort.** 1 day (decide + scope-out doc) OR 1–3 days (implement) depending on the gate.
**Compute.** CPU.
**Depends on.** Phase 07 (working pipeline) and Phase 13 (failure mode breakdown).

## Why this exists

V1 punted on temporal and listed it as a known weakness. V2 had a `temporal_reasoner.py` that *forced verdict=SUPPORTED on any date match* — actively harmful. V3 takes neither shortcut. We commit to one of two paths, decided by data.

## The gate

After Phase 13 produces `artifacts/failures_tagged.json`:

- If `temporal_error` is **≥ 10%** of the 50 categorized failures: implement Phase 09 ("Path A").
- If `temporal_error` is **< 10%**: scope out, document, and move on ("Path B").

Why 10%? Below that, temporal-specific machinery yields <0.5 points of accuracy if perfect — within noise. Engineering effort is better spent elsewhere. This is a binding rule; revisit only if the headline accuracy is out of reach by other means.

---

## Path A — Implement (1–3 days)

### Inputs

- Failures tagged `temporal_error` from Phase 13.
- `dateparser` library (already in `requirements.txt` if Path A is taken).

### Deliverables

- `src/temporal/extract.py` — extracts `(entity, attribute, value, time_window)` quadruples from claim and passages using regex + spaCy + dateparser.
- `src/temporal/compare.py` — pairwise comparison: do claim's quadruples agree with passage's quadruples on shared `(entity, attribute)`? Returns per-pair contradiction signals.
- `src/temporal/integrate.py` — `temporal_signal(claim, passages) -> dict` returning features ready for the calibrator (e.g., `temporal_contradiction_count`, `claim_has_year`, `passage_year_disagreement`).
- Wire into `aggregate.py` from Phase 07: temporal contradictions add a "REFUTED-vote" signal; temporal NEI signals (passages don't cover claim's time window) add an "NEI-vote" signal.
- Re-run Phase 11 eval and Phase 13 categorization to confirm the bucket shrinks.

### Technical approach

- **Extraction.** Years (`r"\b(?:19|20)\d{2}\b"`), full dates (dateparser), relative phrases ("last year", "the 2008 film"), film/release/award qualifiers ("the 2012 sequel", "the original").
- **Entity binding.** Use spaCy NER to attach extracted dates to nearest entity span. Crude but adequate for HoVer-style claims.
- **Comparison logic.** Two quadruples agree if entities match, attributes match, and their time windows overlap. Disagree iff entities + attributes match but time windows are disjoint.
- **Integration.** Emit two new features for Phase 08's calibrator: `temporal_contradiction_count`, `temporal_uncovered_count`. Do *not* short-circuit the verifier — let the calibrator weigh the signal.

### Path-A exit criteria

- [ ] `temporal_error` bucket in re-run Phase 13 drops to <5% of failures.
- [ ] Headline HoVer dev accuracy improves by ≥1 point (CI not overlapping baseline).
- [ ] No regression on non-temporal claims (subset accuracy unchanged within CI).

---

## Path B — Scope out (0.5 day)

### Deliverables

- `src/temporal/__init__.py` exists but is essentially empty (no-op identity passthrough).
- `docs/SCOPED_OUT.md` — single doc collecting all V3 scope-out decisions, with rationale and data link.
- This phase doc updated with "Outcome" stating the bucket size and the decision.

### Path-B exit criteria

- [ ] `docs/SCOPED_OUT.md` exists and is referenced from `docs/FINAL_REPORT.md` (Phase 15).
- [ ] No silent fallback that pretends temporal handling is enabled — the config has no `temporal:` section.

---

## Risks and gotchas

- Temporal extraction is a research problem. Path A's 1–3 day budget covers a competent rule-based extractor, not a learned one. If accuracy gain is below the exit-criteria threshold after 3 days, fall back to Path B with the decision documented.
- HoVer's claim distribution doesn't have many "purely temporal" claims; most temporal failures are mixed (e.g., "the 2012 sequel" entity-binding). The fix may be entity binding, not temporal logic. If error analysis suggests this, reframe as Phase 13.5 entity work, not temporal.

## What NOT to do

- Do not implement a temporal module that, when uncertain, defaults to a verdict (V2's bug). When uncertain, emit a neutral signal — the calibrator decides.
- Do not skip the gate. The gate is the whole point. If you find yourself wanting to "just implement it because it sounds important", stop and re-read the gate.

## Outcome (Phase 09 closed 2026-05-15 — Path B, scoped out)

**Decision: Path B. Temporal reasoning is scoped out of V3.**

### Why

Phase 13's error analysis (50 stratified failures from the production whole-claim configuration) found **temporal_error = 0 / 50 = 0%** of failures. The phase doc's binding rule:
> If `temporal_error` is **< 10%**: scope out, document (Path B).

We're at 0%. Path B is mandated.

### What this means for the codebase

- No `src/temporal/` module ships in V3. The Phase 02 placeholder is empty by design.
- `configs/default.yaml` has no `temporal:` block (intentional — the YAML comment in Phase 02 spelled this out: an empty placeholder would replicate V2's silent-fallback bug).
- A new `docs/SCOPED_OUT.md` collects this decision and is referenced from `docs/FINAL_REPORT.md` (Phase 15).

### Caveat (documented honestly)

Many failing claims *do* contain temporal expressions: years ("1982 Bavarian Championships"), film year qualifiers ("the 2012 sequel"), and "X-before-Y" comparisons. The point is that those failures aren't *because of* temporal handling — they're NEI-bias and entity-binding failures on claims that happen to be temporal. A working temporal extractor wouldn't fix them.

This aligns with Phase 13's broader read: the verifier is the bottleneck, not retrieval or entity resolution or temporal handling.

### If a future iteration wants to revisit

Re-run Phase 13's categorization (`python -m src.analysis.categorize`) on the production output of that iteration. If the resulting `temporal_error` bucket is ≥ 10%, re-trigger Path A with the implementation plan from this doc. The gate doesn't have to fire only once.
