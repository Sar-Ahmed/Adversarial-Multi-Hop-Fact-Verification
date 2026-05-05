# V3 Carry-over Decisions

A flat list of explicit "keep / drop / redesign" rules informed by the V1 + V2 audit. These are binding for V3 design unless overridden by a later phase decision (which must be recorded in the relevant phase doc, not here).

## Keep from V1 (the load-bearing engineering)

| What | Why |
|---|---|
| Phased development with day-by-day roadmap | Produces auditable artifacts; lets us kill failing experiments early (V1 killed Phase 5 CoT and Phase 10 7B upgrade cleanly because the discipline was in place). |
| `llama-cpp-python` + Qwen2.5-3B-Instruct Q4_K_M GGUF | Local, CPU-fast, exposes logprobs for calibration. Q4 quant noise is acceptable on this task per V1's empirical results. |
| `cross-encoder/nli-deberta-v3-base` as the contradiction veto | The single intervention in V1 that moved REFUTED detection meaningfully (Phase 11 ensemble: +4–6% absolute over LLM-only). |
| `BAAI/bge-small-en-v1.5` + `BAAI/bge-reranker-base` + FAISS flat-IP | R@10 = 0.92 on HoVer with 0% adversarial drop. Hard to beat at this scale on CPU. |
| Hard-negative mining with the BM25-skip-top-10 trick | Filters out near-paraphrase positives that pollute the negative pool. |
| Stratified evidence-chain human eval (28 samples, balanced by verdict-correctness) | Surfaced the chain-quality / verdict-correctness correlation that pointed at the verifier as bottleneck. |
| Tagged failure analysis (50 cases, JSON + narrative) | Required by spec; V1's format works. Re-use it. |
| Honest negative-results posture | V1 kept `phase5_eval.json` and `phase5_eval_7b_colab.json` on disk even though both showed worse-than-baseline. V3 does the same. |
| Per-phase JSON eval artifacts | Reproducible from disk; can be regenerated only by re-running the phase script. |

## Keep from V2 (just the surfaces, with real implementations behind them)

| What | Why |
|---|---|
| Rich dataclass schema (`Passage`, `Entity`, `SubClaim`, `EvidenceChain`, `Label` enum, `ReasoningType` enum) | Cleaner abstraction than V1's lighter schema. **Must fix** the `id` vs `sub_claim_id` field-name bug that took V2 down. |
| Folder naming (`reasoning/`, `calibration/`, `adversarial/`, `preprocessing/`) | Better separation of concerns than V1's `reasoner/` + `analysis/` mix. |
| `fastcoref` as the coreference choice | If we end up needing coref (currently optional), this is the right pick. V1 didn't have one. |
| Logistic-regression NEI calibrator concept | Right idea, just never trained. V3 trains it on FEVER NEI in Phase 08. |
| `PipelineConfig` dataclass | V2's intent (single source of config) was right; V2 just didn't follow through. V3 enforces by code review. |
| Adversarial test layout (clean vs adversarial verdict comparison) | The skeleton is right. V3 replaces the binary did-the-verdict-flip metric with calibrated confidence delta. |

## Remove entirely

| From V1 | Why |
|---|---|
| `configs/default.yaml` that no module reads | If config exists, every module reads it. Not optional. |
| Print-based progress output | Replaced by `loguru` structured logging. |
| Range-pinned `requirements.txt` | All `==`, no ranges. |

| From V2 | Why |
|---|---|
| `PHASE_X_COMPLETE.md`, `COMPREHENSIVE_PROJECT_GUIDE.md`, `AUDIT_FIXES_MAPPING.md`, `COMPLETE_REBUILD_SUMMARY.md` | Documentation must describe what is, not market what isn't. Phase docs in this folder *are* the contract; we do not write a victory-lap document at the end of each phase. |
| `pipeline.py` with `self.retriever = None` placeholder | A pipeline that doesn't run is not a pipeline. Phase 02 smoke test gates this. |
| `temporal_reasoner.py` that forces verdict=SUPPORTED on any date match | Worse than not having a temporal module. Phase 09 either implements it properly or scopes it out. No middle path. |
| `general_verifier.py` returning hardcoded NLI scores when its model is missing | Silent fallbacks lie. If a component is missing, raise. |
| Tests that don't exercise the real call path | The Phase 02 smoke test must call `pipeline.verify(claim)` end to end. |
| Heavy `transformers.pipeline()` for the LLM | Replaced by llama-cpp (CPU-fast, logprobs). |

## Redesign entirely

| Topic | V1 / V2 status | V3 redesign |
|---|---|---|
| **REFUTED detection** | V1 worked around with NLI veto; the LLM itself is still ~74% SUPPORTED-biased. | Phase 07: keep the NLI veto AND retest per-sub-claim CoT *post-veto*. Hypothesis: V1's Phase 5 CoT failed because aggregation collapsed via NEI; running CoT through the NLI veto first should change that. |
| **NEI calibration** | V1: per-token logprob threshold (+0.5%, noise). V2: classifier never trained. | Phase 08: logistic regression on 12 features, trained on FEVER NEI dev split, evaluated with bootstrap CIs. |
| **Adversarial distractor mining** | V1: cos-only; "opposite meaning" not enforced. V2: nothing. | Phase 06: two-stage. Stage 1 cos≥0.85; Stage 2 NLI(claim, candidate) contradiction prob ≥ 0.8. Sanity-check 20 mined samples manually before accepting. |
| **Temporal handling** | V1: punted, acknowledged. V2: actively harmful stub. | Phase 09 is *gated* on Phase 13 error-rate threshold. Either implement (entity, attribute, value, time-window) extraction + comparison, or scope-out and document. No half-measures. |
| **Pipeline configuration** | V1 vestigial YAML; V2 unused dataclass. | Phase 02: frozen `PipelineConfig` loaded from YAML, every module takes it as a constructor arg. Type-checked via pydantic. |
| **Integration testing** | V1: zero tests. V2: tests that don't catch the obvious bug. | Phase 02: 5-example end-to-end smoke test. Runs in <5 min on CPU with stubbed verifier; gates every PR. Phase 07 swaps the stub for the real verifier and the smoke test still passes. |
| **Observability** | V1: print to stdout. V2: logger logs crashes. | Phase 00: `loguru` with structured JSON sinks. Per-claim trace dump optional via `--trace`. |
| **Confidence intervals** | V1 reports point estimates only on n=200 (~3% noise). | Phase 11: bootstrap 95% CIs on every reported number. |

## Process rules

These are not technical decisions but they govern how V3 is built:

1. **Phase exit criteria are binding.** A phase is not complete until every checkbox in its "Exit criteria" section is green and the relevant artifacts are saved.
2. **Negative results stay in the repo.** If a phase fails (e.g., fine-tune doesn't beat base), commit the JSON, write the postmortem in the phase doc's "Outcome" section, and move on.
3. **No silent fallbacks in production code.** Stubs, mocks, and "best-effort" returns are tolerated only inside `tests/` and notebooks. Production code raises.
4. **Every metric ships with its sample size and CI.** "Accuracy = 0.575" alone is not acceptable in V3 reports. "Accuracy = 0.575 ± 0.07 (n=200, 95% CI)" is.
5. **Decompose docs from code.** This folder describes intent. The repo (built starting Phase 00) describes implementation. We do not maintain narrative `PHASE_X_COMPLETE.md` files inside the codebase.
