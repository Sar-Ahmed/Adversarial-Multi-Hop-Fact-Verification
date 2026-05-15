# Phase 14 — Human eval protocol for evidence chains

## What you're rating

Each row in `artifacts/human_eval_sample.csv` corresponds to one rendered evidence chain in `artifacts/human_eval_rendered.txt`. Read the rendered chain (claim → sub-claim decomposition → per-sub-claim citations + reasoning → final verdict), then score the chain on five dimensions.

**Do not look at `gold_label` while rating.** It's in the CSV for stratification only — sort the CSV by something else (or hide that column) when scoring.

## Five dimensions (each 1–5)

| Dimension | What you're judging |
|---|---|
| **decomposition** | Are the sub-claims atomic? Do they collectively cover the original claim? Are `depends_on` edges sensible? |
| **citations** | Do the cited passages actually relate to the sub-claim being verified? |
| **reasoning** | Is the verification logic sound given the cited passages? |
| **faithfulness** | Does the reasoning text reflect what the passages actually say (no hallucinations)? |
| **overall** | Could a non-expert auditor follow the chain end-to-end and reach the same verdict? |

## Score anchors

- **5 — excellent.** No issues. A non-expert would understand and agree with the chain.
- **4 — minor flaw.** One small issue that doesn't change the conclusion.
- **3 — mixed.** Some parts work, some don't. A reviewer would want clarification.
- **2 — significantly broken.** Logic, citations, or decomposition fails on a major point.
- **1 — wrong.** Chain is unfollowable, citations are unrelated, or reasoning is hallucinated.

The `overall` score is *not* a strict average of the others — it's the auditability gestalt. A chain with great decomposition but hallucinated reasoning could be overall=2 even if decomposition=5.

## Sanity check (calibration)

Before rating the sample, score these three known-anchor cases to calibrate yourself:

- **Anchor 5 (excellent):** A chain whose sub-claims atomically split the original, whose citations are on-topic, whose reasoning is grounded in the cited passages, and whose final verdict matches what a careful reader would conclude.
- **Anchor 3 (mixed):** A chain with sensible decomposition + relevant citations but where one sub-claim's reasoning ignores the cited passage and asserts something the passage doesn't say.
- **Anchor 1 (wrong):** A chain that cites unrelated passages, whose reasoning text contradicts the passages, or that combines multiple distinct entities into a confused claim.

## How to fill

The CSV starts empty for the five rating columns. Fill each row's `decomposition`, `citations`, `reasoning`, `faithfulness`, `overall` with an integer 1–5. Add a short note in `notes` if the chain has an unusual failure mode worth flagging.

## After rating

Run `python -m src.evidence.aggregate_human_eval` to compute means + bootstrap CIs per dimension + correlations with correctness, written to `artifacts/human_eval_summary.json`.

## Phase 14 spec target

The phase doc asks for n=100. We ship n=50 to keep the manual rating cost (~2 h) tractable. With 50 chains, dimension-level CIs are wider (~±0.4 on a 5-point scale, vs ~±0.3 at n=100), but the headline mean and correlation-with-correctness signals are still informative.
