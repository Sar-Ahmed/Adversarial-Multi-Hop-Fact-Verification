"""Phase 06 sanity check: stratified-sample 20 (claim, distractor) pairs and
write a markdown file with empty rating columns for the engineer to fill in.

Stratifies by `low_confidence` so we see both the strict-threshold distractors
and any padded ones. If >25% of the 20 samples turn out to be NOT genuine
contradictions, the phase doc requires raising the NLI threshold and re-mining.

Run: python -m src.adversarial.sanity_check
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "artifacts" / "distractors_v3.json"
OUT_PATH = ROOT / "artifacts" / "distractor_sanity_check.md"

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _flatten(payload: dict) -> list[dict]:
    """Flatten the per-claim distractors into one row per (claim, distractor)."""
    rows: list[dict] = []
    for uid, rec in payload["results"].items():
        for d in rec["distractors"]:
            rows.append(
                {
                    "uid": uid,
                    "claim": rec["claim"],
                    "claim_label": rec["label"],
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "text": d["text"],
                    "cos": d["cos"],
                    "contra_prob": d["contra_prob"],
                    "entail_prob": d["entail_prob"],
                    "neutral_prob": d["neutral_prob"],
                    "low_confidence": d.get("low_confidence", False),
                }
            )
    return rows


@app.command()
def main(
    n: int = typer.Option(20, help="Sample size to inspect."),
    seed: int = typer.Option(42),
) -> None:
    """Sample distractors and emit a markdown table for manual rating."""
    if not IN_PATH.exists():
        raise FileNotFoundError(f"{IN_PATH} not found — run src.adversarial.mine first")

    payload = json.loads(IN_PATH.read_text(encoding="utf-8"))
    rows = _flatten(payload)

    rng = random.Random(seed)
    rng.shuffle(rows)

    # Stratify: include up to half low-confidence (padded) if available.
    lc = [r for r in rows if r["low_confidence"]]
    hc = [r for r in rows if not r["low_confidence"]]
    n_lc = min(n // 2, len(lc))
    n_hc = n - n_lc
    sample = lc[:n_lc] + hc[:n_hc]
    rng.shuffle(sample)

    lines = [
        "# Phase 06 — Adversarial distractor sanity check",
        "",
        f"**Sampled {len(sample)} (claim, distractor) pairs from `artifacts/distractors_v3.json`.**",
        "",
        f"- Stratified: {n_lc} low-confidence (padded) + {n_hc} high-confidence",
        f"- Seed: {seed}",
        f"- Mining config: cos≥{payload['summary']['cos_threshold']}, "
        f"contra≥{payload['summary']['contra_threshold']} "
        f"(relaxed {payload['summary']['relaxed_contra_threshold']} for padding)",
        f"- NLI model: `{payload['summary']['nli_model']}`",
        "",
        "## How to rate",
        "",
        "For each row, mark **Pass** or **Fail** under the *Verdict* column.",
        "",
        "- **Pass** — the distractor genuinely contradicts the claim, or implies its negation, or describes the same entities/attributes in an incompatible way. Mining did its job.",
        "- **Fail** — the distractor is *on-topic but neither contradictory nor incompatible* (i.e. it's a high-cos passage that just happens to share keywords; V1 would have shipped this too). Mining did not do its job.",
        "",
        "Phase 06 exit criterion: ≤25% Fail rate (i.e. ≥15 Pass out of 20).",
        "",
        "If the Fail rate exceeds 25%, raise `contra_threshold` (e.g. to 0.9) and re-run `python -m src.adversarial.mine`.",
        "",
        "## Sample",
        "",
    ]

    for i, r in enumerate(sample, start=1):
        flag = " ⚠ low-conf" if r["low_confidence"] else ""
        lines.extend(
            [
                f"### {i}. uid={r['uid'][:8]}  cos={r['cos']:.3f}  contra={r['contra_prob']:.3f}{flag}",
                "",
                f"**Claim ({r['claim_label']}):** {r['claim']}",
                "",
                f"**Distractor** ({r['title']} :: sent {r['doc_id'].rsplit('::', 1)[-1]}):",
                f"> {r['text']}",
                "",
                f"**NLI probs:** contra={r['contra_prob']:.3f}, entail={r['entail_prob']:.3f}, neutral={r['neutral_prob']:.3f}",
                "",
                "**Verdict:** ☐ Pass / ☐ Fail",
                "",
                "**Notes:**",
                "",
                "---",
                "",
            ]
        )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"  {len(sample)} samples ready for manual rating")


if __name__ == "__main__":
    app()
