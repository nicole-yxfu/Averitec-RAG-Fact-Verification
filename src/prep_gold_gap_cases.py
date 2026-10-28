"""
prep_gold_gap_cases.py — Week 6 error-analysis prep (planned §2.3 qualitative half).

Samples 20 dev500 claims (seed=42) where the Gold-evidence + plain-prompt run
predicted WRONG, and dumps a per-case brief suitable for manual 4-way
classification into:

  (a) Reasoning limitation — evidence is clear, LLM failed to reason correctly.
  (b) Annotation ambiguity — the gold label itself is a judgment call.
  (c) Prompt limitation   — the plain zero-shot prompt is too terse for this
                             claim type (e.g. Conflicting needs a per-side
                             comparison protocol the prompt does not scaffold).
  (d) Format limitation   — the Q+A pair format buries or scatters the signal
                             the annotator used from the raw source page.

We do NOT stratify by class — the plan asks for a random draw from the ~218
gold-wrong claims so the distribution reflects the actual error mass. Seed
locked at 42 (project convention). Output is a self-contained markdown file.

Usage:
    python src/prep_gold_gap_cases.py                       # -> results/gold_gap_cases_20.md
    python src/prep_gold_gap_cases.py --n 20 --seed 42      # explicit args
"""
import argparse
import json
import os
import random

LLM_LABEL_MAP = {"supported": 0, "refuted": 1,
                 "not enough evidence": 2,
                 "conflicting evidence/cherrypicking": 3}
NAMES = ["Supported", "Refuted", "NEE", "Conflicting"]


def norm_pred(raw: str) -> int:
    t = raw.strip().lower().strip('."\'* \n')
    if t in LLM_LABEL_MAP:
        return LLM_LABEL_MAP[t]
    for k, v in (("conflicting", 3), ("not enough", 2),
                 ("refuted", 1), ("supported", 0)):
        if k in t:
            return v
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-cache",
                    default="results/gold_baseline_official_dev_with_metadata_cache.json")
    ap.add_argument("--dev", default="data/dev.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/gold_gap_cases_20.md")
    args = ap.parse_args()

    cache = json.load(open(args.gold_cache, encoding="utf-8"))
    dev = json.load(open(args.dev, encoding="utf-8"))

    wrong = []
    for cid, entry in cache.items():
        pred = norm_pred(entry["raw_output"])
        if pred == -1 or pred != entry["true_label"]:
            wrong.append((cid, entry["claim_idx"], entry["true_label"],
                          pred, entry["raw_output"]))
    print(f"Total gold-wrong: {len(wrong)} / {len(cache)}")

    rng = random.Random(args.seed)
    picked = rng.sample(wrong, args.n)
    picked.sort(key=lambda x: x[1])  # by claim_idx for readable order

    lines = []
    lines.append(f"# Gold-Wrong Case Dump — {args.n} claims (seed={args.seed})")
    lines.append("")
    lines.append(f"Source: `{args.gold_cache}` filtered to `pred != true_label` (or unparseable). "
                 f"Draw uniform at random from the {len(wrong)} gold-wrong claims.")
    lines.append("")
    lines.append("For each case, classify into ONE of:")
    lines.append("- **(a) Reasoning** — evidence clearly supports one label; LLM chose another.")
    lines.append("- **(b) Annotation** — the true label is a judgment call; either label is defensible.")
    lines.append("- **(c) Prompt** — the plain prompt is too terse for this claim type "
                 "(e.g. Conflicting needs a per-side reasoning protocol).")
    lines.append("- **(d) Format** — the QA-pair format scatters or hides the annotator's key signal "
                 "(a raw source page would have made the answer obvious).")
    lines.append("")
    lines.append("---")
    lines.append("")

    for k, (cid, idx, tlabel, pred, raw) in enumerate(picked, 1):
        item = dev[idx]
        true_name = NAMES[tlabel]
        pred_name = NAMES[pred] if pred >= 0 else f"UNPARSED ({raw!r})"
        lines.append(f"## Case {k} — {cid} (dev.json idx {idx})")
        lines.append("")
        lines.append(f"- **true**: {true_name}   →   **gold-pred**: {pred_name}")
        lines.append(f"- **claim**: {item.get('claim', '').strip()}")
        spk = (item.get("speaker") or "").strip()
        src = (item.get("reporting_source") or "").strip()
        if spk or src:
            lines.append(f"- **speaker / source**: {spk!r} / {src!r}")
        just = (item.get("justification") or "").strip()
        if just:
            lines.append(f"- **annotator justification**: {just}")
        lines.append("")
        lines.append("**Gold QA pairs:**")
        for i, q in enumerate(item.get("questions", []), 1):
            qt = (q.get("question") or "").strip()
            answers = [a.get("answer", "").strip() for a in q.get("answers", [])
                       if a.get("answer")]
            if not (qt and answers):
                continue
            lines.append(f"- Q{i}: {qt}")
            for a in answers:
                a_short = a if len(a) <= 500 else (a[:500] + "…")
                lines.append(f"  - A: {a_short}")
        lines.append("")
        lines.append("**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:")
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.out}  ({len(picked)} cases)")


if __name__ == "__main__":
    main()
