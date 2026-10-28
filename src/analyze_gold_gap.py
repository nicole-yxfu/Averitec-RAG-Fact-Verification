"""
Oracle Gap Decomposition — why does Gold Evidence only reach Macro-F1 0.5633?

Reads the cached gold-evidence predictions (no API calls) and answers, in order:

  1. Where does the 0.44 gap to a perfect score come from, per class?
     -> Conflicting ~55%, NEE ~29%, Supported+Refuted ~16%.
  2. Which confusion cells drive the two failing classes?
     -> NEE is over-predicted (84 predicted vs 34 true): 39 Refuted and
        10 Supported claims leak into NEE, destroying NEE precision.
     -> Conflicting is almost never predicted (6/497): the LLM "picks a
        side" instead (21 -> Refuted, 10 -> Supported).
  3. Two candidate mechanisms, both REJECTED quantitatively:
     -> Evidence truncation at MAX_EVIDENCE_CHARS: only 7/497 claims exceed
        the cap; cannot explain 108 errors.
     -> "No answer could be found" gold answers pushing the LLM to NEE:
        Refuted->NEE errors contain only ~4% unanswerable answers (vs 23%
        for true-NEE claims), so the LLM responds to that signal correctly
        and the errors come from elsewhere.
  4. Counterfactual Macro-F1 if one class were predicted perfectly
     (illustrative upper bounds for the Week 6 report).

Qualitative conclusion from reading the error cells (--show-examples):
  - Refuted->NEE: AVeriTeC annotators refute via pragmatic inference
    (satire source => event never happened; no official record => false),
    while the LLM applies a literal standard ("evidence does not state it
    is false") and outputs NEE. Label-convention mismatch, not a
    comprehension failure.
  - Conflicting->Refuted/Supported: Conflicting/Cherrypicking is a
    pragmatic category ("literally grounded but misleading overall"); the
    zero-shot LLM resolves the conflict to one side instead. Consistent
    with the AVeriTeC paper (p9): conflicting precision ~10% even with
    gold evidence.

Usage (from project root):
    python src/analyze_gold_gap.py
    python src/analyze_gold_gap.py --show-examples 5
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

LLM_LABEL_MAP = {
    "supported":                          0,
    "refuted":                            1,
    "not enough evidence":                2,
    "conflicting evidence/cherrypicking": 3,
}
NAMES = ["Supported", "Refuted", "NEE", "Conflicting"]

MAX_EVIDENCE_CHARS = 3000  # must match gold_baseline.py

# The confusion cells that matter for the narrative, in reporting order.
KEY_CELLS = [
    ((1, 2), "Refuted -> NEE"),
    ((0, 2), "Supported -> NEE"),
    ((3, 1), "Conflicting -> Refuted"),
    ((3, 0), "Conflicting -> Supported"),
    ((3, 2), "Conflicting -> NEE"),
]


def gold_evidence_len(item) -> int:
    """Character length of the gold QA evidence as gold_baseline.py builds it."""
    total = 0
    for q in item.get("questions", []):
        answers = [a.get("answer", "") for a in q.get("answers", []) if a.get("answer")]
        if q.get("question", "") and answers:
            total += len(q["question"]) + sum(len(a) for a in answers)
    return total


def unanswerable_fraction(item) -> float:
    """Fraction of gold answers that are 'No answer could be found' style."""
    tot = una = 0
    for q in item.get("questions", []):
        for a in q.get("answers", []):
            tot += 1
            txt = (a.get("answer") or "").lower()
            if a.get("answer_type") == "Unanswerable" or txt.startswith("no answer could be found"):
                una += 1
    return una / tot if tot else 0.0


def load_rows(cache, dev):
    """One dict per claim; unparseable outputs (Azure CONTENT_FILTERED) get pred=-1."""
    rows = []
    for cid, entry in cache.items():
        pred = LLM_LABEL_MAP.get(entry["raw_output"].strip().lower(), -1)
        item = dev[entry["claim_idx"]]
        rows.append({
            "cid": cid,
            "idx": entry["claim_idx"],
            "true": entry["true_label"],
            "pred": pred,
            "n_qa": entry["n_qa_pairs"],
            "ev_len": gold_evidence_len(item),
            "frac_una": unanswerable_fraction(item),
        })
    return rows


def print_examples(dev, indices, n):
    for idx in indices[:n]:
        item = dev[idx]
        print(f"  --- claim_{idx} | true={item['label']}")
        print(f"  CLAIM: {item['claim'][:200]}")
        for q in item.get("questions", [])[:5]:
            answers = [a.get("answer", "")[:200] for a in q.get("answers", []) if a.get("answer")]
            print(f"    Q: {q.get('question', '')[:160]}")
            for a in answers[:2]:
                print(f"     A: {a}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--cache-file",
                        default="results/gold_baseline_official_dev_with_metadata_cache.json")
    parser.add_argument("--data-file", default="data/dev.json")
    parser.add_argument("--show-examples", type=int, default=0, metavar="N",
                        help="print N gold-evidence examples per key error cell")
    args = parser.parse_args()

    with open(args.cache_file, "r", encoding="utf-8") as f:
        cache = json.load(f)
    with open(args.data_file, "r", encoding="utf-8") as f:
        dev = json.load(f)

    rows = load_rows(cache, dev)
    y_true = [r["true"] for r in rows]
    y_pred = [r["pred"] for r in rows]
    n_filtered = sum(1 for r in rows if r["pred"] == -1)

    # ── 1. Headline + per-class gap decomposition ────────────────────────────
    macro = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3])
    per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
    gap = 1.0 - macro
    print(f"n = {len(rows)} claims ({n_filtered} unparseable/CONTENT_FILTERED, scored as wrong)")
    print(f"Macro-F1 = {macro:.4f}   gap to 1.0 = {gap:.4f}\n")

    print(f"{'class':>12} {'F1':>7} {'gap-share':>10}   counterfactual Macro if class perfect")
    for c, (name, f) in enumerate(zip(NAMES, per)):
        cf = (per.sum() - f + 1.0) / 4
        print(f"{name:>12} {f:7.3f} {100 * (1 - f) / 4 / gap:9.1f}%   {cf:.4f} (Δ +{cf - macro:.4f})")
    print()

    # ── 2. Full report + confusion matrix ────────────────────────────────────
    print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3],
                                target_names=NAMES, digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print("Confusion matrix (rows=true, cols=pred):")
    print("            " + "".join(f"{n:>12}" for n in NAMES))
    for i, row in enumerate(cm):
        print(f"{NAMES[i]:>12}" + "".join(f"{c:>12}" for c in row))
    print()

    # ── 3. Mechanism checks on the key error cells ───────────────────────────
    print("Key error cells (evidence stats + rejected mechanisms):")
    print(f"{'cell':>26} {'n':>4} {'nQA':>5} {'evLen':>6} {'>cap':>5} {'fracUnanswerable':>17}")
    for (t, p), label in KEY_CELLS:
        sel = [r for r in rows if r["true"] == t and r["pred"] == p]
        if not sel:
            continue
        print(f"{label:>26} {len(sel):>4} "
              f"{np.mean([r['n_qa'] for r in sel]):>5.2f} "
              f"{np.mean([r['ev_len'] for r in sel]):>6.0f} "
              f"{sum(r['ev_len'] > MAX_EVIDENCE_CHARS for r in sel):>5} "
              f"{np.mean([r['frac_una'] for r in sel]):>17.3f}")
    for t, label in [(1, "true Refuted (baseline)"), (2, "true NEE (baseline)")]:
        sel = [r for r in rows if r["true"] == t]
        print(f"{label:>26} {len(sel):>4} "
              f"{np.mean([r['n_qa'] for r in sel]):>5.2f} "
              f"{np.mean([r['ev_len'] for r in sel]):>6.0f} "
              f"{sum(r['ev_len'] > MAX_EVIDENCE_CHARS for r in sel):>5} "
              f"{np.mean([r['frac_una'] for r in sel]):>17.3f}")
    n_trunc = sum(r["ev_len"] > MAX_EVIDENCE_CHARS for r in rows)
    print(f"\nTruncation check: only {n_trunc}/{len(rows)} claims exceed the "
          f"{MAX_EVIDENCE_CHARS}-char cap -> cannot explain the error mass.")

    # ── 4. Optional qualitative dump ─────────────────────────────────────────
    if args.show_examples:
        for (t, p), label in KEY_CELLS:
            sel = [r["idx"] for r in rows if r["true"] == t and r["pred"] == p]
            print(f"\n{'=' * 20} {label} ({len(sel)} total) {'=' * 20}")
            print_examples(dev, sel, args.show_examples)


if __name__ == "__main__":
    main()
