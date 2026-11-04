"""
make_results_table.py — regenerate the evaluation comparison table from the
committed prediction caches (no API calls, no re-running experiments).

Reads every dev500 prediction cache in results/, scores it against the gold
labels, and prints a Markdown table of Macro-F1 + per-class F1. This is the
single source of truth for the numbers reported in the README.

Usage:
    python src/make_results_table.py
    python src/make_results_table.py --md > docs/results_table.md
"""
import argparse
import csv
import json
import os

from sklearn.metrics import f1_score

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

LABELS = [0, 1, 2, 3]
CLASS_NAMES = ["Supported", "Refuted", "NEE", "Conflicting"]
LABEL_MAP = {
    "supported": 0,
    "refuted": 1,
    "not enough evidence": 2,
    "conflicting evidence/cherrypicking": 3,
}

# (display name, filename, group) — ordered as the README ladder.
CONFIGS = [
    ("Majority baseline",              "majority_predictions.csv",                              "No evidence"),
    ("Random baseline",                "random_predictions.csv",                                "No evidence"),
    ("Averaged Perceptron",            "perceptron_predictions.csv",                            "No evidence"),
    ("LLM zero-shot (no retrieval)",   "llm_baseline_official_dev_with_metadata_cache.json",    "No evidence"),
    ("BM25 closed-web + plain",        "bm25_official_dev_plain_prompt_cache.json",             "Closed web"),
    ("Dense rerank + plain",           "dense_official_dev_pure_eval_cache.json",               "Closed web"),
    ("BM25 + 8-shot ICL",              "bm25_fewshot8_dev500_cache.json",                       "Closed web"),
    ("DDG open-web + standard prompt", "ddg_direct_dev500_standard_eval_cache.json",            "Open web"),
    ("DDG open-web + plain prompt",    "ddg_direct_dev500_plain_eval_cache.json",               "Open web"),
    ("Gold evidence (oracle)",         "gold_baseline_official_dev_with_metadata_cache.json",   "Oracle"),
]


def normalize(raw: str) -> int:
    """Map raw LLM output text to a 0-3 label, or -1 if unparseable."""
    t = str(raw).strip().lower().strip('."\'* \n')
    if t in LABEL_MAP:
        return LABEL_MAP[t]
    for key, val in (("conflicting", 3), ("not enough", 2), ("refuted", 1), ("supported", 0)):
        if key in t:
            return val
    return -1


def load_cache(path: str):
    """Return (y_true, y_pred) from a JSON prediction cache or a CSV."""
    y_true, y_pred = [], []
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                y_true.append(int(row["true_label"]))
                y_pred.append(int(row["predicted_label"]))
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.values():
            y_true.append(int(entry["true_label"]))
            y_pred.append(normalize(entry["raw_output"]))
    return y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="emit a Markdown table")
    args = ap.parse_args()

    rows = []
    for name, fname, group in CONFIGS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"# skipped (missing): {fname}")
            continue
        y_true, y_pred = load_cache(path)
        macro = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
        per = f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
        rows.append((group, name, macro, per, len(y_true)))

    header = f"| {'Evidence':<11} | {'Configuration':<30} | Macro-F1 | " + " | ".join(f"{c} F1" for c in CLASS_NAMES) + " | n |"
    sep = "|" + "|".join(["---"] * (3 + len(CLASS_NAMES) + 1)) + "|"
    print(header)
    print(sep)
    last_group = None
    for group, name, macro, per, n in rows:
        g = group if group != last_group else ""
        last_group = group
        per_str = " | ".join(f"{p:.3f}" for p in per)
        print(f"| {g:<11} | {name:<30} | **{macro:.4f}** | {per_str} | {n} |")


if __name__ == "__main__":
    main()
