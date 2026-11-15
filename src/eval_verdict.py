"""
eval_verdict.py — score a run's predictions. The only code that owns the
verdict metric.

Reads `runs/<run_id>/predictions.jsonl` and writes the `verdict` section of
`runs/<run_id>/metrics.json`. It never touches the LLM, so a metric bug is
fixed by re-running this script over every past run, not by re-running the
experiment.

WHAT IT REPORTS, AND WHY EACH NUMBER IS THERE
---------------------------------------------
  Macro-F1       the headline, unweighted over 4 classes. On this label
                 distribution (Refuted 61%, Conflicting 8%, NEE 7% of dev)
                 it is dominated by the two rare classes: a change that
                 moves five Conflicting claims outweighs one that moves
                 thirty Refuted claims.
  Accuracy       the opposite bias, and the quantity McNemar can test.
                 Reporting only one of the two hides half of every result.
  Per-class F1    where the Macro-F1 actually came from.
  Confusion      which direction the errors run. The NEE row is the one
                 that mattered in this project: the system said "not
                 enough evidence" on 116 of 500 claims where 35 truly are.
  Bootstrap CI   a Macro-F1 of 0.3644 on 500 claims is not 0.3644 ± 0. The
                 percentile interval over claim resamples says how much of
                 a gap between two configurations is worth discussing at
                 all. It is NOT a substitute for a paired test — two runs'
                 marginal intervals can overlap heavily while the paired
                 difference is significant, because they share claims.
  Unparseable    a reply the label parser could not read is scored wrong,
                 which is correct but must be visible; a prompt change that
                 wrecks the output format would otherwise look like a
                 reasoning failure.
  Filtered       Azure content-filter refusals. A property of the
                 moderation layer, not of the system's reasoning, and not
                 stable across prompts — a longer system prompt trips the
                 filter on claims a shorter one passed. Counted separately
                 so a comparison can exclude the union.

USAGE
-----
    python src/eval_verdict.py --run runs/bm25_dev500_plain
    python src/eval_verdict.py --run runs/bm25_dev500_plain --no-write
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runs import GOLD_LABELS, load_run, write_metrics  # noqa: E402

# The canonical encoding. Three scripts in this repo once had three
# different orders; this is the one, and it matches GOLD_LABELS.
LABEL_TO_INT = {name: i for i, name in enumerate(GOLD_LABELS)}
SHORT_NAMES = ["Supported", "Refuted", "NEE", "Conflicting"]

# Older caches stored the gold label as an int already in this order.
FILTER_MARKERS = ("CONTENT_FILTERED", "ERROR")

UNPARSEABLE = -1


def normalize_llm_output(raw_output: str) -> int:
    """
    Maps a model reply to a label index, or UNPARSEABLE.

    Substring order matters and is not arbitrary: a verbose reply may
    mention several label words ("this is not enough evidence to call it
    refuted"), so the most specific label is tested first. 'conflicting'
    before 'not enough' before 'refuted' before 'supported' reproduces the
    behaviour every cache in this repo was scored with — changing the order
    would silently re-label historical runs.
    """
    text = (raw_output or "").strip().lower().strip('."\'* \n')
    if text in LABEL_TO_INT:
        return LABEL_TO_INT[text]
    direct = {
        "supported": 0,
        "refuted": 1,
        "not enough evidence": 2,
        "conflicting evidence/cherrypicking": 3,
    }
    if text in direct:
        return direct[text]
    if "conflicting" in text:
        return 3
    if "not enough" in text:
        return 2
    if "refuted" in text:
        return 1
    if "supported" in text:
        return 0
    return UNPARSEABLE


def gold_to_int(label) -> int:
    """Accepts the dataset string or a legacy int."""
    if isinstance(label, bool):
        return UNPARSEABLE
    if isinstance(label, int):
        return label if 0 <= label <= 3 else UNPARSEABLE
    return LABEL_TO_INT.get(str(label), UNPARSEABLE)


def is_filtered(raw_output: str) -> bool:
    return str(raw_output or "").strip() in FILTER_MARKERS


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=[0, 1, 2, 3],
                          average="macro", zero_division=0))


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """
    Percentile bootstrap over claims for Macro-F1.

    Resampling claims with replacement can produce a sample missing a rare
    class entirely; `zero_division=0` then scores that class 0, which
    widens the lower tail. That is a real property of a 38-claim class, not
    an artefact to be smoothed away, so such resamples are kept.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = macro_f1(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n_boot": n_boot, "seed": seed,
            "lo": float(lo), "hi": float(hi),
            "mean": float(stats.mean()), "sd": float(stats.std(ddof=1))}


def score_predictions(predictions: list[dict], n_boot: int = 1000,
                      seed: int = 0) -> dict:
    """Scores predictions.jsonl rows. Rows with no usable gold are dropped."""
    kept, dropped_gold = [], 0
    for row in predictions:
        g = gold_to_int(row.get("gold_label"))
        if g == UNPARSEABLE:
            dropped_gold += 1
            continue
        kept.append((row["claim_id"], g,
                     normalize_llm_output(row.get("raw_output", "")),
                     is_filtered(row.get("raw_output", ""))))

    if not kept:
        raise SystemExit("no predictions with a usable gold label")

    y_true = np.array([g for _, g, _, _ in kept])
    y_pred = np.array([p for _, _, p, _ in kept])
    per_class = f1_score(y_true, y_pred, labels=[0, 1, 2, 3],
                         average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    return {
        "n": len(kept),
        "n_dropped_no_gold": dropped_gold,
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": {SHORT_NAMES[i]: float(per_class[i]) for i in range(4)},
        "macro_f1_ci95": bootstrap_ci(y_true, y_pred, n_boot, seed),
        "gold_dist": {SHORT_NAMES[i]: int((y_true == i).sum()) for i in range(4)},
        "pred_dist": {SHORT_NAMES[i]: int((y_pred == i).sum()) for i in range(4)},
        "n_unparseable": int((y_pred == UNPARSEABLE).sum()),
        "n_content_filtered": sum(1 for _, _, _, f in kept if f),
        "filtered_claim_ids": sorted(c for c, _, _, f in kept if f),
        "confusion_matrix": {
            "rows": "gold", "cols": "pred", "order": SHORT_NAMES,
            "counts": cm.tolist(),
        },
    }


def print_report(label: str, m: dict) -> None:
    ci = m["macro_f1_ci95"]
    print(f"\n{label}   n = {m['n']}")
    print(f"  Macro-F1  {m['macro_f1']:.4f}   "
          f"95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}]")
    print(f"  Accuracy  {m['accuracy']:.4f}")
    print("\n  | Class | gold n | pred n | F1 |")
    print("  |---|---|---|---|")
    for name in SHORT_NAMES:
        print(f"  | {name} | {m['gold_dist'][name]} | {m['pred_dist'][name]} "
              f"| {m['per_class_f1'][name]:.3f} |")

    print("\n  Confusion (rows = gold, cols = pred):")
    header = "          " + "".join(f"{n[:5]:>8}" for n in SHORT_NAMES)
    print(header)
    for i, name in enumerate(SHORT_NAMES):
        row = "".join(f"{v:>8}" for v in m["confusion_matrix"]["counts"][i])
        print(f"  {name[:8]:<8}{row}")

    if m["n_unparseable"]:
        print(f"\n  [warning] {m['n_unparseable']} replies could not be "
              f"parsed into a label and are scored wrong.")
    if m["n_content_filtered"]:
        print(f"  [note] {m['n_content_filtered']} claims were refused by "
              f"the content filter (a moderation decision, not a reasoning "
              f"error). Exclude the union when comparing runs.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", required=True, help="Path to runs/<run_id>/")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true",
                    help="Print only; do not touch metrics.json.")
    args = ap.parse_args()

    run = load_run(args.run)
    if not run.predictions:
        raise SystemExit(f"{args.run}: no predictions.jsonl")

    m = score_predictions(run.predictions, args.n_boot, args.seed)
    print_report(os.path.basename(os.path.normpath(args.run)), m)

    if not args.no_write:
        path = write_metrics(args.run, "verdict", m)
        print(f"\nWrote {path}")
    print()


if __name__ == "__main__":
    main()
