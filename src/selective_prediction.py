"""
selective_prediction.py — how much work can this system take off a human?

Macro-F1 scores a system that must answer every claim. No deployed
fact-checking pipeline works that way: Full Fact, Meta's third-party
programme and every content-moderation stack use the model as *triage* —
it handles what it is confident about and routes the rest to a person. The
operative question is therefore not "how accurate is it?" but:

    at what coverage, and at what accuracy, can it run unattended?

This script answers that with a risk-coverage curve. Claims are ranked by a
confidence score; at coverage c the system answers the top c fraction and
abstains on the rest, and we report the accuracy over what it answered.
A flat curve means the confidence score is worthless (abstaining at random
would do as well). A curve that climbs as coverage drops means the system
knows what it does not know, which is what makes it deployable.

CONFIDENCE FEATURES
-------------------
All three are available at inference time on an unseen claim — none peeks
at the gold label:

  faithfulness  the RAGAS-style judge's verdict on this system's own
                answer (src/rag_eval.py): is the verdict grounded in the
                retrieved evidence? Costs one extra LLM call.
  agreement     whether a second prompt reaches the same verdict
                (self-consistency). Costs one extra LLM call.
  predicted     the predicted class itself. Not a trick: per-class
    class       reliability differs enormously here — the open-web system
                is right 85% of the time when it says `refuted` and 15%
                when it says `not enough evidence`. "Never auto-answer
                NEE" is a legitimate, shippable policy.

WHY CROSS-FITTING
-----------------
Turning those features into one score needs a fitted model, and fitting it
on the same 500 claims it is then scored on would leak: the "never trust
NEE" rule would be learned from the very claims used to report its benefit.
So the score is produced by K-fold CROSS-FITTING — each claim is scored by
a model fitted on the other folds only, so no claim contributes to its own
score. A rule-based score that needs no fitting at all is reported
alongside as a sanity check.

This is still dev data that the project has developed against, so the
honest framing is "measured out-of-fold on dev", not "held-out test".

USAGE
-----
    python src/selective_prediction.py \\
        --predictions   results/ddg_direct_dev500_plain_eval_cache.json \\
        --faithfulness  results/faithfulness_ddg_plain_dev500.json \\
        --second-prompt results/ddg_dev500_nee_guard_eval_cache.json \\
        --out results/selective_prediction_dev500.json
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from compare_runs import normalize_llm_output, to_gold_int

TARGET_NAMES = ["Supported", "Refuted", "NEE", "Conflicting"]
COVERAGE_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def load_rows(pred_path: str, faith_path: str | None,
              second_path: str | None) -> list[dict]:
    """Join the prediction cache with whatever confidence signals exist."""
    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    faith = {}
    if faith_path:
        with open(faith_path, "r", encoding="utf-8") as f:
            faith = json.load(f)
    second = {}
    if second_path:
        with open(second_path, "r", encoding="utf-8") as f:
            second = json.load(f)

    rows = []
    for cid, entry in preds.items():
        gold = to_gold_int(entry.get("true_label"))
        if gold < 0:
            continue
        pred = normalize_llm_output(entry.get("raw_output", ""))
        # faithful: 1 grounded, 0 ungrounded, -1/absent unknown (the judge
        # itself was content-filtered). Unknown is its own category, not
        # silently merged into "ungrounded".
        f_raw = faith.get(cid, {}).get("faithful")
        faithful = f_raw if f_raw in (0, 1) else None
        agree = None
        if cid in second:
            agree = int(pred == normalize_llm_output(
                second[cid].get("raw_output", "")))
        rows.append({"cid": cid, "gold": gold, "pred": pred,
                     "correct": int(pred == gold),
                     "faithful": faithful, "agree": agree})
    return rows


def featurize(rows: list[dict]) -> np.ndarray:
    """One-hot everything; no ordinal assumptions, no missing-value
    imputation that would invent a signal."""
    feats = []
    for r in rows:
        v = []
        v += [1.0 if r["faithful"] == 1 else 0.0,
              1.0 if r["faithful"] == 0 else 0.0]      # else: unknown
        v += [1.0 if r["agree"] == 1 else 0.0,
              1.0 if r["agree"] == 0 else 0.0]
        v += [1.0 if r["pred"] == i else 0.0 for i in range(4)]
        feats.append(v)
    return np.array(feats, dtype=float)


def crossfit_scores(X: np.ndarray, y: np.ndarray, n_splits: int,
                    seed: int) -> np.ndarray:
    """Out-of-fold P(correct): every claim is scored by a model that never
    saw it. Without this the reported benefit is partly self-fulfilling."""
    scores = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        scores[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return scores


def rule_based_scores(rows: list[dict]) -> np.ndarray:
    """A fitted-nothing baseline: +1 for a grounded verdict, +1 for the two
    prompts agreeing. Uses no label information whatsoever, so it cannot
    leak; it just cannot learn that NEE is untrustworthy either."""
    return np.array([(1.0 if r["faithful"] == 1 else 0.0)
                     + (1.0 if r["agree"] == 1 else 0.0) for r in rows])


def risk_coverage(scores: np.ndarray, correct: np.ndarray) -> dict:
    """
    Accuracy over the top-k most confident claims, for every k.

    Ties are broken by a fixed random permutation rather than by input
    order: claim ids are in dataset order, and the rule-based score takes
    only three distinct values, so breaking ties by position would let
    dataset ordering masquerade as confidence.
    """
    rng = np.random.default_rng(0)
    order = np.lexsort((rng.permutation(len(scores)), -scores))
    ordered = correct[order]
    cum_correct = np.cumsum(ordered)
    k = np.arange(1, len(ordered) + 1)
    acc_at_k = cum_correct / k
    # AURC over the error rate; lower is better. Random ordering gives
    # (1 - overall accuracy) exactly, which is the comparison that matters.
    aurc = float(np.mean(1.0 - acc_at_k))
    return {"acc_at_k": acc_at_k, "order": order, "aurc": aurc}


def summarize(rows: list[dict], rc: dict, name: str) -> dict:
    n = len(rows)
    acc_at_k = rc["acc_at_k"]
    order = rc["order"]
    overall = float(acc_at_k[-1])
    points = []
    for cov in COVERAGE_GRID:
        k = max(1, int(round(cov * n)))
        kept = [rows[i] for i in order[:k]]
        dist = {TARGET_NAMES[i]: sum(1 for r in kept if r["pred"] == i)
                for i in range(4)}
        points.append({
            "coverage": round(k / n, 4), "n_answered": k,
            "accuracy": round(float(acc_at_k[k - 1]), 4),
            "errors_made": int(round((1 - acc_at_k[k - 1]) * k)),
            "answered_dist": dist,
        })
    return {"name": name, "n": n, "overall_accuracy": round(overall, 4),
            "aurc": round(rc["aurc"], 4),
            "aurc_random": round(1 - overall, 4), "points": points}


def print_report(summ: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"{summ['name']}   n = {summ['n']}   "
          f"accuracy at full coverage = {summ['overall_accuracy']:.4f}")
    print(f"AURC {summ['aurc']:.4f}  vs  {summ['aurc_random']:.4f} for "
          f"abstaining at random (lower is better)")
    print("\n| Coverage | Answered | Accuracy | Errors made | "
          "Supported | Refuted | NEE | Conflicting |")
    print("|---|---|---|---|---|---|---|---|")
    for p in summ["points"]:
        d = p["answered_dist"]
        print(f"| {p['coverage']:.0%} | {p['n_answered']} | "
              f"{p['accuracy']:.4f} | {p['errors_made']} | "
              f"{d['Supported']} | {d['Refuted']} | {d['NEE']} | "
              f"{d['Conflicting']} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--predictions", required=True,
                    help="Eval cache holding the verdicts to gate.")
    ap.add_argument("--faithfulness", default=None,
                    help="rag_eval.py cache for the SAME run.")
    ap.add_argument("--second-prompt", default=None,
                    help="Eval cache from a second prompt on the same "
                         "retrieval, for a self-consistency signal.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.predictions, args.faithfulness, args.second_prompt)
    n_f = sum(1 for r in rows if r["faithful"] is not None)
    n_a = sum(1 for r in rows if r["agree"] is not None)
    print(f"Loaded {len(rows)} claims  "
          f"(faithfulness on {n_f}, second prompt on {n_a})")

    y = np.array([r["correct"] for r in rows])
    X = featurize(rows)

    summaries = []
    fitted = summarize(
        rows, risk_coverage(crossfit_scores(X, y, args.folds, args.seed), y),
        f"Cross-fitted confidence ({args.folds}-fold, out-of-fold)")
    summaries.append(fitted)
    print_report(fitted)

    rule = summarize(rows, risk_coverage(rule_based_scores(rows), y),
                     "Rule-based: grounded + prompts agree (nothing fitted)")
    summaries.append(rule)
    print_report(rule)

    print("\nRead the Accuracy column against the full-coverage row: that is "
          "what abstaining at random would give at EVERY coverage. The gap "
          "is what the confidence signal is worth.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"predictions": args.predictions,
                       "faithfulness": args.faithfulness,
                       "second_prompt": args.second_prompt,
                       "folds": args.folds, "seed": args.seed,
                       "summaries": summaries}, f, indent=2)
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
