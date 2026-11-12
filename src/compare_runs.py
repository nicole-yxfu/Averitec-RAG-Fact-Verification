"""
compare_runs.py — score eval caches side by side with a significance test.

Reads one or more LLM eval caches (the `{claim_id: {true_label, raw_output}}`
files reprompt_eval.py writes), reports Macro-F1 and per-class F1 for each,
and runs an exact McNemar test of every variant against the first one.

WHY MCNEMAR AND NOT A DIFFERENCE IN F1
--------------------------------------
Two systems scored on the SAME claims give paired, not independent,
observations. A raw Macro-F1 gap says nothing about whether it would survive
a different sample of claims. McNemar conditions on exactly the claims where
the two systems disagree and asks whether the disagreements are lopsided:

    b = claims system A got right and system B got wrong
    c = claims system A got wrong and system B got right

Under the null (the two systems are equally accurate) each discordant claim
is a fair coin, so b ~ Binomial(b + c, 0.5). The exact two-sided binomial
test is used rather than the chi-square approximation because b + c here is
typically a few dozen, where the approximation is unreliable.

The test is on ACCURACY (per-claim correctness), which is what pairing makes
available. Macro-F1 is not a per-claim quantity, so it cannot be tested this
way — the two numbers answer different questions and both are reported.

Note the asymmetry with Macro-F1 on this label distribution: Refuted is 61%
of dev, so a change that only moves Refuted claims can shift accuracy while
barely moving Macro-F1, and a change confined to the 38 Conflicting claims
can move Macro-F1 a lot while leaving accuracy almost unchanged. Read both.

USAGE
-----
    python src/compare_runs.py \\
        --caches results/ddg_direct_dev500_plain_eval_cache.json \\
                 results/hybrid_dev150_rrf_eval_cache.json \\
        --labels baseline rrf \\
        --restrict-to-common
"""
import argparse
import json
import os

from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

LLM_LABEL_MAP = {
    "supported": 0,
    "refuted": 1,
    "not enough evidence": 2,
    "conflicting evidence/cherrypicking": 3,
}
GOLD_LABEL_MAP = {
    "Supported": 0,
    "Refuted": 1,
    "Not Enough Evidence": 2,
    "Conflicting Evidence/Cherrypicking": 3,
}
TARGET_NAMES = ["Supported", "Refuted", "NEE", "Conflicting"]


def normalize_llm_output(raw_output: str) -> int:
    """Identical to reprompt_eval.normalize_llm_output; substring order
    matters because 'not enough evidence' contains neither of the others but
    a verbose reply may contain several label words."""
    text = (raw_output or "").strip().lower().strip('."\'* \n')
    if text in LLM_LABEL_MAP:
        return LLM_LABEL_MAP[text]
    if "conflicting" in text:
        return 3
    if "not enough" in text:
        return 2
    if "refuted" in text:
        return 1
    if "supported" in text:
        return 0
    return -1


def to_gold_int(label) -> int:
    """Caches store the gold label as either an int or the dataset string."""
    if isinstance(label, int):
        return label
    return GOLD_LABEL_MAP.get(str(label), -1)


def load_cache(path: str) -> dict[str, tuple[int, int]]:
    """{claim_id: (gold, pred)}, skipping entries with no usable gold label."""
    with open(path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    out = {}
    for cid, entry in cache.items():
        gold = to_gold_int(entry.get("true_label"))
        if gold < 0:
            continue
        out[cid] = (gold, normalize_llm_output(entry.get("raw_output", "")))
    return out


def filtered_ids(path: str) -> set[str]:
    """
    Claim ids the Azure content filter refused ('CONTENT_FILTERED') or that
    errored out.

    These matter for a fair comparison because the refusal is a property of
    the moderation layer, not of the system's reasoning, and it is NOT
    stable across prompts: a longer system prompt can trip the filter on
    claims a shorter one passed. Such a claim is scored wrong for both runs,
    so leaving it in charges one variant for a moderation decision rather
    than a verification error. --exclude-filtered drops the UNION of both
    runs' refusals so the remaining set is identical for every variant.
    """
    with open(path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    return {cid for cid, e in cache.items()
            if str(e.get("raw_output", "")).strip()
            in ("CONTENT_FILTERED", "ERROR")}


def score(rows: dict[str, tuple[int, int]]) -> dict:
    y_true = [g for g, _ in rows.values()]
    y_pred = [p for _, p in rows.values()]
    per_class = f1_score(y_true, y_pred, labels=[0, 1, 2, 3],
                         average=None, zero_division=0)
    return {
        "n": len(rows),
        "macro_f1": f1_score(y_true, y_pred, labels=[0, 1, 2, 3],
                             average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "per_class_f1": dict(zip(TARGET_NAMES, per_class)),
        "n_unparseable": sum(1 for p in y_pred if p == -1),
        "pred_dist": {TARGET_NAMES[i]: sum(1 for p in y_pred if p == i)
                      for i in range(4)},
    }


def mcnemar(a: dict[str, tuple[int, int]],
            b: dict[str, tuple[int, int]]) -> dict:
    """Exact two-sided McNemar over the claims both runs cover."""
    common = sorted(set(a) & set(b))
    b_only = c_only = both_right = both_wrong = 0
    for cid in common:
        a_ok = a[cid][1] == a[cid][0]
        b_ok = b[cid][1] == b[cid][0]
        if a_ok and not b_ok:
            b_only += 1
        elif b_ok and not a_ok:
            c_only += 1
        elif a_ok:
            both_right += 1
        else:
            both_wrong += 1
    n_disc = b_only + c_only
    p = (binomtest(b_only, n_disc, 0.5, alternative="two-sided").pvalue
         if n_disc else 1.0)
    return {"n_common": len(common), "both_right": both_right,
            "both_wrong": both_wrong, "only_a_right": b_only,
            "only_b_right": c_only, "n_discordant": n_disc, "p_value": p}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--caches", nargs="+", required=True,
                    help="Eval cache JSONs. The FIRST is the reference every "
                         "other one is tested against.")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Display names, in the same order as --caches.")
    ap.add_argument("--restrict-to-common", action="store_true",
                    help="Score every run on the intersection of claim ids. "
                         "Use this whenever the caches cover different claim "
                         "counts (e.g. a dev150 variant against the dev500 "
                         "baseline) — otherwise the Macro-F1 column compares "
                         "different test sets and is meaningless.")
    ap.add_argument("--exclude-filtered", action="store_true",
                    help="Drop claims that ANY run's content filter refused, "
                         "so a variant is not charged for a moderation "
                         "decision instead of a reasoning error. Report both "
                         "with and without.")
    ap.add_argument("--out", default=None, help="Write the report as JSON.")
    args = ap.parse_args()

    labels = args.labels or [
        os.path.basename(p).replace("_eval_cache.json", "")
        for p in args.caches]
    if len(labels) != len(args.caches):
        raise SystemExit("--labels must have the same length as --caches")

    runs = {lab: load_cache(path) for lab, path in zip(labels, args.caches)}
    for lab, path in zip(labels, args.caches):
        n_filt = len(filtered_ids(path))
        print(f"{lab:<14} {len(runs[lab]):>4} claims  "
              f"({n_filt} content-filtered)  {path}")

    dropped = set()
    if args.exclude_filtered:
        for path in args.caches:
            dropped |= filtered_ids(path)
        runs = {lab: {c: v for c, v in r.items() if c not in dropped}
                for lab, r in runs.items()}
        print(f"\n--exclude-filtered: dropped {len(dropped)} claims refused "
              f"by at least one run's content filter.")

    if args.restrict_to_common:
        common = set.intersection(*(set(r) for r in runs.values()))
        print(f"\nRestricting all runs to {len(common)} common claim ids.")
        runs = {lab: {c: r[c] for c in common} for lab, r in runs.items()}
    else:
        sizes = {len(r) for r in runs.values()}
        if len(sizes) > 1:
            print(f"\n  [warning] runs cover different claim counts {sorted(sizes)}; "
                  f"Macro-F1 is NOT comparable across them. "
                  f"Re-run with --restrict-to-common.")

    scores = {lab: score(r) for lab, r in runs.items()}

    print("\n| Run | n | Macro-F1 | Accuracy | Supported | Refuted | NEE | Conflicting |")
    print("|---|---|---|---|---|---|---|---|")
    for lab in labels:
        s = scores[lab]
        pc = s["per_class_f1"]
        print(f"| {lab} | {s['n']} | {s['macro_f1']:.4f} | {s['accuracy']:.4f} | "
              f"{pc['Supported']:.3f} | {pc['Refuted']:.3f} | "
              f"{pc['NEE']:.3f} | {pc['Conflicting']:.3f} |")

    ref = labels[0]
    tests = {}
    if len(labels) > 1:
        print(f"\nExact McNemar on accuracy, each run vs `{ref}`:\n")
        print("| Run | ΔMacro-F1 | ΔAcc | only ref right | only this right | p |")
        print("|---|---|---|---|---|---|")
        for lab in labels[1:]:
            m = mcnemar(runs[ref], runs[lab])
            tests[lab] = m
            d_f1 = scores[lab]["macro_f1"] - scores[ref]["macro_f1"]
            d_acc = scores[lab]["accuracy"] - scores[ref]["accuracy"]
            star = ("***" if m["p_value"] < 0.001 else
                    "**" if m["p_value"] < 0.01 else
                    "*" if m["p_value"] < 0.05 else "ns")
            print(f"| {lab} | {d_f1:+.4f} | {d_acc:+.4f} | {m['only_a_right']} | "
                  f"{m['only_b_right']} | {m['p_value']:.4g} {star} |")
        print("\n`ns` = not significant at 0.05. A small ΔMacro-F1 with a "
              "high p-value means the variant is indistinguishable from the "
              "reference on this sample, which is a result, not a failure.")

    print("\nPredicted-label distribution (watch the NEE column — Finding 2 "
          "is about unjustified NEE verdicts):\n")
    print("| Run | Supported | Refuted | NEE | Conflicting | unparseable |")
    print("|---|---|---|---|---|---|")
    for lab in labels:
        s = scores[lab]
        d = s["pred_dist"]
        print(f"| {lab} | {d['Supported']} | {d['Refuted']} | {d['NEE']} | "
              f"{d['Conflicting']} | {s['n_unparseable']} |")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"caches": dict(zip(labels, args.caches)),
                       "restricted_to_common": args.restrict_to_common,
                       "excluded_filtered": sorted(dropped),
                       "scores": scores, "mcnemar_vs_reference": tests},
                      f, indent=2)
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
