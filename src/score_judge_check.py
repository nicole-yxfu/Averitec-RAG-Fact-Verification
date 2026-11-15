"""
score_judge_check.py — how far can the faithfulness judge be trusted?

Compares a person's blind labels (judge_check/label.html) with the LLM
judge's labels (judge_check/key.json) on the 50 sampled verdicts.

WHAT IT REPORTS
---------------
The sample is stratified: 25 verdicts the judge called faithful, 25 it
called unfaithful. So the natural numbers are per-direction:

  "When the judge says FAITHFUL, the person agrees  a/25 times"
  "When the judge says UNFAITHFUL, the person agrees b/25 times"

Each comes with a Wilson 95% interval — at n = 25 they are wide, and
saying so is the point of reporting them.

Raw agreement on the 50 is NOT the judge's real-world agreement, because
the sample over-represents "unfaithful" (50% here, 25% in reality). The
population estimate re-weights the two directions by the judge's actual
split on dev500 (365 faithful / 120 unfaithful).

The same re-weighting gives a human-corrected faithfulness RATE for the
system: what the README's 0.753 would be if the person, not the judge, had
labelled every verdict. This is the number that says whether the headline
faithfulness figure is biased, and in which direction.

Cohen's kappa is reported on the sample as well, with that caveat: kappa
depends on the label mix, and this mix is artificial.

"Can't tell" answers are excluded from agreement and counted separately.

USAGE
-----
    python src/score_judge_check.py --answers ~/Downloads/judge_check_answers.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cohen_kappa(pairs: list[tuple[int, int]]) -> float | None:
    """pairs of (judge, human), each 0/1."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pj = sum(a for a, _ in pairs) / n
    ph = sum(b for _, b in pairs) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    return None if pe == 1 else (po - pe) / (1 - pe)


def score(answers: dict, meta: dict) -> dict:
    key = meta["key"]
    pop = meta["population"]
    w_f = pop["faithful"] / (pop["faithful"] + pop["unfaithful"])

    missing = [c for c in key if not (answers.get(c) or {}).get("answer")]
    unsure = [c for c in key if (answers.get(c) or {}).get("answer") == "unsure"]
    pairs, rows = [], []
    for cid, k in key.items():
        a = (answers.get(cid) or {}).get("answer")
        if a not in ("faithful", "unfaithful"):
            continue
        h = 1 if a == "faithful" else 0
        pairs.append((k["judge_faithful"], h))
        rows.append({"id": cid, "judge": k["judge_faithful"], "human": h,
                     "verdict": k["verdict"],
                     "note": (answers.get(cid) or {}).get("note", ""),
                     "seconds": (answers.get(cid) or {}).get("seconds")})

    def direction(judge_label: int) -> dict:
        sub = [r for r in rows if r["judge"] == judge_label]
        agree = sum(1 for r in sub if r["human"] == judge_label)
        ci = wilson(agree, len(sub))
        return {"n": len(sub), "agree": agree,
                "rate": agree / len(sub) if sub else None,
                "ci95": list(ci) if ci else None}

    on_f, on_u = direction(1), direction(0)
    out = {
        "n_items": len(key), "n_scored": len(rows),
        "n_unsure": len(unsure), "n_missing": len(missing),
        "judge_says_faithful": on_f,
        "judge_says_unfaithful": on_u,
        "sample_agreement": (sum(1 for a, b in pairs if a == b) / len(pairs)
                             if pairs else None),
        "sample_kappa": cohen_kappa(pairs),
        "population_weight_faithful": round(w_f, 4),
    }
    if on_f["rate"] is not None and on_u["rate"] is not None:
        out["population_agreement"] = w_f * on_f["rate"] + (1 - w_f) * on_u["rate"]
        # P(human says faithful) in the population, via the judge's split:
        human_rate = w_f * on_f["rate"] + (1 - w_f) * (1 - on_u["rate"])
        out["judge_faithfulness_rate"] = w_f
        out["human_corrected_faithfulness_rate"] = human_rate
        lenient = sum(1 for r in rows if r["judge"] == 1 and r["human"] == 0)
        strict = sum(1 for r in rows if r["judge"] == 0 and r["human"] == 1)
        out["disagreements"] = {"judge_too_lenient": lenient, "judge_too_strict": strict}
    # Per system verdict: where the disagreement actually lives.
    by_verdict = {}
    for v in sorted({r["verdict"] for r in rows}):
        sub = [r for r in rows if r["verdict"] == v]
        by_verdict[v] = {
            "n": len(sub),
            "agree": sum(1 for r in sub if r["judge"] == r["human"]),
            "judge_says_faithful": sum(r["judge"] for r in sub),
            "human_says_faithful": sum(r["human"] for r in sub),
        }
    out["by_verdict"] = by_verdict
    secs = [r["seconds"] for r in rows if r["seconds"]]
    out["median_seconds_per_item"] = sorted(secs)[len(secs) // 2] if secs else None
    out["disagreement_cases"] = [r for r in rows if r["judge"] != r["human"]]
    out["missing"] = missing
    return out


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--answers", required=True)
    ap.add_argument("--key", default=os.path.join(REPO, "judge_check", "key.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "judge_check_report.json"))
    args = ap.parse_args()

    with open(os.path.expanduser(args.answers), encoding="utf-8") as f:
        answers = json.load(f)["answers"]
    with open(args.key, encoding="utf-8") as f:
        meta = json.load(f)
    r = score(answers, meta)

    print(f"Scored {r['n_scored']} of {r['n_items']} "
          f"({r['n_unsure']} can't-tell, {r['n_missing']} unanswered)\n")
    for name, d in (("FAITHFUL", r["judge_says_faithful"]),
                    ("UNFAITHFUL", r["judge_says_unfaithful"])):
        ci = d["ci95"]
        print(f"  Judge says {name:<10}: person agrees {d['agree']}/{d['n']} = "
              f"{pct(d['rate'])}" + (f"   95% CI [{ci[0]:.0%}, {ci[1]:.0%}]" if ci else ""))
    if "population_agreement" in r:
        print(f"\n  Estimated agreement on all dev500 verdicts: {pct(r['population_agreement'])}")
        print(f"  Faithfulness rate: judge {pct(r['judge_faithfulness_rate'])} vs "
              f"human-corrected {pct(r['human_corrected_faithfulness_rate'])}")
        d = r["disagreements"]
        print(f"  Disagreements: judge too lenient {d['judge_too_lenient']}, "
              f"too strict {d['judge_too_strict']}")
    k = r["sample_kappa"]
    print(f"  Cohen's kappa on the (balanced) sample: "
          f"{'n/a' if k is None else f'{k:.2f}'}")
    print("\n  By system verdict:   agree   judge says faithful   person says faithful")
    for v, d in r["by_verdict"].items():
        print(f"    {v:<22} {d['agree']:>2}/{d['n']:<3}   {d['judge_says_faithful']:>2}/{d['n']:<17}"
              f"   {d['human_says_faithful']:>2}/{d['n']}")
    if r["median_seconds_per_item"]:
        print(f"  Median time per item: {r['median_seconds_per_item']} s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {os.path.relpath(args.out, REPO)}")


if __name__ == "__main__":
    main()
