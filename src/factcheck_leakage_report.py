"""
How much of the open-web gain comes from retrieved fact-check articles?

Compares, on the same claims:
  bm25      closed Knowledge Store, BM25 top-5, plain prompt
  web       open web (DuckDuckGo top-5), plain prompt, as published
  web_rerun the same open-web run re-prompted in Nov 2026 (model-drift control)
  web_nofc  open web with fact-check URLs removed (filter_factcheck.py)

Scores are reported on all common claims and split by whether the original
top-5 contained a fact-check URL. Claims refused by the content filter in any
run are excluded from every run. Exact McNemar on accuracy.

    python src/factcheck_leakage_report.py --out results/factcheck_leakage_control.json
"""

import argparse
import json
import os
import sys

from scipy.stats import binomtest
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_verdict import normalize_llm_output  # noqa: E402
from filter_factcheck import is_factcheck_url  # noqa: E402

R = "results/"
DEFAULTS = {
    "retrieval": R + "ddg_direct_dev500_retrieval_cache.json",
    "bm25": R + "bm25_official_dev_plain_prompt_cache.json",
    "web": R + "ddg_direct_dev500_plain_eval_cache.json",
    "web_rerun": R + "ddg_direct_dev500_plain_rerun_eval_cache.json",
    "web_nofc": R + "ddg_direct_dev500_nofactcheck_plain_eval_cache.json",
}
REFUSED = {"CONTENT_FILTERED", "ERROR"}
COMPARISONS = [("bm25", "web_nofc"), ("web_nofc", "web_rerun"),
               ("bm25", "web_rerun"), ("web", "web_rerun")]


def _correct(run, k):
    return normalize_llm_output(run[k]["raw_output"]) == run[k]["true_label"]


def score(run, ks):
    y = [run[k]["true_label"] for k in ks]
    yh = [normalize_llm_output(run[k]["raw_output"]) for k in ks]
    return {
        "macro_f1": round(f1_score(y, yh, labels=[0, 1, 2, 3], average="macro",
                                   zero_division=0), 4),
        "accuracy": round(sum(a == b for a, b in zip(y, yh)) / len(ks), 4),
        "nee_predicted": sum(p == 2 for p in yh),
    }


def mcnemar(a, b, ks):
    only_a = sum(_correct(a, k) and not _correct(b, k) for k in ks)
    only_b = sum(_correct(b, k) and not _correct(a, k) for k in ks)
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return {"only_a_right": only_a, "only_b_right": only_b, "p_value": p}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name, path in DEFAULTS.items():
        ap.add_argument(f"--{name.replace('_', '-')}", default=path)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    load = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731
    ret = load(args.retrieval)
    runs = {n: load(getattr(args, n)) for n in ("bm25", "web", "web_rerun", "web_nofc")}

    common = [k for k in ret
              if all(k in r and r[k]["raw_output"] not in REFUSED for r in runs.values())]
    hit = {k for k in ret if any(is_factcheck_url(p[1]) for p in ret[k]["top_passages"])}
    groups = {
        "all": common,
        "factcheck_hit": [k for k in common if k in hit],
        "no_hit": [k for k in common if k not in hit],
    }

    out = {"n": len(common), "excluded_refused": len(ret) - len(common), "groups": {}}
    for g, ks in groups.items():
        out["groups"][g] = {
            "n": len(ks),
            "scores": {n: score(r, ks) for n, r in runs.items()},
            "mcnemar": {f"{a}->{b}": mcnemar(runs[a], runs[b], ks) for a, b in COMPARISONS},
        }
        print(f"\n== {g} (n={len(ks)})")
        for n, s in out["groups"][g]["scores"].items():
            print(f"  {n:9s} Macro-F1 {s['macro_f1']:.4f}  acc {s['accuracy']:.4f}  NEE {s['nee_predicted']}")
        for c, m in out["groups"][g]["mcnemar"].items():
            print(f"  {c}: {m['only_a_right']} vs {m['only_b_right']}, p = {m['p_value']:.3g}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
