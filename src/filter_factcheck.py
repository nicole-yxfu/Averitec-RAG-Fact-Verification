"""
Remove fact-check sites from a cached open-web retrieval run.

Open-web search has no date filter, so for many claims it returns the
published fact-check of that very claim (PolitiFact, Snopes, AFP Fact Check,
...). That leaks the answer. This script drops every passage whose URL looks
like a fact-check page and writes a new retrieval cache, so the same LLM step
can be re-run on what is left (`reprompt_eval.py --retrieval-cache ...`).
No new search is made: the remaining passages keep their original order, and
claims simply end up with fewer than five.

Matching is by URL only, so it is approximate: a fact-check page on an
unlisted site is missed, and a news article written after the claim stays in.

    python src/filter_factcheck.py \
        --retrieval-cache results/ddg_direct_dev500_retrieval_cache.json \
        --out results/ddg_direct_dev500_nofactcheck_retrieval_cache.json
"""

import argparse
import json
import os
import re
from collections import Counter

FACTCHECK_URL = re.compile(
    r"politifact|snopes|factcheck|fact-check|fact_check|fullfact|factly"
    r"|boomlive|altnews|leadstories|truthorfiction|checkyourfact"
    r"|healthfeedback|africacheck|verafiles|newsmeter|vishvasnews"
    r"|newschecker|afp\.com.*fact|reuters\.com/fact|logically|teyit"
    r"|maldita|newtral|chequeado|aosfatos",
    re.IGNORECASE,
)


def is_factcheck_url(url: str) -> bool:
    return bool(FACTCHECK_URL.search(url or ""))


def domain(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url).split("/")[0]


def filter_cache(cache: dict) -> tuple[dict, dict]:
    """Return (filtered cache, stats). Input entries are not modified."""
    out, removed = {}, Counter()
    n_claims_hit = n_left_empty = n_before = n_after = 0
    for cid, entry in cache.items():
        passages = entry.get("top_passages", [])
        kept = [p for p in passages if not is_factcheck_url(p[1])]
        for p in passages:
            if is_factcheck_url(p[1]):
                removed[domain(p[1])] += 1
        n_before += len(passages)
        n_after += len(kept)
        if len(kept) < len(passages):
            n_claims_hit += 1
            if not kept:
                n_left_empty += 1
        new_entry = dict(entry)
        new_entry["top_passages"] = kept
        new_entry["n_passages"] = len(kept)
        new_entry["factcheck_removed"] = len(passages) - len(kept)
        out[cid] = new_entry
    stats = {
        "claims": len(cache),
        "claims_with_factcheck_hit": n_claims_hit,
        "claims_left_without_evidence": n_left_empty,
        "passages_before": n_before,
        "passages_after": n_after,
        "removed_by_domain": dict(removed.most_common()),
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--retrieval-cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.abspath(args.out) == os.path.abspath(args.retrieval_cache):
        raise SystemExit("refusing to overwrite the input cache")

    with open(args.retrieval_cache, encoding="utf-8") as f:
        cache = json.load(f)
    filtered, stats = filter_cache(cache)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "removed_by_domain"}, indent=1))
    print("top removed domains:", list(stats["removed_by_domain"].items())[:10])


if __name__ == "__main__":
    main()
