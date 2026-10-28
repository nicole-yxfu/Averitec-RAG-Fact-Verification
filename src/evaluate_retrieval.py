"""
evaluate_retrieval.py — Computes Recall@5 for a bm25_baseline.py result
cache against AVeriTeC's official gold evidence URLs. This is the Week 2
deliverable: "Query generation module with Strategy A and Strategy B,
evaluated by Recall@5."

PURE POST-HOC ANALYSIS — this script does NOT re-run any retrieval. It
reads the `top_passages` URLs and `claim_idx` already stored in a
bm25_baseline.py result-cache file (produced with --query-strategy direct
or --query-strategy keyword) and compares them against the gold evidence
URLs in the original AVeriTeC source JSON
(claim['questions'][i]['answers'][j]['source_url'] / ['cached_source_url']).

Matching is done by POSITIONAL claim_idx, not claim text. dev.json /
train.json contain duplicate claim texts (confirmed earlier: 9 duplicates
among dev.json's 500 claims) — a text-based join would silently merge or
misattribute gold URLs between unrelated claims. claim_idx is the same
field bm25_baseline.py already validated and stored in every result-cache
entry, so this reuses that fix rather than reintroducing the bug.

Recall@k definition (standard IR recall-at-k):
  For a claim with G distinct gold URLs and the top-k retrieved passage
  URLs, Recall@k = |gold URLs found in top-k| / |G|. Claims with G=0 (no
  gold URLs in the annotation) are excluded from the average and reported
  separately, not silently dropped.

Hit@k (secondary metric): 1 if at least one gold URL appears in the top-k,
else 0. Reported alongside Recall@k since most claims have only 1-3 gold
URLs, where strict recall can look harsh even for a genuinely useful
retrieval.

--source-file note: pass the file claim_idx was resolved against, i.e.
the SAME source_idx_file bm25_baseline.py used for that run:
  - internal_dev (or a sample drawn from it) -> data/train.json
  - official_dev (or a sample drawn from it)  -> data/dev.json

Usage:
  # Single cache
  python src/evaluate_retrieval.py \\
      --result-cache results/bm25_baseline_internal_dev_sample150_with_metadata_top5_cache.json \\
      --source-file data/train.json --label "Strategy A (Direct)"

  # Side-by-side comparison (Strategy A vs Strategy B)
  python src/evaluate_retrieval.py --compare \\
      --strategy-a-cache results/bm25_baseline_internal_dev_sample150_with_metadata_top5_cache.json \\
      --strategy-b-cache results/bm25_baseline_internal_dev_sample150_with_metadata_top5_keyword_cache.json \\
      --source-file data/train.json
"""

import os
import json
import argparse


def load_gold_urls_by_idx(source_json_file: str) -> dict:
    """
    Builds claim_idx (positional index in source file) -> set of gold
    evidence URLs, from the official AVeriTeC question-answer annotations.

    Both 'source_url' and 'cached_source_url' (archive.org mirror) count
    as valid matches, since a retrieved Knowledge Store document may
    reference either the live or archived version of the same source.
    """
    with open(source_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gold_map = {}
    for idx, item in enumerate(data):
        urls = set()
        for q in item.get('questions', []):
            for a in q.get('answers', []):
                if a.get('source_url'):
                    urls.add(a['source_url'])
                if a.get('cached_source_url'):
                    urls.add(a['cached_source_url'])
        gold_map[idx] = urls

    return gold_map


def load_retrieved_urls(result_cache_file: str, top_k: int) -> dict:
    """
    Builds cid -> (claim_idx, [retrieved URLs in rank order, top-k]),
    reading directly from a bm25_baseline.py result-cache file. No
    retrieval is re-run — top_passages was already saved by bm25_baseline.py.
    """
    with open(result_cache_file, 'r', encoding='utf-8') as f:
        cache = json.load(f)

    retrieved = {}
    for cid, entry in cache.items():
        claim_idx = entry.get('claim_idx')
        top_passages = entry.get('top_passages', [])  # [[score, url, preview], ...]
        urls = [p[1] for p in top_passages[:top_k]]
        retrieved[cid] = (claim_idx, urls)

    return retrieved


def compute_recall_at_k(gold_urls_by_idx: dict, retrieved_by_cid: dict) -> dict:
    """Computes per-claim and averaged Recall@k / Hit@k."""
    recalls = []
    hits = []
    no_gold_count = 0
    no_idx_count = 0
    no_gold_claim_ids = []

    for cid, (claim_idx, retrieved_urls) in retrieved_by_cid.items():
        if claim_idx is None:
            no_idx_count += 1
            continue

        gold_urls = gold_urls_by_idx.get(claim_idx, set())
        if not gold_urls:
            no_gold_count += 1
            no_gold_claim_ids.append(cid)
            continue

        retrieved_set = set(retrieved_urls)
        found = gold_urls & retrieved_set
        recall = len(found) / len(gold_urls)
        hit = 1.0 if found else 0.0

        recalls.append(recall)
        hits.append(hit)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_hit = sum(hits) / len(hits) if hits else 0.0

    return {
        "n_evaluated":      len(recalls),
        "n_no_gold":        no_gold_count,
        "n_no_idx":         no_idx_count,
        "no_gold_claim_ids": no_gold_claim_ids,
        "avg_recall_at_k":  avg_recall,
        "avg_hit_at_k":     avg_hit,
        "per_claim_recall": recalls,
    }


def print_report(label: str, stats: dict, top_k: int) -> None:
    print(f"\n{'─' * 60}")
    print(f"Recall@{top_k} Report: {label}")
    print(f"{'─' * 60}")
    print(f"Claims evaluated                    : {stats['n_evaluated']}")
    print(f"Claims with no gold URL (excluded)  : {stats['n_no_gold']}")
    if stats['n_no_idx']:
        print(f"Claims with no resolvable claim_idx : {stats['n_no_idx']} "
              f"(re-run bm25_baseline.py for this cache first)")
    print(f"Average Recall@{top_k}                  : {stats['avg_recall_at_k']:.4f}")
    print(f"Hit Rate@{top_k} (>=1 gold URL found)   : {stats['avg_hit_at_k']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Recall@k for a bm25_baseline.py result cache "
                    "against AVeriTeC gold evidence URLs."
    )
    parser.add_argument("--result-cache",
                        help="Single result-cache file to evaluate")
    parser.add_argument("--label", default=None,
                        help="Display label for single-cache mode")
    parser.add_argument("--source-file", required=True,
                        help="Original AVeriTeC JSON the claim_idx values "
                             "were resolved against (train.json for "
                             "internal_dev-derived caches, dev.json for "
                             "official_dev-derived caches)")
    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument("--compare", action="store_true",
                        help="Compare Strategy A vs Strategy B side by side")
    parser.add_argument("--strategy-a-cache",
                        help="Strategy A (direct) result cache, for --compare")
    parser.add_argument("--strategy-b-cache",
                        help="Strategy B (keyword) result cache, for --compare")

    args = parser.parse_args()

    print(f"Loading gold evidence URLs from: {args.source_file}")
    gold_map = load_gold_urls_by_idx(args.source_file)
    n_with_gold = sum(1 for urls in gold_map.values() if urls)
    print(f"  Indexed {len(gold_map)} claims, "
          f"{n_with_gold} have at least one gold URL.")

    if args.compare:
        if not args.strategy_a_cache or not args.strategy_b_cache:
            parser.error("--compare requires --strategy-a-cache and --strategy-b-cache")

        retrieved_a = load_retrieved_urls(args.strategy_a_cache, args.top_k)
        retrieved_b = load_retrieved_urls(args.strategy_b_cache, args.top_k)

        stats_a = compute_recall_at_k(gold_map, retrieved_a)
        stats_b = compute_recall_at_k(gold_map, retrieved_b)

        print_report("Strategy A (Direct Query)", stats_a, args.top_k)
        print_report("Strategy B (Keyword Query)", stats_b, args.top_k)

        print(f"\n{'=' * 60}")
        print("Summary — Strategy A vs Strategy B")
        print(f"{'=' * 60}")
        print(f"{'Metric':<25} {'Strategy A':>12} {'Strategy B':>12} {'Delta':>10}")
        print(f"{'Recall@' + str(args.top_k):<25} "
              f"{stats_a['avg_recall_at_k']:>12.4f} "
              f"{stats_b['avg_recall_at_k']:>12.4f} "
              f"{stats_b['avg_recall_at_k'] - stats_a['avg_recall_at_k']:>+10.4f}")
        print(f"{'Hit@' + str(args.top_k):<25} "
              f"{stats_a['avg_hit_at_k']:>12.4f} "
              f"{stats_b['avg_hit_at_k']:>12.4f} "
              f"{stats_b['avg_hit_at_k'] - stats_a['avg_hit_at_k']:>+10.4f}")

        if stats_a['n_evaluated'] != stats_b['n_evaluated']:
            print(f"\n⚠️  Warning: different claim counts evaluated "
                  f"(A={stats_a['n_evaluated']}, B={stats_b['n_evaluated']}). "
                  f"Make sure both caches were run on the SAME claim set "
                  f"before trusting this comparison.")

        winner = ("Strategy B" if stats_b['avg_recall_at_k'] > stats_a['avg_recall_at_k']
                  else "Strategy A" if stats_a['avg_recall_at_k'] > stats_b['avg_recall_at_k']
                  else "Tie")
        print(f"\n→ Higher Recall@{args.top_k}: {winner}")

    else:
        if not args.result_cache:
            parser.error("--result-cache is required outside --compare mode")
        retrieved = load_retrieved_urls(args.result_cache, args.top_k)
        stats = compute_recall_at_k(gold_map, retrieved)
        label = args.label or os.path.basename(args.result_cache)
        print_report(label, stats, args.top_k)


if __name__ == '__main__':
    main()
