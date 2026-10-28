"""
Text-overlap Recall@5 evaluation — dual criterion.

Reports two criteria side by side:
  - Asymmetric (gold-coverage): |gold ∩ passage| / |gold|
      Interpretation: how much of the gold answer's content is recoverable
      from a retrieved passage. Insensitive to passage length.
  - Symmetric: |gold ∩ passage| / max(|gold|, |passage|)
      Stricter; penalizes long passages with little gold content (low precision).

Both are reported because URL exact-match is known to underestimate true recall
on AVeriTeC (gold cached on web.archive.org). Neither is the "right" answer;
together they bracket the true recall.
"""
import argparse
import json
import re


def tokenize(text: str) -> set:
    if not text:
        return set()
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


def overlap_asymmetric(gold: str, passage: str) -> float:
    g, p = tokenize(gold), tokenize(passage)
    if not g:
        return 0.0
    return len(g & p) / len(g)


def overlap_symmetric(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def evaluate(cache_path: str, source_path: str, threshold: float):
    cache = json.load(open(cache_path))
    source = json.load(open(source_path))

    stats = {
        "asym": {"hit": 0, "recall_sum": 0.0},
        "sym":  {"hit": 0, "recall_sum": 0.0},
    }
    n_claims = 0
    n_skipped = 0

    for entry in cache.values():
        ci = entry.get("claim_idx")
        if not isinstance(ci, int) or ci >= len(source):
            n_skipped += 1
            continue
        retrieved = [p[2] for p in entry.get("top_passages", [])[:5]
                     if len(p) > 2 and p[2]]
        gold_texts = [ev.get("answer", "").strip()
                      for qa in source[ci].get("questions", [])
                      for ev in qa.get("answers", [])
                      if ev.get("answer", "").strip()]
        if not gold_texts or not retrieved:
            n_skipped += 1
            continue
        n_claims += 1

        for name, fn in [("asym", overlap_asymmetric),
                         ("sym",  overlap_symmetric)]:
            covered = sum(
                1 for g in gold_texts
                if any(fn(g, r) >= threshold for r in retrieved)
            )
            if covered > 0:
                stats[name]["hit"] += 1
            stats[name]["recall_sum"] += covered / len(gold_texts)

    print(f"\n{'='*64}")
    print(f"Text-overlap evaluation @ threshold={threshold}")
    print(f"Cache: {cache_path}")
    print(f"{'='*64}")
    print(f"Claims evaluated : {n_claims}  (skipped: {n_skipped})")
    print(f"")
    print(f"{'Criterion':<28} {'Hit@5':>10} {'Recall@5':>10}")
    print(f"{'-'*50}")
    for name, label in [("asym", "Asymmetric (gold-coverage)"),
                        ("sym",  "Symmetric")]:
        hit = stats[name]["hit"] / n_claims if n_claims else 0
        rec = stats[name]["recall_sum"] / n_claims if n_claims else 0
        print(f"{label:<28} {hit:>10.4f} {rec:>10.4f}")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--source-file", default="data/train.json")
    p.add_argument("--threshold", type=float, default=0.3)
    a = p.parse_args()
    evaluate(a.cache, a.source_file, a.threshold)
