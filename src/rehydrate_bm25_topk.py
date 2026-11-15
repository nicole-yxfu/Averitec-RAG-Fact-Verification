"""
rehydrate_bm25_topk.py — recover the FULL text of the BM25 passages behind
the published closed-corpus runs, and measure how much of it the LLM saw.

WHY THIS IS NEEDED
------------------
`bm25_baseline.py` stored each retrieved passage in its result cache as a
300-character preview (`p[:300]`) to keep the cache small. Two things
downstream silently depended on that preview being the passage:

  1. `reprompt_eval.py` re-prompts FROM the cache, so every re-prompted
     BM25 run (plain prompt 0.3644, few-shot, ...) showed the LLM five
     300-character previews — not the ~1,100-1,400-character chunks the
     original standard-prompt run (0.3241) was given.
  2. The original run itself built its prompt with `build_user_input`,
     which stops adding passages once the evidence reaches
     MAX_EVIDENCE_CHARS = 3000. With full chunks that is about two
     passages, not five.

So "BM25 top-5" meant two different things in two published rows, and the
gap between them is not a pure prompt effect. This script recomputes the
rankings from the cached tokenized corpus (no KS drive needed), checks that
they are the published rankings, and records for every claim how many
passages each evidence view actually delivered.

CORRECTNESS CHECK
-----------------
For ranks 1-5 the recomputed passage must have the same URL, the same score
to 4 decimals, and full text that STARTS WITH the cached preview. Any claim
failing that is flagged, not silently replaced.

RESUMABLE
---------
One JSON line per claim is appended as soon as it is done; on restart,
finished claims are skipped. Closing the laptop mid-run loses at most the
claims in flight. Ranking caches are also written at depth 10, so a second
pass over the same claims is fast.

USAGE
-----
    python src/rehydrate_bm25_topk.py --workers 4
    python src/rehydrate_bm25_topk.py --summary      # just report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_DIR, "results", "bm25_dev500_fulltext_top10.jsonl")
DEFAULT_TOP5 = os.path.join(
    REPO_DIR, "results", "bm25_baseline_official_dev_with_metadata_top5_cache.json")
PREVIEW_CHARS = 300


def passages_seen(passages: list[tuple[float, str, str]],
                  max_chars: int) -> int:
    """
    How many passages build_user_input() actually put in the prompt.

    Mirrors its loop exactly: each entry is "[rank] url\\npassage", and the
    loop BREAKS (not skips) at the first entry that would cross the cap.
    """
    total = 0
    for rank, (_, url, passage) in enumerate(passages, 1):
        entry = f"[{rank}] {url}\n{passage}"
        if total + len(entry) > max_chars:
            return rank - 1
        total += len(entry)
    return len(passages)


def _work(task: tuple) -> dict:
    claim_id, claim_idx, claim, cached_top5, depth, ks_root, corpus_dir, rank_dir = task
    from collections import defaultdict
    from bm25_baseline import (CHUNK_SIZE, MAX_EVIDENCE_CHARS, get_ks_file,
                               retrieve_bm25_cached)
    t0 = time.time()
    timings, counters = defaultdict(float), defaultdict(int)
    ranked, n_docs = retrieve_bm25_cached(
        query_text=claim, claim_idx=claim_idx,
        ks_file=get_ks_file(ks_root, "dev", claim_idx), split_ks_name="dev",
        top_k=depth, chunk_size=CHUNK_SIZE, corpus_cache_dir=corpus_dir,
        ranking_cache_dir=rank_dir, ranking_depth=depth,
        timings=timings, counters=counters)
    ranked = [(float(s), u, t) for s, u, t in ranked]
    if not ranked and cached_top5:
        # The published run retrieved passages for this claim, so an empty
        # ranking means the corpus is unavailable, not that it is empty.
        # Recording it would mark the claim "done" and skip it forever.
        raise RuntimeError(
            f"{claim_id}: corpus unavailable (n_docs={n_docs}) but the "
            f"published run retrieved {len(cached_top5)} passages. Rebuild "
            f"the corpus cache (src/warmup_dev_corpus.py) first.")

    problems = []
    for i, (s_old, u_old, t_old) in enumerate(cached_top5):
        if i >= len(ranked):
            problems.append(f"rank{i + 1}: missing")
            continue
        s, u, t = ranked[i]
        if u != u_old:
            problems.append(f"rank{i + 1}: url")
        elif round(s, 4) != round(float(s_old), 4):
            problems.append(f"rank{i + 1}: score {s:.4f}!={float(s_old):.4f}")
        elif not t.startswith(t_old):
            problems.append(f"rank{i + 1}: text prefix")

    top5_full = ranked[:5]
    previews = [(s, u, t[:PREVIEW_CHARS]) for s, u, t in top5_full]
    return {
        "claim_id": claim_id, "claim_idx": claim_idx, "n_docs": n_docs,
        "ranked": ranked,
        "matches_published_top5": not problems,
        "problems": problems,
        "seen_full_chunks": passages_seen(top5_full, MAX_EVIDENCE_CHARS),
        "seen_previews": passages_seen(previews, MAX_EVIDENCE_CHARS),
        "evidence_chars_full": sum(len(t) for _, _, t in top5_full),
        "seconds": round(time.time() - t0, 1),
    }


def load_done(path: str) -> dict[str, dict]:
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # torn final line from a hard stop
                    done[row["claim_id"]] = row
    return done


def summarize(rows: dict[str, dict], n_total: int) -> None:
    vals = list(rows.values())
    print(f"\n{len(vals)}/{n_total} claims rehydrated")
    if not vals:
        return
    bad = [r["claim_id"] for r in vals if not r["matches_published_top5"]]
    print(f"  match published top-5: {len(vals) - len(bad)}/{len(vals)}"
          + (f"   MISMATCH: {bad[:10]}" if bad else ""))
    for key, label in (("seen_full_chunks", "full chunks (standard run, 0.3241)"),
                       ("seen_previews", "300-char previews (re-prompted runs, 0.3644)")):
        c = Counter(r[key] for r in vals)
        mean = sum(r[key] for r in vals) / len(vals)
        dist = "  ".join(f"{k}:{c[k]}" for k in sorted(c))
        print(f"  passages the LLM saw, {label}: mean {mean:.2f}   [{dist}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--top5-cache", default=DEFAULT_TOP5)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--ks-root", default=("/Volumes/Seagate Backup Plus Drive/"
                                          "AVeriTeC_knowledge_store/data_store/"
                                          "knowledge_store"))
    ap.add_argument("--corpus-cache", default=os.path.join(
        REPO_DIR, "results", "bm25_corpus_cache"))
    ap.add_argument("--ranking-cache", default=os.path.join(
        REPO_DIR, "results", "bm25_ranking_cache"))
    args = ap.parse_args()

    with open(args.top5_cache, encoding="utf-8") as f:
        top5 = json.load(f)
    ids = sorted(top5, key=lambda c: int(c.split("_")[1]))
    if args.limit:
        ids = ids[:args.limit]

    done = load_done(args.out)
    if args.summary:
        summarize(done, len(ids))
        return

    todo = [c for c in ids if c not in done and top5[c].get("top_passages")]
    skipped_empty = [c for c in ids if not top5[c].get("top_passages")]
    print(f"{len(done)} already done, {len(todo)} to go, "
          f"{len(skipped_empty)} with no KS passages at all (skipped)")

    tasks = [(c, int(top5[c]["claim_idx"]), top5[c]["claim"],
              top5[c]["top_passages"], args.depth, args.ks_root,
              args.corpus_cache, args.ranking_cache) for c in todo]

    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as out, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_work, t): t[0] for t in tasks}
        for n, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            done[row["claim_id"]] = row
            if n % 10 == 0 or n == len(tasks):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(tasks)}  {rate:.1f} s/claim  "
                      f"eta {rate * (len(tasks) - n) / 60:.0f} min", flush=True)

    summarize(done, len(ids))


if __name__ == "__main__":
    main()
