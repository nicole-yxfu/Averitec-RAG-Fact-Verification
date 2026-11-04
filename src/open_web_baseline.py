"""
open_web_baseline.py — DuckDuckGo open-web retrieval on AVeriTeC.

Swaps the closed AVeriTeC Knowledge Store for live open-web search, holding
the LLM and prompt fixed so the only variable is the evidence SOURCE. This is
the "open-web" rung of the evidence-availability ladder (closed KS -> open web
-> gold oracle).

Retrieval configuration:

  - Query = the AVeriTeC claim text, .strip() only, no reformulation (direct).
  - Engine = ddgs.DDGS().text(query, max_results=5)  (defaults; no region /
    safesearch overrides).
  - Retries: up to 3 attempts, sleep(2) between attempts.
  - Per-result throttle: sleep(1) after each of the 5 results is unpacked
    (conservative pacing to avoid rate limits).
  - Evidence unit stored for each hit: dict(title, snippet, url).

Each hit is written as a per-passage tuple (score=0.0, url, "Title: ...\\n
Snippet: ...") so `reprompt_eval.py` consumes this cache unchanged — it uses
the same schema as the BM25 retrieval caches (`top_passages: list[[score,
url, passage]]`). The rendered LLM input via `build_user_input()` then reads
'[i] {url}\\nTitle: ...\\nSnippet: ...'. Keeping retrieval and the LLM step
decoupled behind a shared cache is what lets every ablation change exactly
one component.

Disk cache: results/duckduckgo_cache/{sha1(query)}.json — one file per
UNIQUE query text, so retries after a DDG ban don't re-hit the network for
claims already fetched. Ban-safe by construction.

Usage:
  # s150 pilot (Rule of 150)
  python src/open_web_baseline.py \\
      --data-file data/internal_dev_sample150.json \\
      --output-cache results/ddg_direct_s150_retrieval_cache.json \\
      --sample -1

  # dev500 anchor run
  python src/open_web_baseline.py \\
      --data-file data/dev.json \\
      --output-cache results/ddg_direct_dev500_retrieval_cache.json \\
      --sample -1

Then score with reprompt_eval.py (LLM step is decoupled, per Protocol Rule #2):
  python src/reprompt_eval.py \\
      --data-file data/internal_dev_sample150.json \\
      --retrieval-cache results/ddg_direct_s150_retrieval_cache.json \\
      --output-cache results/ddg_direct_s150_plain_eval_cache.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from ddgs import DDGS

from dataset import read_averitec_json
from bm25_baseline import _write_json_atomic


# Character caps.
# 500/passage keeps the natural (title + snippet + a bit of body) intact —
# DDG snippets are typically 150–250 chars, titles ~60 chars, so this rarely
# truncates real content. reprompt_eval.py -> build_user_input() will further
# cap the concatenated evidence at MAX_EVIDENCE_CHARS=3000.
PASSAGE_CHAR_CAP = 500


def _query_hash(query: str) -> str:
    return hashlib.sha1(query.encode("utf-8")).hexdigest()


def ddg_search_cached(query: str, cache_dir: str, top_k: int = 5,
                      max_retries: int = 3, retry_sleep: float = 2.0,
                      per_result_sleep: float = 1.0) -> list[dict]:
    """
    DuckDuckGo text search with retry + throttling: up to `max_retries`
    attempts, sleep(retry_sleep) between attempts, sleep(per_result_sleep)
    between per-result unpacks. Returns a list of dicts with keys
    (title, snippet, url); [] on total failure.

    Disk-cached at cache_dir/{sha1(query)}.json so the network is hit at
    most once per unique query string across all runs of this script.
    """
    query = (query or "").strip()
    if not query:
        return []

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{_query_hash(query)}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass  # corrupt cache — refetch below

    results: list[dict] = []
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=top_k))
            if raw:
                break
        except Exception as e:
            print(f"    [ddg attempt {attempt + 1}/{max_retries}] "
                  f"failed for query: {query[:80]!r}  err={e}")
            raw = []
        time.sleep(retry_sleep)

    for r in raw:
        results.append({
            "title":   (r.get("title") or "").strip(),
            "snippet": (r.get("body") or "").strip(),
            "url":     (r.get("href") or "").strip(),
        })
        time.sleep(per_result_sleep)

    _write_json_atomic(cache_path, results)
    return results


def hits_to_passages(hits: list[dict]) -> list[tuple[float, str, str]]:
    """
    Pack DDG hits into the `top_passages` schema reprompt_eval.py expects:
      list of (score, url, passage_text)
    We have no ranking score from DDG (results already come ranked), so the
    per-hit score is fixed at 0.0 — reprompt_eval / build_user_input do not
    use the score field for anything except a diagnostic table.
    """
    packed = []
    for h in hits:
        title = h.get("title", "")
        snippet = h.get("snippet", "")
        passage = f"Title: {title}\nSnippet: {snippet}"[:PASSAGE_CHAR_CAP]
        packed.append((0.0, h.get("url", ""), passage))
    return packed


def run_open_web_baseline(data_file: str, output_cache: str,
                          cache_dir: str, sample: int,
                          request_sleep: float):
    texts = read_averitec_json(data_file)
    texts = [t for t in texts if t.label is not None]
    if sample > 0:
        texts = texts[:sample]

    print(f"Loaded {len(texts)} claims from {data_file}")
    print(f"Retrieval cache : {output_cache}")
    print(f"Per-query disk cache dir : {cache_dir}")

    os.makedirs(os.path.dirname(output_cache) or ".", exist_ok=True)

    final_cache = {}
    if os.path.exists(output_cache):
        with open(output_cache, "r", encoding="utf-8") as f:
            final_cache = json.load(f)
        print(f"Resume: {len(final_cache)} pre-existing entries")

    n_empty = 0
    total_hits = 0
    total_snippet_chars = 0

    for i, text in enumerate(texts, 1):
        cid = text.text_id
        if cid in final_cache:
            entry = final_cache[cid]
            hits = entry.get("top_passages", [])
            if hits:
                total_hits += len(hits)
                total_snippet_chars += sum(len(p) for _, _, p in hits)
            else:
                n_empty += 1
            continue

        query = (text.original_claim or "").strip()
        hits = ddg_search_cached(query, cache_dir)
        passages = hits_to_passages(hits)

        entry = {
            "claim":          text.original_claim,
            "speaker":        text.speaker,
            "source":         text.reporting_source,
            "claim_idx":      i - 1,
            "query_strategy": "direct",
            "query_text":     query,
            "n_passages":     len(passages),
            "top_passages":   passages,
            "input_mode":     "with_metadata",
            "true_label":     text.label,
            "raw_output":     "",
            "prompt_tokens":  0,
            "completion_tokens": 0,
        }
        final_cache[cid] = entry
        _write_json_atomic(output_cache, final_cache)

        if passages:
            total_hits += len(passages)
            total_snippet_chars += sum(len(p) for _, _, p in passages)
            status = f"{len(passages)} hits"
        else:
            n_empty += 1
            status = "NO EVIDENCE"

        print(f"  {i}/{len(texts)} | {cid} | {status}")
        time.sleep(request_sleep)

    n_non_empty = len(texts) - n_empty
    mean_pass_len = total_snippet_chars / total_hits if total_hits else 0.0
    print("\n" + "=" * 60)
    print("Open-web DDG retrieval summary")
    print(f"  Claims processed         : {len(texts)}")
    print(f"  Claims with ≥1 hit       : {n_non_empty} "
          f"({100 * n_non_empty / len(texts):.1f}%)")
    print(f"  Claims with 0 hits       : {n_empty}")
    print(f"  Total hits stored        : {total_hits}")
    print(f"  Mean passage length      : {mean_pass_len:.0f} chars")
    print(f"  Output retrieval cache   : {output_cache}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-file", required=True,
                        help="AVeriTeC JSON (dev.json or internal_dev_sample150.json)")
    parser.add_argument("--output-cache", required=True,
                        help="Where to write the retrieval cache "
                             "(top_passages schema; reprompt_eval.py-compatible)")
    parser.add_argument("--cache-dir", default="results/duckduckgo_cache",
                        help="Per-query disk cache; ban-safe resumability.")
    parser.add_argument("--sample", type=int, default=-1,
                        help="Process only the first N claims (-1 = all)")
    parser.add_argument("--request-sleep", type=float, default=2.0,
                        help="Sleep between DDG calls (burst is the main ban "
                             "cause; 2.0s is conservative for no-API-key use)")
    args = parser.parse_args()

    run_open_web_baseline(
        data_file=args.data_file,
        output_cache=args.output_cache,
        cache_dir=args.cache_dir,
        sample=args.sample,
        request_sleep=args.request_sleep,
    )
