"""
hybrid_retrieval.py — hybrid retrieval (lexical + dense via RRF) and
cross-encoder re-ranking on the open-web evidence rung.

WHY THE OPEN WEB AND NOT THE KNOWLEDGE STORE
--------------------------------------------
The closed AVeriTeC Knowledge Store lives on an external drive and its
tokenised-corpus cache takes ~10 h to rebuild (see warmup_dev_corpus.py), so
re-running wide-pool BM25 over it is not reproducible from a clean checkout.
The open-web rung is the one the deployed service actually uses
(src/service.py) and the stronger of the two (Macro-F1 0.4495 vs 0.3644), so
any gain found here is shippable rather than archival. The same ranking code
path works on a Knowledge-Store pool if one is available — only the
candidate-pool producer differs.

THE PIPELINE
------------
    stage 1  pool     DDG text search, max_results = --pool-size (default 30)
    stage 2  rank     score the pool with up to three independent rankers
    stage 3  fuse     Reciprocal Rank Fusion over those rankers
    stage 4  rerank   cross-encoder over the fused head (optional)
    stage 5  cut      keep --top-k passages

The three stage-2 rankers are deliberately different in kind:

  engine  the order DuckDuckGo itself returned. This is NOT a throwaway
          ranker — the dev500 baseline *is* engine-order top-5, so a fusion
          that discards it would be competing against a strawman. It carries
          signal no local ranker has (link graph, freshness, click data).
  bm25    rank_bm25 Okapi over the pooled passages, using bm25_baseline's
          own _tokenize (lowercase, depunctuate, stopword-filter, Porter
          stem) so lexical matching is identical to the Knowledge-Store
          baseline.
  dense   cosine(claim, passage) from sentence-transformers
          all-mpnet-base-v2 — catches paraphrase that BM25 cannot
          ("capital city" vs "seat of government").

RRF rather than a weighted score sum: the three rankers' scores are on
incomparable scales (DDG gives no score at all), and RRF needs no weight
tuning, which there is no held-out budget to do honestly.

    rrf(d) = sum over rankers r of  1 / (k + rank_r(d))      k = 60

DECOUPLE AND CACHE
------------------
The network stage and the ranking stage are separate. Stage 1 writes the raw
pool to --pool-file once; every ranking variant is then recomputed from that
file with zero network calls, and the LLM step is deferred entirely to
reprompt_eval.py. Each variant is written in the shared `top_passages`
schema, so reprompt_eval.py and rag_eval.py consume it unchanged and every
ablation differs in exactly one component.

USAGE
-----
    # stage 1 only — fetch the candidate pool (slow, network-bound)
    python src/hybrid_retrieval.py --data-file data/dev.json --sample 150 \\
        --pool-file results/hybrid_dev150_pool30.json --pool-only

    # stages 2-5 — every variant, no network, ~1 min on CPU
    python src/hybrid_retrieval.py --data-file data/dev.json --sample 150 \\
        --pool-file results/hybrid_dev150_pool30.json \\
        --out-prefix results/hybrid_dev150

    # then score each variant under the SAME plain prompt
    python src/reprompt_eval.py --data-file data/dev.json \\
        --retrieval-cache results/hybrid_dev150_rrf_retrieval_cache.json \\
        --output-cache    results/hybrid_dev150_rrf_eval_cache.json
"""
import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from rank_bm25 import BM25Okapi

from dataset import read_averitec_json
from bm25_baseline import _tokenize, _write_json_atomic
from open_web_baseline import ddg_search_cached, hits_to_passages

# Standard RRF constant from Cormack et al. 2009. Large k flattens the
# contribution curve so no single ranker's top hit can dominate the fusion.
RRF_K = 60

VARIANTS = ("engine", "bm25", "dense", "rrf", "rrf_ce")


# ----------------------------------------------------------------------
# stage 1 — candidate pool
# ----------------------------------------------------------------------
def build_pool(data_file: str, sample: int, pool_file: str, pool_size: int,
               cache_dir: str, request_sleep: float,
               per_result_sleep: float) -> dict:
    """
    Fetch `pool_size` open-web hits per claim and persist them.

    The per-query disk cache is keyed on sha1(query) ALONE, with no pool
    size in the key, so the existing top-5 cache directory would silently
    satisfy a request for 30 and cap the pool at 5. A separate cache_dir is
    therefore mandatory for wide pools — the default is
    results/duckduckgo_pool30_cache, not the top-5 directory.
    """
    texts = read_averitec_json(data_file)
    texts = [t for t in texts if t.label is not None]
    if sample > 0:
        texts = texts[:sample]

    pool = {}
    if os.path.exists(pool_file):
        with open(pool_file, "r", encoding="utf-8") as f:
            pool = json.load(f)
        print(f"Resume: {len(pool)} claims already pooled")

    print(f"Pooling {len(texts)} claims at pool_size={pool_size}")
    print(f"Per-query cache dir : {cache_dir}")

    t0 = time.perf_counter()
    for i, text in enumerate(texts, 1):
        cid = text.text_id
        if cid in pool:
            continue

        query = (text.original_claim or "").strip()
        hits = ddg_search_cached(query, cache_dir, top_k=pool_size,
                                 per_result_sleep=per_result_sleep)
        pool[cid] = {
            "claim": text.original_claim,
            "speaker": text.speaker,
            "source": text.reporting_source,
            "claim_idx": i - 1,
            "query_text": query,
            "true_label": text.label,
            # Engine order is preserved by list position — that IS the
            # `engine` ranker, so never sort this list in place.
            "pool": hits,
        }
        _write_json_atomic(pool_file, pool)
        print(f"  {i}/{len(texts)} | {cid} | {len(hits)} hits")
        time.sleep(request_sleep)

    sizes = [len(v["pool"]) for v in pool.values()]
    n_empty = sum(1 for s in sizes if s == 0)
    print("\n" + "=" * 60)
    print("Candidate pool summary")
    print(f"  Claims            : {len(pool)}")
    print(f"  Claims with 0 hits: {n_empty}")
    print(f"  Mean pool size    : {sum(sizes) / len(sizes):.1f}"
          if sizes else "  Mean pool size    : 0")
    print(f"  Max pool size     : {max(sizes) if sizes else 0}")
    print(f"  Elapsed           : {time.perf_counter() - t0:.0f} s")
    print(f"  Pool file         : {pool_file}")
    return pool


# ----------------------------------------------------------------------
# stage 2 — the three rankers
# ----------------------------------------------------------------------
def rank_engine(hits: list[dict]) -> list[int]:
    """Pool indices in the order DuckDuckGo returned them."""
    return list(range(len(hits)))


def rank_bm25_pool(claim: str, passages: list[str]) -> list[int]:
    """
    Pool indices by Okapi BM25 over the pooled passages.

    BM25 is fitted on this claim's pool only (~30 short documents), not on a
    global corpus, so the IDF statistics are local. That is the correct
    choice here: the task is to re-order one candidate set, and a term that
    is rare across the whole web but common in these 30 hits genuinely
    carries little discriminative power *for this decision*.

    CAVEAT on very small pools. rank_bm25's Okapi IDF is
    log(N - n + 0.5) - log(n + 0.5), which is exactly 0 when N = 2 and
    n = 1, and negative (floored by its epsilon) for terms in most of a
    tiny pool. So a claim with only 2-3 hits scores every passage 0 and
    this function degrades to pool order — i.e. silently back to the
    `engine` ranker. That is the right fallback (there is no lexical
    signal to extract from two documents) but it means the lexical arm of
    the fusion contributes nothing for such claims. Count them before
    attributing a null result to BM25 being useless.
    """
    tokenized = [_tokenize(p) for p in passages]
    # An all-empty corpus (every passage stopworded away) makes BM25Okapi
    # divide by a zero average document length.
    if not any(tokenized):
        return list(range(len(passages)))
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(claim))
    return sorted(range(len(passages)), key=lambda i: -scores[i])


def rank_dense(st_model, claim: str, passages: list[str]) -> list[int]:
    """Pool indices by cosine(claim, passage) under all-mpnet-base-v2."""
    from sentence_transformers import util
    claim_emb = st_model.encode(claim, convert_to_tensor=True)
    passage_embs = st_model.encode(passages, convert_to_tensor=True)
    scores = util.cos_sim(claim_emb, passage_embs)[0].cpu().numpy()
    return sorted(range(len(passages)), key=lambda i: -scores[i])


# ----------------------------------------------------------------------
# stage 3 — fusion
# ----------------------------------------------------------------------
def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """
    Reciprocal Rank Fusion. `rankings` is a list of ranked index lists over
    the same candidate set; returns the fused ranking.

    Ties are broken by original pool position (stable sort over an ascending
    index enumeration), which keeps the output deterministic across runs.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


# ----------------------------------------------------------------------
# stage 4 — cross-encoder re-rank
# ----------------------------------------------------------------------
def rerank_cross_encoder(ce_model, claim: str, passages: list[str],
                         order: list[int], depth: int) -> list[int]:
    """
    Re-score the head of `order` with a cross-encoder and return the full
    ranking with that head re-ordered.

    Only the top `depth` are re-scored: a cross-encoder runs the claim and
    the passage through one transformer jointly, so it costs a forward pass
    per pair and does not scale to the whole pool. The tail keeps its fused
    order, which is what makes this a re-ranker rather than a retriever.

    The NaN check is not defensive padding. ms-marco-MiniLM-L-6-v2 emits NaN
    from its first encoder block in this environment (torch 2.6 /
    transformers 4.51 on Apple Silicon; embeddings are finite, layer 1 is
    not, and the same stack runs all-mpnet-base-v2 correctly). NaN compares
    False against everything, so `sorted` leaves the list untouched and the
    re-ranker silently becomes the identity function — it produced a
    plausible-looking results table in which rrf_ce was byte-identical to
    rrf across all 150 claims. A null result that is really a dead model is
    the worst possible outcome here, so this raises instead.
    """
    head, tail = order[:depth], order[depth:]
    if not head:
        return order
    pairs = [(claim, passages[i]) for i in head]
    scores = [float(s) for s in ce_model.predict(pairs)]
    if any(s != s for s in scores):          # NaN != NaN
        raise RuntimeError(
            f"cross-encoder returned NaN for {sum(1 for s in scores if s != s)}"
            f"/{len(scores)} pairs. The model is broken in this environment; "
            f"re-ranking would silently degrade to the identity. Try "
            f"--ce-model cross-encoder/ms-marco-MiniLM-L-12-v2.")
    head_sorted = [i for _, i in sorted(zip(scores, head),
                                        key=lambda t: (-t[0], t[1]))]
    return head_sorted + tail


def assert_cross_encoder_works(ce_model) -> None:
    """
    Fail fast at startup on a pair whose ordering is not in question.

    Checks finiteness AND discrimination: a model that returns a constant
    for every pair also reduces re-ranking to the identity without raising,
    which the NaN check alone would not catch.
    """
    probe = [("who directed the film Jaws",
              "Steven Spielberg directed Jaws, released in 1975."),
             ("who directed the film Jaws",
              "A recipe for sourdough bread starts with a flour starter.")]
    s = [float(x) for x in ce_model.predict(probe)]
    if any(x != x for x in s):
        raise RuntimeError(f"cross-encoder self-test returned NaN: {s}")
    if s[0] <= s[1]:
        raise RuntimeError(
            f"cross-encoder self-test failed: it scored an irrelevant "
            f"passage at least as high as a relevant one ({s}). Re-ranking "
            f"with it would be noise.")
    print(f"  cross-encoder self-test OK (relevant {s[0]:.2f} > "
          f"irrelevant {s[1]:.2f})")


# ----------------------------------------------------------------------
# stages 2-5 driver
# ----------------------------------------------------------------------
def rank_all_variants(pool: dict, out_prefix: str, top_k: int,
                      variants: list[str], ce_depth: int,
                      device: str, ce_model_name: str) -> dict[str, str]:
    """
    Produce one reprompt_eval.py-compatible retrieval cache per variant.

    Every variant reads the SAME pool, so a difference in downstream
    Macro-F1 is attributable to ranking alone — not to a different set of
    pages having been fetched on a different day.
    """
    need_dense = any(v in ("dense", "rrf", "rrf_ce") for v in variants)
    need_ce = "rrf_ce" in variants

    st_model = ce_model = None
    if need_dense:
        from sentence_transformers import SentenceTransformer
        # CPU by default: mpnet on this Mac's MPS backend deadlocks inside
        # Metal on long inputs (the same failure dense_rerank.py documents).
        print(f"Loading all-mpnet-base-v2 on device={device} ...")
        st_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    if need_ce:
        from sentence_transformers import CrossEncoder
        print(f"Loading {ce_model_name} on device={device} ...")
        ce_model = CrossEncoder(ce_model_name, device=device)
        assert_cross_encoder_works(ce_model)

    caches: dict[str, dict] = {v: {} for v in variants}
    # Diagnostic: how much each variant actually moves the evidence set. A
    # variant whose top-k is identical to the baseline's cannot change any
    # verdict, so a null result would be a no-op, not a negative finding.
    overlap_sum = {v: 0 for v in variants}
    n_ranked = 0

    t0 = time.perf_counter()
    for n, (cid, entry) in enumerate(pool.items(), 1):
        hits = entry["pool"]
        passages = [f"Title: {h.get('title','')}\nSnippet: {h.get('snippet','')}"
                    for h in hits]
        claim = entry["claim"]

        orders: dict[str, list[int]] = {}
        if hits:
            eng = rank_engine(hits)
            orders["engine"] = eng
            if "bm25" in variants:
                orders["bm25"] = rank_bm25_pool(claim, passages)
            if need_dense:
                bm = orders.get("bm25") or rank_bm25_pool(claim, passages)
                dn = rank_dense(st_model, claim, passages)
                orders["dense"] = dn
                fused = rrf_fuse([eng, bm, dn])
                orders["rrf"] = fused
                if need_ce:
                    orders["rrf_ce"] = rerank_cross_encoder(
                        ce_model, claim, passages, fused, ce_depth)
            n_ranked += 1

        baseline_top = set(orders.get("engine", [])[:top_k])
        for v in variants:
            order = orders.get(v, [])
            chosen = [hits[i] for i in order[:top_k]]
            packed = hits_to_passages(chosen)
            caches[v][cid] = {
                "claim": claim,
                "speaker": entry.get("speaker"),
                "source": entry.get("source"),
                "claim_idx": entry.get("claim_idx"),
                "query_strategy": f"openweb_pool/{v}",
                "query_text": entry.get("query_text"),
                "n_passages": len(packed),
                "top_passages": packed,
                "input_mode": "with_metadata",
                "true_label": entry["true_label"],
                "raw_output": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
            if hits:
                overlap_sum[v] += len(baseline_top & set(order[:top_k]))

        if n % 25 == 0:
            print(f"  ... ranked {n}/{len(pool)}")

    written = {}
    for v in variants:
        path = f"{out_prefix}_{v}_retrieval_cache.json"
        _write_json_atomic(path, caches[v])
        written[v] = path

    print("\n" + "=" * 60)
    print(f"Ranked {n_ranked} non-empty pools in "
          f"{time.perf_counter() - t0:.0f} s  (top_k={top_k})")
    print(f"\n| Variant | Mean top-{top_k} overlap with engine order | Cache |")
    print("|---|---|---|")
    for v in variants:
        mean_ov = overlap_sum[v] / n_ranked if n_ranked else 0.0
        print(f"| {v} | {mean_ov:.2f} / {top_k} | "
              f"{os.path.basename(written[v])} |")
    print("\nScore each with reprompt_eval.py under the same --prompt-style "
          "plain, then compare with src/compare_runs.py.")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-file", default="data/dev.json")
    ap.add_argument("--sample", type=int, default=-1,
                    help="Use only the first N labelled claims (-1 = all).")
    ap.add_argument("--pool-file", required=True,
                    help="Where the raw candidate pool is persisted.")
    ap.add_argument("--pool-size", type=int, default=30,
                    help="Open-web hits requested per claim.")
    ap.add_argument("--pool-cache-dir",
                    default="results/duckduckgo_pool30_cache",
                    help="Per-query disk cache for the WIDE pool. Must not "
                         "be the top-5 directory: the cache key is "
                         "sha1(query) with no pool size in it, so the "
                         "narrow cache would cap the pool at 5.")
    ap.add_argument("--pool-only", action="store_true",
                    help="Stop after stage 1 (fetch the pool, rank nothing).")
    ap.add_argument("--out-prefix", default=None,
                    help="Prefix for the per-variant retrieval caches. "
                         "Required unless --pool-only.")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help=f"Comma-separated subset of {','.join(VARIANTS)}.")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Passages handed to the LLM. 5 matches the baseline.")
    ap.add_argument("--ce-depth", type=int, default=15,
                    help="How deep into the fused ranking the cross-encoder "
                         "re-scores.")
    ap.add_argument("--ce-model",
                    default="cross-encoder/ms-marco-MiniLM-L-12-v2",
                    help="Cross-encoder for the rrf_ce variant. NOT the "
                         "L-6 checkpoint: both of its repo spellings emit "
                         "NaN from the first encoder block here, which turns "
                         "re-ranking into a silent no-op. A startup "
                         "self-test rejects any model that does this.")
    ap.add_argument("--request-sleep", type=float, default=2.0)
    ap.add_argument("--per-result-sleep", type=float, default=0.2,
                    help="Post-hoc pacing per unpacked hit. The default 1.0s "
                         "in open_web_baseline would add 30 s per claim at "
                         "pool_size=30; the network call has already "
                         "returned by then, so 0.2 is ample.")
    ap.add_argument("--device", default=os.environ.get("EMBED_DEVICE", "cpu"))
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"Unknown variant(s) {bad}; choose from {VARIANTS}")
    if not args.pool_only and not args.out_prefix:
        raise SystemExit("--out-prefix is required unless --pool-only")

    pool = build_pool(args.data_file, args.sample, args.pool_file,
                      args.pool_size, args.pool_cache_dir,
                      args.request_sleep, args.per_result_sleep)

    if args.pool_only:
        print("\n--pool-only: stopping before the ranking stage.")
        return

    rank_all_variants(pool, args.out_prefix, args.top_k, variants,
                      args.ce_depth, args.device, args.ce_model)


if __name__ == "__main__":
    main()
