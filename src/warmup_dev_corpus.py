"""
warmup_dev_corpus.py — Parallel multi-process warm-up for the
official_dev BM25 corpus cache.

Why this exists: dense_rerank.py on dev (500 claims) was projected at
~13 hours single-threaded, because ~466 of 500 dev claims had no
corpus cache yet — every retrieve_bm25_cached call had to read the KS
file from disk and run NLTK tokenisation/stemming over potentially
800+ documents per claim. That step is CPU-bound and embarrassingly
parallel across claims, but dense_rerank.py runs it serially.

This script reuses bm25_baseline.warm_up_corpus_cache_parallel
(ProcessPoolExecutor, default 8 workers) to build the missing corpus
caches in parallel. After it completes, dense_rerank.py --retrieval-only
hits the corpus cache for every claim and finishes in minutes (mpnet
re-rank only, no tokenisation).

KS root: $WARMUP_KS_ROOT, else the Seagate drive if mounted, else
~/local_ks_cache. Already-built corpora are skipped, so the run is
resumable: stop it, unplug, and re-run later.

    WARMUP_WORKERS=2 python src/warmup_dev_corpus.py

Use 2 workers on an 8 GB machine. The 2026-11-13 cache loss happened under
memory pressure from 3 workers each holding a multi-GB corpus.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

from dataset import read_averitec_json
from bm25_baseline import (
    build_text_id_to_source_idx,
    get_ks_file,
    warm_up_corpus_cache_parallel,
    CHUNK_SIZE,
)


def main():
    data_file = os.path.join(PROJECT_DIR, "data", "dev.json")
    source_idx_file = data_file  # official_dev maps to itself
    # KS root: WARMUP_KS_ROOT if set, else the external drive if mounted,
    # else the old local SSD copy. The local copy was deleted to free space,
    # so in practice this is the Seagate drive.
    drive = ("/Volumes/Seagate Backup Plus Drive/"
             "AVeriTeC_knowledge_store/data_store/knowledge_store")
    ks_root = (os.environ.get("WARMUP_KS_ROOT")
               or (drive if os.path.isdir(drive)
                   else os.path.expanduser("~/local_ks_cache")))
    split_ks_name = "dev"
    if not os.path.isdir(os.path.join(ks_root, split_ks_name)):
        # Refuse up front rather than run 500 claims that each find no KS
        # file. (Since 2026-11-13 a missing KS file no longer overwrites a
        # cache, but a run that silently does nothing is still a failure.)
        sys.exit(f"KS directory not found: {os.path.join(ks_root, split_ks_name)}\n"
                 f"Mount the Knowledge Store drive or set WARMUP_KS_ROOT.")
    corpus_cache_dir = os.path.join(PROJECT_DIR, "results", "bm25_corpus_cache")
    # Override via CLI / env. 8 workers OOM'd on dev (some claims have 800+
    # docs, peak hundreds of MB per worker × 8 ≈ spike beyond 16GB RAM).
    # Default 4 is a safer M2 fit; bump back to 8 only if you know the
    # claim mix is small.
    workers = int(os.environ.get("WARMUP_WORKERS", "4"))
    limit = int(os.environ.get("WARMUP_LIMIT", "0"))

    texts = read_averitec_json(data_file)
    texts = [t for t in texts if t.label is not None]
    text_id_to_idx = build_text_id_to_source_idx(data_file, source_idx_file)

    pending = []
    for t in texts:
        claim_idx = text_id_to_idx.get(t.text_id)
        if claim_idx is None:
            continue
        ks_file = get_ks_file(ks_root, split_ks_name, claim_idx)
        pending.append((t.text_id, claim_idx, ks_file, split_ks_name))

    if limit > 0:
        pending = pending[:limit]
        print(f"[smoke test mode] limited to first {limit} claims")
    print(f"Warming up {len(pending)} dev corpora with {workers} workers "
          f"(local SSD KS root: {ks_root}) ...", flush=True)
    stats = warm_up_corpus_cache_parallel(
        pending_claims=pending,
        chunk_size=CHUNK_SIZE,
        corpus_cache_dir=corpus_cache_dir,
        max_workers=workers,
    )
    print(f"\nDone. hits={stats.get('hits',0)}  misses={stats.get('misses',0)}  "
          f"ks_missing={stats.get('ks_missing',0)}  "
          f"errors={stats.get('errors',0)}  elapsed={stats.get('elapsed',0):.1f}s")


if __name__ == '__main__':
    main()
