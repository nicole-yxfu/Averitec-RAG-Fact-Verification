"""
bm25_baseline.py  —  BM25 Retrieval + LLM Fact Verification (NLTK Enhanced)

Phase 2 baseline: for each claim, retrieves the top-k most relevant passages
from the AVeriTeC Knowledge Store using BM25 Okapi, then passes the retrieved
evidence to the LLM for 4-class veracity classification.

Architecture:
  AVeriTeC Knowledge Store (pre-scraped web pages, one JSONL file per claim)
        ↓
  Text → word chunks (200 words, 50% stride overlap)
        ↓
  NLTK Pipeline: Lowercase → Punctuation removal → Stopword filtering → Stemming
        ↓
  BM25 Okapi scoring vs. tokenised claim query
        ↓
  Top-k passages → LLM prompt → veracity label (0-3)

Knowledge Store layout (chenxwh/AVeriTeC on HuggingFace):
  <ks_root>/train/{original_claim_idx}.json   ← JSONL, each line: {url, url2text}
  <ks_root>/dev/{original_claim_idx}.json

Index safety note:
  KS files are named by the claim's ORIGINAL position in train.json / dev.json.
  This script resolves original indices by dataset row id or full-record matching,
  never by assuming that claim texts are unique. This matters because AVeriTeC
  contains duplicate claim strings.

Parallel corpus warm-up (pure speed optimisation, zero change to results):
  The dominant cost for a NEW claim is "KS load + chunk + tokenize"
  (CPU + disk bound, not LLM-bound). This step is independent across claims,
  so before the main sequential loop runs, all claims that still need
  processing have their tokenized corpus pre-built across multiple OS
  processes (ProcessPoolExecutor, bypasses the GIL — unlike threads, which
  do NOT help with CPU-bound work). The corpus cache files produced this
  way are byte-for-byte identical to what the original sequential code
  would write; only WHO builds them and WHEN changes, not WHAT is computed.
  Tokenization, chunking, stemming, stopwords, prompt, and the LLM call
  itself are completely untouched. The result cache, content-filter
  handling, and reproducibility of cached results are unaffected.

Dependencies:
  pip install rank-bm25 nltk python-dotenv scikit-learn langchain-openai openai

Usage:
  # Quick smoke-test (10 claims + retrieval diagnostic)
  python src/bm25_baseline.py --sample 10

  # Full official_dev evaluation (parallel warm-up enabled by default)
  python src/bm25_baseline.py --sample -1

  # Choose worker count explicitly (default: min(8, CPU cores))
  python src/bm25_baseline.py --sample -1 --workers 6

  # Disable parallel warm-up (original strictly-sequential behaviour)
  python src/bm25_baseline.py --sample -1 --no-parallel-warmup

  # Ablation: top-k comparison
  python src/bm25_baseline.py --sample -1 --top-k 3
  python src/bm25_baseline.py --sample -1 --top-k 10

  # No-metadata ablation
  python src/bm25_baseline.py --sample -1 --no-metadata
"""

import os
import json
import random
import string
import argparse
import gzip
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
from sklearn.metrics import f1_score, classification_report

from dataset import read_averitec_json
from query_generator import (
    generate_query, load_query_cache, save_query_cache,
    QUERY_CACHE_FILE_DEFAULT, DECOMPOSE_CACHE_FILE_DEFAULT,
)

# Reproducibility (consistent with all other scripts in this project)
random.seed(42)
load_dotenv()

# ── Hyper-parameters ──────────────────────────────────────────────────────────
CHUNK_SIZE = 200   # words per passage chunk
CHUNK_STRIDE = 100   # sliding-window stride (50% overlap)
TOP_K = 5     # passages retrieved per claim
MAX_EVIDENCE_CHARS = 3000  # hard cap on evidence text sent to LLM
RETRIEVAL_CACHE_DEPTH = 10  # supports top-k 3/5/10 ablations
RETRIEVAL_CACHE_VERSION = 1
TOKENIZER_VERSION = "nltk_stopwords_porter_v1"

# ── NLTK Setup ────────────────────────────────────────────────────────────────
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("⏳ Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# ── Label mappings (identical to llm_baseline.py) ─────────────────────────────
LLM_LABEL_MAP = {
    "supported":                          0,
    "refuted":                            1,
    "not enough evidence":                2,
    "conflicting evidence/cherrypicking": 3,
}
TARGET_NAMES = ["Supported (0)", "Refuted (1)", "NEE (2)", "Conflicting (3)"]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert fact-checker. You are given a claim and a set of evidence passages retrieved from the web.

Based ONLY on the provided evidence, determine the claim's veracity.

Output ONLY one of the following four labels, in lowercase, with no additional words, punctuation, or explanation:

supported
refuted
not enough evidence
conflicting evidence/cherrypicking"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Knowledge Store helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_ks_file(ks_root: str, split_ks_name: str, claim_idx: int) -> str:
    """Returns the expected KS file path for a given original claim index."""
    return os.path.join(ks_root, split_ks_name, f"{claim_idx}.json")


def load_ks_file(ks_file: str) -> list[tuple[str, str]]:
    """
    Robust loader for AVeriTeC JSONL knowledge store files.

    Format: one JSON object per line, each with:
      - 'url'      : source URL
      - 'url2text' : list of text tokens (joined into a string)

    Falls back to 'content2' / 'content' / 'text' if 'url2text' is absent.
    Returns an empty list for missing or malformed files.
    """
    if not ks_file or not os.path.isfile(ks_file):
        return []

    docs = []
    try:
        with open(ks_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    url = item.get('url', '')

                    if 'url2text' in item and isinstance(item['url2text'], list):
                        text = " ".join(item['url2text'])
                    else:
                        text = (item.get('content2')
                                or item.get('content')
                                or item.get('text')
                                or '')

                    if isinstance(text, str) and text.strip():
                        docs.append((url, text.strip()))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    return docs


def _load_json_records(json_file: str) -> list[dict]:
    """Loads a JSON dataset file as a list of raw AVeriTeC records."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def _record_key(item: dict) -> str:
    """
    Stable full-record key used for source-index recovery.

    Full-record matching is safer than claim-text matching because several
    AVeriTeC examples intentionally share the same claim string.
    """
    return json.dumps(
        item,
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':'),
    )


def _metadata_key(item: dict) -> tuple:
    """
    Secondary key for recovering source indices if a copied subset differs
    slightly from the original JSON formatting or ancillary fields.
    """
    return (
        item.get('claim', ''),
        item.get('label', ''),
        item.get('speaker') or 'none',
        item.get('reporting_source') or 'none',
        item.get('claim_date') or '',
        item.get('fact_checking_article') or '',
        item.get('original_claim_url') or '',
    )


def _first_unused(indices: list[int], used_indices: set[int]) -> int | None:
    """Returns the first source index that has not already been assigned."""
    for idx in indices:
        if idx not in used_indices:
            used_indices.add(idx)
            return idx
    return None


def build_text_id_to_source_idx(data_file: str,
                                source_json_file: str) -> dict[str, int]:
    """
    Builds a mapping from Text.text_id (claim_0, claim_1, ...) to the original
    source index used by the AVeriTeC Knowledge Store.

    Why this avoids duplicate-claim bugs:
      - official_dev reads directly from dev.json, so claim_i maps to row i.
      - internal_dev may be a subset of train.json, so we recover the original
        train row by matching complete records first, then rich metadata.
      - plain claim-text matching is used only as a last-resort fallback.

    This preserves all 500 evaluation rows; duplicate claim texts are not
    dropped or merged.
    """
    data_records = _load_json_records(data_file)

    if os.path.abspath(data_file) == os.path.abspath(source_json_file):
        return {f"claim_{i}": i for i in range(len(data_records))}

    source_records = _load_json_records(source_json_file)

    exact_to_indices: dict[str, list[int]] = {}
    metadata_to_indices: dict[tuple, list[int]] = {}
    claim_to_indices: dict[str, list[int]] = {}

    for idx, item in enumerate(source_records):
        exact_to_indices.setdefault(_record_key(item), []).append(idx)
        metadata_to_indices.setdefault(_metadata_key(item), []).append(idx)
        claim_to_indices.setdefault(item.get('claim', ''), []).append(idx)

    text_id_to_idx: dict[str, int] = {}
    used_indices: set[int] = set()

    for i, item in enumerate(data_records):
        cid = f"claim_{i}"

        # Prefer explicit source indices if a future preprocessing step stores
        # them in the subset file.
        explicit_idx = (
            item.get('original_idx')
            if item.get('original_idx') is not None
            else item.get('claim_idx')
        )
        if isinstance(explicit_idx, int):
            text_id_to_idx[cid] = explicit_idx
            used_indices.add(explicit_idx)
            continue

        idx = _first_unused(
            exact_to_indices.get(_record_key(item), []), used_indices
        )
        if idx is None:
            idx = _first_unused(
                metadata_to_indices.get(_metadata_key(item), []), used_indices
            )
        if idx is None:
            idx = _first_unused(
                claim_to_indices.get(item.get('claim', ''), []), used_indices
            )

        if idx is not None:
            text_id_to_idx[cid] = idx

    return text_id_to_idx


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BM25 retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               stride: int = CHUNK_STRIDE) -> list[str]:
    """
    Splits text into overlapping word-level chunks (50% stride).

    Overlap ensures that evidence spanning a chunk boundary is not missed.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    for i in range(0, len(words), stride):
        chunk = words[i: i + chunk_size]
        if chunk:
            chunks.append(' '.join(chunk))
    return chunks


def _tokenize(s: str) -> list[str]:
    """
    NLTK-enhanced tokeniser:
      1. Lowercase
      2. Remove punctuation
      3. Filter English stopwords (removes noise tokens like 'the', 'is', 'a')
      4. Porter stemming (normalises 'directed' → 'direct', 'running' → 'run')

    Applied identically to both corpus chunks and the claim query,
    so BM25 term matching operates on the same normalised vocabulary.
    """
    clean = s.lower().translate(str.maketrans('', '', string.punctuation))
    return [STEMMER.stem(w) for w in clean.split() if w not in STOP_WORDS]


def _hash_text(text: str) -> str:
    """Returns a short stable hash for cache filenames."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _safe_cache_path(cache_dir: str, name: str) -> str:
    """Builds a gzip JSON cache path under cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{name}.json.gz")


def _read_json_gz(path: str) -> dict | None:
    """Reads a gzip-compressed JSON cache file, returning None if invalid."""
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, EOFError):
        return None


def _write_json_gz_atomic(path: str, payload: dict) -> None:
    """Writes gzip JSON atomically to avoid corrupt cache files on interruption."""
    tmp_path = f"{path}.tmp"
    with gzip.open(tmp_path, 'wt', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def _write_json_atomic(path: str, payload, indent: int = 2) -> None:
    """
    Writes plain JSON atomically: serialize to <path>.tmp, then os.replace.

    Why this matters for the result cache: a previous incident truncated the
    official_dev BM25 result cache from 500 entries down to 8, because the
    cache is rewritten in-place inside the main loop. If a run starts with an
    empty cache and is interrupted mid-loop (or another run with --sample 10
    is launched against the same cache file), the non-atomic
    `open(path,'w') + json.dump(...)` first truncates the file to zero bytes
    and only then rewrites it — a window in which 16 hours of cached work can
    vanish. os.replace() is atomic on POSIX, so readers always see either
    the old fully-valid file or the new fully-valid file, never an
    in-progress truncation. Reused by both bm25 and gold result-cache writes.
    """
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)
    os.replace(tmp_path, path)


def _file_fingerprint(path: str) -> tuple:
    """
    Returns (size_bytes, mtime_int) for staleness detection, or (None, None)
    if the file does not exist.

    Used to detect when a KS file's on-disk content has changed since a
    corpus cache entry was written for it (e.g. a file that was missing or
    empty when first cached — such as during the train/ folder's incomplete
    zip-extraction — and was later fixed/replaced). Without this check, the
    cache would silently keep returning the OLD (e.g. empty) result forever,
    since claim_idx/chunk_size/split_ks_name alone don't change when only
    the file's content is fixed.

    NOTE: mtime is still stored here for diagnostics, but cache-hit
    comparison only uses size_bytes (the first element). The KS files live
    on an NTFS external drive mounted via a userspace driver, whose mtime is
    unstable across remounts/reads — it drifts even when file content is
    unchanged, which caused spurious 100% cache misses. File size is stable
    and still changes whenever real content changes, so it is a safe key.
    See _fingerprints_match() for the comparison logic.
    """
    try:
        st = os.stat(path)
        return (st.st_size, int(st.st_mtime))
    except OSError:
        return (None, None)


def _fingerprints_match(cached_fp, current_fp) -> bool:
    """
    Decide whether a cached corpus entry is still valid for the current KS
    file. Compares ONLY the size component (index 0), deliberately ignoring
    mtime because the NTFS external drive's mtime is unstable (see
    _file_fingerprint). Robust to cached_fp being a list (from JSON) or a
    tuple, and to missing/None values.

    Special case — KS file deleted to free disk space: if the cache has a
    valid stored fingerprint but the live KS file is gone (current_fp ==
    (None, None) because os.stat raised), we TRUST the cache rather than
    treat it as stale. Otherwise a 10-hour corpus-warm-up would be silently
    overwritten with empty data the next time bm25_baseline.py runs against
    a deleted KS path. Rebuilding from scratch is the explicit alternative
    (delete the corpus cache too, or re-mount the KS drive).
    """
    if cached_fp is None:
        return False
    try:
        cached_size = tuple(cached_fp)[0]
    except (IndexError, TypeError):
        return False
    if cached_size is None:
        return False
    # KS file missing on disk → trust cache (user explicitly freed the raw KS)
    if current_fp is None or tuple(current_fp)[0] is None:
        return True
    current_size = tuple(current_fp)[0]
    return cached_size == current_size


def _corpus_cache_name(split_ks_name: str,
                       claim_idx: int,
                       chunk_size: int) -> str:
    """Cache key for KS chunks + tokenized corpus."""
    return (
        f"corpus_v{RETRIEVAL_CACHE_VERSION}_{TOKENIZER_VERSION}_"
        f"{split_ks_name}_{claim_idx}_chunk{chunk_size}_"
        f"stride{CHUNK_STRIDE}"
    )


def _ranking_cache_name(corpus_name: str,
                        query_text: str,
                        depth: int) -> str:
    """Cache key for BM25 ranking results for a specific query text."""
    return f"rank_{corpus_name}_q{_hash_text(query_text)}_depth{depth}"


def _build_tokenized_corpus(documents: list[tuple[str, str]],
                            chunk_size: int
                            ) -> tuple[list[tuple[str, str]], list[list[str]]]:
    """Builds retrieval chunks and their tokenized representation."""
    corpus: list[tuple[str, str]] = []
    for url, text in documents:
        for chunk in chunk_text(text, chunk_size, CHUNK_STRIDE):
            corpus.append((url, chunk))

    tokenized_corpus = [_tokenize(chunk) for _, chunk in corpus]
    return corpus, tokenized_corpus


def get_or_build_tokenized_corpus(
    claim_idx: int,
    ks_file: str,
    split_ks_name: str,
    chunk_size: int,
    corpus_cache_dir: str | None,
    timings: dict[str, float],
    counters: dict[str, int],
) -> tuple[list[tuple[str, str]], list[list[str]], int]:
    """
    Loads cached KS chunks + tokenized corpus, or builds and caches them.

    The cache is independent of query text and top-k. Different query strategies
    can reuse the same tokenized corpus while still recomputing BM25 scores.
    """
    cache_name = _corpus_cache_name(split_ks_name, claim_idx, chunk_size)
    cache_path = (_safe_cache_path(corpus_cache_dir, cache_name)
                  if corpus_cache_dir else None)
    current_fingerprint = _file_fingerprint(ks_file)

    start = time.perf_counter()
    if cache_path:
        cached = _read_json_gz(cache_path)

        cached_fp = cached.get("ks_fingerprint") if cached else None
        if (cached and cached.get("cache_name") == cache_name
                and _fingerprints_match(cached_fp, current_fingerprint)):

            corpus_raw = cached.get("corpus", [])
            tokenized_corpus = cached.get("tokenized_corpus", [])
            if len(corpus_raw) == len(tokenized_corpus):
                counters["corpus_cache_hits"] += 1
                timings["ks_tokenize"] += time.perf_counter() - start
                corpus = [(url, chunk) for url, chunk in corpus_raw]
                return corpus, tokenized_corpus, cached.get("n_docs", 0)

    documents = load_ks_file(ks_file)
    corpus, tokenized_corpus = _build_tokenized_corpus(documents, chunk_size)
    n_docs = len(documents)
    counters["corpus_cache_misses"] += 1

    if cache_path:
        payload = {
            "cache_name": cache_name,
            "version": RETRIEVAL_CACHE_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "ks_file": ks_file,
            "ks_fingerprint": current_fingerprint,
            "split_ks_name": split_ks_name,
            "claim_idx": claim_idx,
            "chunk_size": chunk_size,
            "stride": CHUNK_STRIDE,
            "n_docs": n_docs,
            "corpus": corpus,
            "tokenized_corpus": tokenized_corpus,
        }
        _write_json_gz_atomic(cache_path, payload)

    timings["ks_tokenize"] += time.perf_counter() - start
    return corpus, tokenized_corpus, n_docs


# ── Parallel corpus warm-up ─────────────────────────────────────────────────
#
# The functions below are a pure speed optimisation layered on top of the
# cache already implemented in get_or_build_tokenized_corpus(). They do not
# introduce a second tokenizer, a second chunker, or a second cache format.
# _corpus_worker() calls the exact same helpers (load_ks_file,
# _build_tokenized_corpus) and writes via the exact same
# _corpus_cache_name() / _write_json_gz_atomic() used by the sequential
# path, so a corpus cache file built in a worker process is indistinguishable
# from one built by the original single-process code.
#
# Must be a top-level function (not a closure/lambda) so it can be pickled
# and sent to worker processes by ProcessPoolExecutor.

def _corpus_worker(task: tuple) -> tuple[int, str, int]:
    """
    Builds (or confirms) the on-disk tokenized-corpus cache for ONE claim.

    Runs inside a separate OS process, so CPU-bound NLTK tokenization and
    BM25 corpus construction for different claims execute truly in
    parallel (unlike threads, which stay serialized by the GIL for
    CPU-bound work).

    Returns only small scalars (claim_idx, hit/miss, n_docs) — the actual
    corpus payload is written straight to the gzip cache file and never
    shipped back through inter-process communication, keeping IPC cheap.
    """
    claim_idx, ks_file, split_ks_name, chunk_size, corpus_cache_dir = task

    cache_name = _corpus_cache_name(split_ks_name, claim_idx, chunk_size)
    cache_path = _safe_cache_path(corpus_cache_dir, cache_name)
    current_fingerprint = _file_fingerprint(ks_file)

    cached = _read_json_gz(cache_path)
    cached_fp = cached.get("ks_fingerprint") if cached else None
    if (cached and cached.get("cache_name") == cache_name
            and _fingerprints_match(cached_fp, current_fingerprint)):

        corpus_raw = cached.get("corpus", [])
        tokenized_corpus = cached.get("tokenized_corpus", [])
        if len(corpus_raw) == len(tokenized_corpus):
            return claim_idx, "hit", cached.get("n_docs", 0)

    documents = load_ks_file(ks_file)
    corpus, tokenized_corpus = _build_tokenized_corpus(documents, chunk_size)
    n_docs = len(documents)

    payload = {
        "cache_name": cache_name,
        "version": RETRIEVAL_CACHE_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "ks_file": ks_file,
        "ks_fingerprint": current_fingerprint,
        "split_ks_name": split_ks_name,
        "claim_idx": claim_idx,
        "chunk_size": chunk_size,
        "stride": CHUNK_STRIDE,
        "n_docs": n_docs,
        "corpus": corpus,
        "tokenized_corpus": tokenized_corpus,
    }
    _write_json_gz_atomic(cache_path, payload)
    return claim_idx, "miss", n_docs


def warm_up_corpus_cache_parallel(
    pending_claims: list[tuple[str, int, str, str]],
    chunk_size: int,
    corpus_cache_dir: str,
    max_workers: int,
) -> dict[str, float | int]:
    """
    Pre-builds the tokenized-corpus cache for every claim in `pending_claims`
    using multiple OS processes, before the main sequential loop runs.

    pending_claims : list of (cid, claim_idx, ks_file, split_ks_name)
        Only claims NOT already present in the result cache need to be
        passed here — claims with a cached LLM answer never reach the
        retrieval step in the main loop, so warming up their corpus would
        be wasted work.

    After this returns, every claim's corpus cache file already exists on
    disk. The unchanged sequential loop that follows will then hit
    "corpus_cache_hits" for all of them via get_or_build_tokenized_corpus(),
    and only has to do BM25 scoring + the LLM call itself.
    """
    tasks = [
        (claim_idx, ks_file, split_ks_name, chunk_size, corpus_cache_dir)
        for (_cid, claim_idx, ks_file, split_ks_name) in pending_claims
        if claim_idx is not None and ks_file is not None
    ]

    if not tasks:
        return {"hits": 0, "misses": 0, "elapsed": 0.0}

    print(f"\n{'─' * 60}")
    print(f"Parallel corpus warm-up: {len(tasks)} claims, "
          f"{max_workers} worker processes")
    print("(builds the same on-disk corpus cache the sequential path "
          "would — tokenizer/chunking unchanged)")
    print(f"{'─' * 60}")

    hits = misses = errors = 0
    done = 0
    total = len(tasks)
    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_corpus_worker, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                _claim_idx, status, _n_docs = future.result()
                if status == "hit":
                    hits += 1
                else:
                    misses += 1
            except Exception as e:
                errors += 1
                claim_idx = futures[future][0]
                print(f"  [warm-up error] claim_idx={claim_idx}: {e}")

            done += 1
            if done % 25 == 0 or done == total:
                elapsed = time.perf_counter() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                print(f"  … {done}/{total} corpora ready "
                      f"({hits} already cached, {misses} newly built, "
                      f"{errors} errors, {elapsed:.1f}s elapsed, "
                      f"{rate:.2f} claims/s)")

    elapsed = time.perf_counter() - start
    print(f"Warm-up complete: {hits} already cached, {misses} newly built, "
          f"{errors} errors, {elapsed:.1f}s total\n")

    return {"hits": hits, "misses": misses, "errors": errors,
            "elapsed": elapsed}


def rank_bm25_from_tokenized(query_text: str,
                             corpus: list[tuple[str, str]],
                             tokenized_corpus: list[list[str]],
                             depth: int) -> list[tuple[float, str, str]]:
    """Ranks cached corpus chunks for a query using the existing BM25 setup."""
    if not corpus or not tokenized_corpus:
        return []

    tokenized_query = _tokenize(query_text)
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)),
                         key=lambda idx: scores[idx],
                         reverse=True)[:depth]

    return [(scores[idx], corpus[idx][0], corpus[idx][1])
            for idx in top_indices]


def retrieve_bm25_cached(
    query_text: str,
    claim_idx: int,
    ks_file: str,
    split_ks_name: str,
    top_k: int,
    chunk_size: int,
    corpus_cache_dir: str | None,
    ranking_cache_dir: str | None,
    ranking_depth: int,
    timings: dict[str, float],
    counters: dict[str, int],
) -> tuple[list[tuple[float, str, str]], int]:
    """
    Cached BM25 retrieval.

    Level 1 cache: KS docs -> chunks -> tokenized corpus, independent of query.
    Level 2 cache: query-specific BM25 ranking, reusable across top-k values.
    """
    corpus, tokenized_corpus, n_docs = get_or_build_tokenized_corpus(
        claim_idx=claim_idx,
        ks_file=ks_file,
        split_ks_name=split_ks_name,
        chunk_size=chunk_size,
        corpus_cache_dir=corpus_cache_dir,
        timings=timings,
        counters=counters,
    )

    depth = max(top_k, ranking_depth)
    corpus_name = _corpus_cache_name(split_ks_name, claim_idx, chunk_size)
    rank_name = _ranking_cache_name(corpus_name, query_text, depth)
    rank_path = (_safe_cache_path(ranking_cache_dir, rank_name)
                 if ranking_cache_dir else None)

    start = time.perf_counter()
    if rank_path:
        cached = _read_json_gz(rank_path)
        if cached and cached.get("cache_name") == rank_name:
            ranked_raw = cached.get("ranked_passages", [])
            if len(ranked_raw) >= top_k:
                counters["ranking_cache_hits"] += 1
                timings["bm25_scoring"] += time.perf_counter() - start
                ranked = [(score, url, passage)
                          for score, url, passage in ranked_raw]
                return ranked[:top_k], n_docs

    ranked = rank_bm25_from_tokenized(
        query_text=query_text,
        corpus=corpus,
        tokenized_corpus=tokenized_corpus,
        depth=depth,
    )
    counters["ranking_cache_misses"] += 1

    if rank_path:
        payload = {
            "cache_name": rank_name,
            "version": RETRIEVAL_CACHE_VERSION,
            "corpus_cache_name": corpus_name,
            "query_hash": _hash_text(query_text),
            "depth": depth,
            "ranked_passages": [(float(score), url, passage)
                                for score, url, passage in ranked],
        }
        _write_json_gz_atomic(rank_path, payload)

    timings["bm25_scoring"] += time.perf_counter() - start
    return ranked[:top_k], n_docs


def retrieve_bm25_multi_query(
    query_list: list[str],
    claim_idx: int,
    ks_file: str,
    split_ks_name: str,
    top_k: int,
    chunk_size: int,
    corpus_cache_dir: str | None,
    ranking_cache_dir: str | None,
    ranking_depth: int,
    timings: dict[str, float],
    counters: dict[str, int],
    per_query_pool: int | None = None,
) -> tuple[list[tuple[float, str, str]], int]:
    """
    Strategy C support: run BM25 retrieval for several sub-questions against
    the same KS, dedupe passages, return final top-k by max BM25 score.

    per_query_pool: how many passages to pull per sub-question before dedup.
        Defaults to top_k (so 3 sub-queries × top_k = 3*top_k candidates
        before dedup). Each sub-question hits its own ranking-cache entry
        keyed by query hash, so subsequent runs are free.

    Dedup key: (url, first 80 chars of passage). The first-80-chars prefix
    is unique enough for our 200-word chunks because two passages from the
    same URL only collide if they're literally the same chunk; the corpus
    chunker uses a 100-word stride so adjacent chunks differ by at least
    half their length. The same passage retrieved by two different
    sub-questions keeps its highest BM25 score across them.
    """
    if not query_list:
        return [], 0

    pool = per_query_pool if per_query_pool is not None else top_k
    seen: dict[tuple[str, str], tuple[float, str, str]] = {}
    n_docs = 0

    for q in query_list:
        passages, this_n_docs = retrieve_bm25_cached(
            query_text=q,
            claim_idx=claim_idx,
            ks_file=ks_file,
            split_ks_name=split_ks_name,
            top_k=pool,
            chunk_size=chunk_size,
            corpus_cache_dir=corpus_cache_dir,
            ranking_cache_dir=ranking_cache_dir,
            ranking_depth=max(ranking_depth, pool),
            timings=timings,
            counters=counters,
        )
        n_docs = max(n_docs, this_n_docs)  # constant per claim_idx
        for score, url, passage in passages:
            key = (url, passage[:80])
            existing = seen.get(key)
            if existing is None or existing[0] < score:
                seen[key] = (score, url, passage)

    merged = sorted(seen.values(), key=lambda x: -x[0])[:top_k]
    return merged, n_docs


def retrieve_bm25(claim_text: str,
                  documents: list[tuple[str, str]],
                  top_k: int = TOP_K,
                  chunk_size: int = CHUNK_SIZE) -> list[tuple[float, str, str]]:
    """
    Retrieves the top-k most relevant passages for a claim using BM25 Okapi.

    Parameters
    ----------
    claim_text : str
        The claim text used as the BM25 query.
    documents : list of (url, text) tuples
        All documents from the claim's KS entry.
    top_k : int
        Number of passages to return.
    chunk_size : int
        Words per chunk when splitting documents into retrieval units.

    Returns
    -------
    List of (bm25_score, url, passage_text) sorted by score descending.
    """
    if not documents:
        return []

    corpus, tokenized_corpus = _build_tokenized_corpus(documents, chunk_size)

    if not corpus:
        return []

    return rank_bm25_from_tokenized(
        query_text=claim_text,
        corpus=corpus,
        tokenized_corpus=tokenized_corpus,
        depth=top_k,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LLM clients (mirrors llm_baseline.py exactly for consistency)
# ═══════════════════════════════════════════════════════════════════════════════

def build_langchain_client():
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0.0,
    )


def call_langchain(client, system_prompt: str, user_text: str):
    from langchain_core.messages import SystemMessage, HumanMessage
    response = client.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ])
    usage = response.response_metadata.get("token_usage", {})
    return (response.content,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0))


def build_native_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def call_native(client, system_prompt: str, user_text: str):
    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ],
        temperature=0.0,
    )
    return (response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Label normalisation & prompt builder
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_llm_output(raw_output: str) -> int | None:
    """
    Maps raw LLM output to a numeric label (0-3), or None if unparseable.
    Keyword order avoids substring collisions ('conflicting' before 'refuted').
    Identical logic to llm_baseline.py for cross-system consistency.
    """
    text = raw_output.strip().lower().strip('."\'* ')
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
    return None


def build_user_input(text,
                     passages: list[tuple[float, str, str]],
                     use_metadata: bool = True) -> str:
    """
    Constructs the LLM user message: claim (+ optional metadata) + evidence.

    Evidence is truncated at MAX_EVIDENCE_CHARS to stay within token limits.
    When no passages are found, the LLM is told explicitly so it can
    reasonably return 'not enough evidence' instead of hallucinating.
    """
    lines = [f"Claim: {text.original_claim}"]

    if use_metadata:
        lines.append(f"Speaker: {text.speaker}")
        lines.append(f"Source:  {text.reporting_source}")

    lines.append("")  # blank separator

    if passages:
        lines.append("Retrieved Evidence:")
        total_chars = 0
        for rank, (score, url, passage) in enumerate(passages, 1):
            entry = f"[{rank}] {url}\n{passage}"
            if total_chars + len(entry) > MAX_EVIDENCE_CHARS:
                break
            lines.append(entry)
            lines.append("")
            total_chars += len(entry)
    else:
        lines.append(
            "Note: No evidence passages were found in the knowledge store "
            "for this claim."
        )

    lines.append(
        "Based on the evidence above, what is the veracity of this claim?"
    )
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_bm25_baseline(
    split_name:         str,
    data_file:          str,
    ks_root:            str,
    split_ks_name:      str,   # "train" or "dev"
    source_idx_file:    str,   # JSON file used to build claim→original_idx map
    backend:            str,
    sample:             int | None,
    use_metadata:       bool,
    top_k:              int,
    chunk_size:         int,
    results_dir:        str,
    use_retrieval_cache: bool = True,
    retrieval_cache_depth: int = RETRIEVAL_CACHE_DEPTH,
    parallel_warmup:    bool = True,
    max_workers:        int | None = None,
    query_strategy:     str = "direct",
    diagnostic:         bool = False,
    retrieval_only:     bool = False,
) -> dict:
    """
    BM25 retrieval + LLM classification with incremental result caching.

    Index safety:
    Both splits resolve KS filenames through a text-id → original-index map.
    The resolver does not assume claim texts are unique, so duplicate claims
    remain separate evaluation rows with their own KS files.

    query_strategy : "direct" (Strategy A, default) uses the raw claim text
        as the BM25 query — this is what every prior run used, so the
        already-completed 500-claim runs remain valid and their cache files
        are reused unchanged. "keyword" (Strategy B) rewrites the claim via
        an LLM into a short search query before retrieval; the claim text
        shown to the LLM in the final verification prompt is unaffected
        either way (see query_generator.py).

    Cache filename:
      results/bm25_baseline_{split}_{mode}_top{k}_cache.json            (Strategy A — unchanged)
      results/bm25_baseline_{split}_{mode}_top{k}_keyword_cache.json    (Strategy B — separate file)
    """
    texts = read_averitec_json(data_file)
    texts = [t for t in texts if t.label is not None]

    mode_suffix = "with_metadata" if use_metadata else "no_metadata"
    mode_label = (f"BM25 top-{top_k} + LLM "
                  f"({'w/ metadata' if use_metadata else 'claim only'})")

    print(f"\n{'=' * 60}")
    print(f"BM25 Baseline: {mode_label}")
    print(f"Split   : {split_name}  |  Backend: {backend}")
    print(f"KS root : {ks_root}")
    print(f"Params  : top_k={top_k}, chunk_size={chunk_size}, "
          f"stride={CHUNK_STRIDE}")
    print(f"{'=' * 60}")

    if sample is not None:
        texts = texts[:sample]
        print(f"  ⚠️  Debug mode: processing first {sample} claims only.")
    print(f"Total claims to process: {len(texts)}")

    # ── Build text-id → original index lookup (BOTH splits) ──────────────────
    # KS filenames use the claim's ORIGINAL position in train.json / dev.json.
    # Text-id based resolution preserves duplicate claim texts instead of
    # collapsing them into a single dict key.
    print("Building text-id → source-index map from: "
          f"{os.path.basename(data_file)} against "
          f"{os.path.basename(source_idx_file)} ...")
    text_id_to_source_idx = build_text_id_to_source_idx(
        data_file, source_idx_file
    )
    print(f"  Resolved {len(text_id_to_source_idx)} data rows.")

    os.makedirs(results_dir, exist_ok=True)

    # Strategy A (default) keeps the EXACT original filename, so the
    # already-completed 500-claim official_dev result cache (16 hours of
    # work) is found and reused unchanged. Strategy B gets its own file —
    # it must never be confused with or overwrite Strategy A's results.
    strategy_suffix = "" if query_strategy == "direct" else f"_{query_strategy}"
    cache_file = os.path.join(
        results_dir,
        f"bm25_baseline_{split_name}_{mode_suffix}_top{top_k}{strategy_suffix}_cache.json"
    )

    cache: dict = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached results from {cache_file}")

    corpus_cache_dir = None
    ranking_cache_dir = None
    if use_retrieval_cache:
        corpus_cache_dir = os.path.join(results_dir, "bm25_corpus_cache")
        ranking_cache_dir = os.path.join(results_dir, "bm25_ranking_cache")
        print("Retrieval cache: enabled "
              f"(depth={max(top_k, retrieval_cache_depth)})")
    else:
        print("Retrieval cache: disabled")

    # ── Parallel corpus warm-up ────────────────────────────────────────────
    # Only claims that are NOT already in the result cache will reach the
    # retrieval step in the main loop below, so only those need their
    # corpus pre-built. This must run BEFORE the LLM client is created and
    # BEFORE the sequential loop starts, so the loop below is unchanged:
    # it will simply find every corpus cache file already warm.
    if use_retrieval_cache and parallel_warmup:
        pending_claims = []
        for text in texts:
            cid = text.text_id
            claim_idx = text_id_to_source_idx.get(cid)
            already_cached = (
                cid in cache and cache[cid].get("claim_idx") == claim_idx
            )
            if not already_cached and claim_idx is not None:
                ks_file = get_ks_file(ks_root, split_ks_name, claim_idx)
                pending_claims.append((cid, claim_idx, ks_file, split_ks_name))

        workers = max_workers or min(8, os.cpu_count() or 4)
        warm_up_corpus_cache_parallel(
            pending_claims=pending_claims,
            chunk_size=chunk_size,
            corpus_cache_dir=corpus_cache_dir,
            max_workers=workers,
        )

    # LLM client is needed for the verification step AND for Strategy
    # B/C query generation. Build it whenever either is required.
    needs_llm = (not retrieval_only) or (query_strategy in ("keyword", "decompose"))
    if needs_llm:
        client = (build_langchain_client() if backend == "langchain"
                  else build_native_client())
        call_fn = call_langchain if backend == "langchain" else call_native
    else:
        client = None
        call_fn = None
        print("LLM backend: skipped (--retrieval-only with Strategy A)")
    if retrieval_only:
        print("Mode: --retrieval-only (writes top_passages cache, skips LLM "
              "verification — feed cache to reprompt_eval.py)")

    # Per-strategy query cache — separate from the result/corpus/ranking
    # caches above, since a claim's LLM-generated query does not depend on
    # top_k, chunk_size, or retrieval method. Each strategy uses its own
    # cache file so they don't collide if both ablations run.
    query_cache = None
    query_cache_path = None
    if query_strategy == "keyword":
        query_cache_path = os.path.join(results_dir, QUERY_CACHE_FILE_DEFAULT)
        query_cache = load_query_cache(query_cache_path)
        print(f"Query strategy: 'keyword' (Strategy B) "
              f"({len(query_cache)} rewritten queries cached)")
    elif query_strategy == "decompose":
        query_cache_path = os.path.join(results_dir, DECOMPOSE_CACHE_FILE_DEFAULT)
        query_cache = load_query_cache(query_cache_path)
        print(f"Query strategy: 'decompose' (Strategy C — UHH-style sub-questions) "
              f"({len(query_cache)} claim decompositions cached)")
    else:
        print("Query strategy: 'direct' (Strategy A — raw claim text)")

    y_true, y_pred = [], []
    new_prompt_tokens, new_completion_tokens = 0, 0
    total_prompt_tokens, total_completion_tokens = 0, 0
    unparseable = new_calls = no_evidence_count = idx_miss_count = 0
    timings = {
        "ks_tokenize": 0.0,
        "bm25_scoring": 0.0,
        "llm_call": 0.0,
    }
    counters = {
        "corpus_cache_hits": 0,
        "corpus_cache_misses": 0,
        "ranking_cache_hits": 0,
        "ranking_cache_misses": 0,
    }

    for i, text in enumerate(texts):
        cid = text.text_id
        claim_idx = text_id_to_source_idx.get(cid)
        ks_file = (get_ks_file(ks_root, split_ks_name, claim_idx)
                   if claim_idx is not None else None)

        cache_is_valid = (cid in cache
                          and cache[cid].get("claim_idx") == claim_idx)

        # Self-healing check: a cached entry that previously found ZERO
        # documents is the single biggest tell of contamination from a
        # source-data problem (e.g. the train/ folder's incomplete zip
        # extraction discovered during this project). Such an entry is
        # cheap to re-verify — re-read just that one KS file and compare —
        # so we never silently keep serving a "no evidence" answer once the
        # underlying data has actually been fixed.
        if cache_is_valid and cache[cid].get("n_docs") == 0 and ks_file:
            live_doc_count = len(load_ks_file(ks_file))
            if live_doc_count > 0:
                cache_is_valid = False
                print(f"  [cache refresh] {cid}: cached n_docs=0 but "
                      f"{live_doc_count} docs now found on disk — "
                      f"source data was likely fixed since this was cached")

        if cache_is_valid:
            entry = cache[cid]

        else:
            if cid in cache and cache[cid].get("claim_idx") != claim_idx:
                print(
                    f"  [cache refresh] {cid}: cached claim_idx="
                    f"{cache[cid].get('claim_idx')} expected={claim_idx}"
                )

            # ── Resolve original KS index via text-id/source mapping ──────
            # Safe for both splits: not affected by label filtering or
            # duplicate claim strings.
            if claim_idx is None:
                idx_miss_count += 1
                print(f"  [warn] {cid}: claim not found in index — "
                      f"KS unavailable. claim='{text.original_claim[:60]}...'")

            # Strategy A returns the claim text unchanged; Strategy B returns
            # one LLM-rewritten keyword query; Strategy C returns a LIST of
            # 2-3 sub-questions (handled by the multi-query path below).
            # Either way, the claim text shown to the LLM in the final
            # verification prompt (build_user_input) is unaffected — only
            # the text used to QUERY the retrieval index changes.
            query_out = generate_query(
                text.original_claim, query_strategy,
                client=client, call_fn=call_fn,
                cache=query_cache, cache_path=query_cache_path,
            )

            if use_retrieval_cache and claim_idx is not None:
                if query_strategy == "decompose":
                    # query_out is a list[str] of sub-questions
                    passages, n_docs = retrieve_bm25_multi_query(
                        query_list=query_out,
                        claim_idx=claim_idx,
                        ks_file=ks_file,
                        split_ks_name=split_ks_name,
                        top_k=top_k,
                        chunk_size=chunk_size,
                        corpus_cache_dir=corpus_cache_dir,
                        ranking_cache_dir=ranking_cache_dir,
                        ranking_depth=retrieval_cache_depth,
                        timings=timings,
                        counters=counters,
                    )
                    query_text = query_out  # store the list for cache record
                else:
                    # Strategy A / B: single-query path (unchanged)
                    query_text = query_out
                    passages, n_docs = retrieve_bm25_cached(
                        query_text=query_text,
                        claim_idx=claim_idx,
                        ks_file=ks_file,
                        split_ks_name=split_ks_name,
                        top_k=top_k,
                        chunk_size=chunk_size,
                        corpus_cache_dir=corpus_cache_dir,
                        ranking_cache_dir=ranking_cache_dir,
                        ranking_depth=retrieval_cache_depth,
                        timings=timings,
                        counters=counters,
                    )
            else:
                # Non-cached path: Strategy C falls back to running all
                # sub-questions on the freshly-loaded documents (rare path,
                # only when --no-retrieval-cache is set).
                start = time.perf_counter()
                documents = load_ks_file(ks_file)
                timings["ks_tokenize"] += time.perf_counter() - start
                n_docs = len(documents)

                start = time.perf_counter()
                if query_strategy == "decompose":
                    seen: dict = {}
                    for sub_q in query_out:
                        for s, u, p in retrieve_bm25(sub_q, documents, top_k, chunk_size):
                            key = (u, p[:80])
                            if key not in seen or seen[key][0] < s:
                                seen[key] = (s, u, p)
                    passages = sorted(seen.values(), key=lambda x: -x[0])[:top_k]
                    query_text = query_out
                else:
                    query_text = query_out
                    passages = retrieve_bm25(
                        query_text, documents, top_k, chunk_size
                    )
                timings["bm25_scoring"] += time.perf_counter() - start

            if not n_docs:
                no_evidence_count += 1
                if no_evidence_count <= 5:   # limit log spam
                    print(f"  [no evidence] {cid} → ks_file={ks_file}")

            # ── Retrieval diagnostic (first new claim only) ───────────────
            if diagnostic and new_calls == 0:
                print(f"\n[Diagnostic] First claim retrieval sample:")
                print(f"  Claim     : {text.original_claim}")
                print(f"  KS file   : {ks_file}")
                print(f"  Docs loaded: {n_docs}")
                for rank, (score, url, passage) in enumerate(passages[:2], 1):
                    print(f"  [{rank}] BM25={score:.2f}  URL={url}")
                    print(f"       {passage[:250]}...")
                print()

            # ── LLM call (skipped under --retrieval-only) ────────────────
            # Retrieval-only mode (Protocol Rule #2: Decouple & Cache):
            # write top_passages to cache and let reprompt_eval.py do the
            # LLM step separately. This lets one expensive BM25 run feed
            # any number of cheap prompt-ablation runs without re-retrieving.
            if retrieval_only:
                entry = {
                    "claim":             text.original_claim,
                    "speaker":           text.speaker,
                    "source":            text.reporting_source,
                    "claim_idx":         claim_idx,
                    "ks_file":           ks_file,
                    "query_strategy":    query_strategy,
                    "query_text":        query_text,
                    "n_docs":            n_docs,
                    "n_passages":        len(passages),
                    "top_passages":      [(round(s, 4), u, p[:300])
                                          for s, u, p in passages],
                    "input_mode":        mode_suffix,
                    "true_label":        text.label,
                    "raw_output":        "",
                    "prompt_tokens":     0,
                    "completion_tokens": 0,
                }
            else:
                user_input = build_user_input(text, passages, use_metadata)

                try:
                    llm_start = time.perf_counter()
                    raw_text, p_tok, c_tok = call_fn(
                        client, SYSTEM_PROMPT, user_input
                    )
                    timings["llm_call"] += time.perf_counter() - llm_start
                    entry = {
                        "claim":             text.original_claim,
                        "speaker":           text.speaker,
                        "source":            text.reporting_source,
                        "claim_idx":         claim_idx,
                        "ks_file":           ks_file,
                        "query_strategy":    query_strategy,
                        "query_text":        query_text,
                        "n_docs":            n_docs,
                        "n_passages":        len(passages),
                        "top_passages":      [(round(s, 4), u, p[:300])
                                              for s, u, p in passages],
                        "input_mode":        mode_suffix,
                        "true_label":        text.label,
                        "raw_output":        raw_text,
                        "prompt_tokens":     p_tok,
                        "completion_tokens": c_tok,
                    }
                except Exception as e:
                    if 'llm_start' in locals():
                        timings["llm_call"] += time.perf_counter() - llm_start
                    err = str(e)
                    if "content_filter" in err or "ResponsibleAI" in err:
                        print(f"  [content_filter] {cid} — skipped")
                        entry = {
                            "claim":             text.original_claim,
                            "claim_idx":         claim_idx,
                            "query_strategy":    query_strategy,
                            "query_text":        query_text,
                            "n_docs":            n_docs,
                            "n_passages":        len(passages),
                            "input_mode":        mode_suffix,
                            "true_label":        text.label,
                            "raw_output":        "CONTENT_FILTERED",
                            "prompt_tokens":     0,
                            "completion_tokens": 0,
                        }
                    else:
                        raise

            cache[cid] = entry
            new_calls += 1
            new_prompt_tokens += entry.get("prompt_tokens", 0)
            new_completion_tokens += entry.get("completion_tokens", 0)

            # Incremental cache save — partial progress is never lost.
            # Atomic write: any reader (including a re-launched run) sees the
            # OLD fully-valid file until the rename completes, then the NEW
            # fully-valid file. See _write_json_atomic() for why this matters.
            _write_json_atomic(cache_file, cache)

        # ── Parse label ───────────────────────────────────────────────────
        pred = normalize_llm_output(entry["raw_output"])
        if pred is None:
            unparseable += 1
            print(f"  [unparseable] {cid}  raw={entry['raw_output']!r}")
            pred = -1   # treated as wrong for all classes

        y_true.append(entry["true_label"])
        y_pred.append(pred)
        total_prompt_tokens += entry.get("prompt_tokens", 0)
        total_completion_tokens += entry.get("completion_tokens", 0)

        if (i + 1) % 50 == 0:
            print(f"  … {i+1}/{len(texts)} processed "
                  f"({new_calls} new API calls so far)")

    # ── Evaluation ────────────────────────────────────────────────────────────
    # labels=[0,1,2,3]: enforces true 4-class Macro-F1 even when a class is
    # absent from the evaluated subset (e.g. Conflicting at ~6% prevalence).
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=[0, 1, 2, 3])

    print(f"\n{'─' * 60}")
    print(f"Results: {split_name}  |  {mode_label}")
    print(f"{'─' * 60}")
    print(f"Macro-F1                   : {macro_f1:.4f}")
    print(f"Claims with no KS evidence : {no_evidence_count}/{len(texts)}")
    print(f"KS index misses            : {idx_miss_count}/{len(texts)}")
    print(f"Unparseable outputs        : {unparseable}/{len(texts)}")
    print(f"New API calls this run     : {new_calls}")
    print(f"Token usage (new)          : "
          f"{new_prompt_tokens} prompt + {new_completion_tokens} completion")
    print(f"Token usage (all cached)   : "
          f"{total_prompt_tokens} prompt + "
          f"{total_completion_tokens} completion")
    print("Timing diagnostics (new work this run):")
    print(f"  KS load + chunk + tokenize : {timings['ks_tokenize']:.2f}s "
          f"(cache hits={counters['corpus_cache_hits']}, "
          f"misses={counters['corpus_cache_misses']})")
    print(f"  BM25 scoring + ranking    : {timings['bm25_scoring']:.2f}s "
          f"(rank hits={counters['ranking_cache_hits']}, "
          f"misses={counters['ranking_cache_misses']})")
    print(f"  LLM calls                 : {timings['llm_call']:.2f}s")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3],
        target_names=TARGET_NAMES, zero_division=0))

    return {
        "split":                  split_name,
        "mode":                   mode_label,
        "n_claims":               len(texts),
        "macro_f1":               macro_f1,
        "no_evidence_count":      no_evidence_count,
        "idx_miss_count":         idx_miss_count,
        "unparseable":            unparseable,
        "new_calls":              new_calls,
        "new_prompt_tokens":      new_prompt_tokens,
        "new_completion_tokens":  new_completion_tokens,
        "total_prompt_tokens":    total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "timings":                timings,
        "retrieval_cache_counters": counters,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "BM25 Baseline (NLTK Enhanced) — Retrieval-augmented 4-class "
            "fact verification on AVeriTeC.\n\n"
            "Retrieves evidence from the Knowledge Store using BM25 Okapi "
            "(with stopword removal + Porter stemming),\nthen classifies each "
            "claim with the LLM grounded on the retrieved passages.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split", choices=["internal_dev", "official_dev", "both"],
        default="official_dev",
        help="Split(s) to evaluate on (default: official_dev)")
    parser.add_argument(
        "--backend", choices=["langchain", "native"], default="langchain",
        help="LLM API backend (default: langchain)")
    parser.add_argument(
        "--sample", type=int, default=10,
        help="Claims to process; -1 = full split (default: 10 for debug)")
    parser.add_argument(
        "--no-metadata", action="store_true",
        help="Exclude speaker/source metadata (mirrors llm_baseline --no-metadata)")
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"BM25 passages to retrieve per claim (default: {TOP_K})")
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Words per passage chunk (default: {CHUNK_SIZE})")
    parser.add_argument(
        "--no-retrieval-cache", action="store_true",
        help="Disable BM25 corpus/ranking caches and use the original retrieval path")
    parser.add_argument(
        "--retrieval-cache-depth", type=int, default=RETRIEVAL_CACHE_DEPTH,
        help=("Number of ranked BM25 passages to cache per query "
              f"(default: {RETRIEVAL_CACHE_DEPTH})"))
    parser.add_argument(
        "--no-parallel-warmup", action="store_true",
        help=("Disable the multiprocess corpus warm-up phase and build "
              "each claim's corpus cache sequentially (slower, but useful "
              "for isolating timing or debugging worker-process issues)"))
    parser.add_argument(
        "--workers", type=int, default=None,
        help=("Number of worker processes for corpus warm-up "
              "(default: min(8, CPU core count))"))
    parser.add_argument(
        "--ks-root",
        default=(
            "/Volumes/Seagate Backup Plus Drive/"
            "AVeriTeC_knowledge_store/data_store/knowledge_store"
        ),
        help="Root directory of the AVeriTeC Knowledge Store")
    parser.add_argument(
        "--query-strategy", choices=["direct", "keyword", "decompose"],
        default="direct",
        help=("'direct' = Strategy A, raw claim text as BM25 query "
              "(default — matches all prior runs). 'keyword' = Strategy B, "
              "LLM-rewritten keyword query. 'decompose' = Strategy C, "
              "UHH-style claim decomposition into 2-3 sub-questions, each "
              "retrieved independently; results deduped + scored by max "
              "BM25 score. See query_generator.py for prompts."))
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help=("Skip the LLM verification step; only persist top_passages "
              "to the result cache. Feed the cache to reprompt_eval.py to "
              "score with any prompt (apples-to-apples ablation). Aligns "
              "with Protocol Rule #2 (Decouple & Cache)."))
    parser.add_argument(
        "--data-file", default=None,
        help=("Override the claims file to evaluate (e.g. a stratified "
              "subset produced by sample_stratified.py). The KS sub-dir "
              "and source-index file for claim_idx resolution are still "
              "determined by --split (internal_dev -> train.json+train/, "
              "official_dev -> dev.json+dev/). Not compatible with "
              "--split both."))
    parser.add_argument(
        "--split-label", default=None,
        help=("Label used for result-cache filenames and the printed "
              "summary, so a --data-file run never collides with the "
              "full split's cache. Defaults to the value of --split, or "
              "to the --data-file basename if --data-file is given."))
    parser.add_argument(
        "--diagnostic", action="store_true",
        help="Print retrieved passages for the first claim to verify KS quality")
    args = parser.parse_args()

    if args.data_file and args.split == "both":
        parser.error("--data-file cannot be combined with --split both")

    sample = None if args.sample == -1 else args.sample
    use_metadata = not args.no_metadata

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    results_dir = os.path.join(project_dir, 'results')

    # split name → (data_file, ks_sub_dir, source_idx_file)
    #
    # source_idx_file is the ORIGINAL JSON file for that split.
    # It is used to build the text-id → original-idx lookup so that KS
    # filenames are always resolved correctly, independent of filtering and
    # duplicate claim strings.
    #
    #   internal_dev: claims come from train.json  → KS sub-dir = "train"
    #   official_dev: claims come from dev.json    → KS sub-dir = "dev"
    split_configs = {
        "internal_dev": (
            os.path.join(data_dir, "internal_dev.json"),
            "train",
            os.path.join(data_dir, "train.json"),   # source for idx lookup
        ),
        "official_dev": (
            os.path.join(data_dir, "dev.json"),
            "dev",
            os.path.join(data_dir, "dev.json"),     # source for idx lookup
        ),
    }

    splits_to_run = (
        ["internal_dev", "official_dev"] if args.split == "both"
        else [args.split]
    )

    summary = []
    for split_name in splits_to_run:
        data_file, split_ks_name, source_idx_file = split_configs[split_name]

        # --data-file overrides WHICH claims are evaluated; split_ks_name and
        # source_idx_file stay tied to --split, since claim_idx resolution
        # must still be checked against the correct original file
        # (train.json for an internal_dev-derived subset, dev.json for an
        # official_dev-derived subset) regardless of which claims subset
        # is being scored this run.
        if args.data_file:
            data_file = args.data_file

        run_label = (
            args.split_label
            or (os.path.splitext(os.path.basename(args.data_file))[0]
                if args.data_file else split_name)
        )

        result = run_bm25_baseline(
            split_name=run_label,
            data_file=data_file,
            ks_root=args.ks_root,
            split_ks_name=split_ks_name,
            source_idx_file=source_idx_file,
            backend=args.backend,
            sample=sample,
            use_metadata=use_metadata,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            results_dir=results_dir,
            use_retrieval_cache=not args.no_retrieval_cache,
            retrieval_cache_depth=args.retrieval_cache_depth,
            parallel_warmup=not args.no_parallel_warmup,
            max_workers=args.workers,
            query_strategy=args.query_strategy,
            diagnostic=args.diagnostic,
            retrieval_only=args.retrieval_only,
        )
        summary.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for r in summary:
        print(
            f"{r['split']:<15} | {r['mode']}\n"
            f"  n={r['n_claims']}  Macro-F1={r['macro_f1']:.4f}  "
            f"no_evidence={r['no_evidence_count']}  "
            f"idx_miss={r['idx_miss_count']}  "
            f"unparseable={r['unparseable']}  "
            f"new_calls={r['new_calls']}\n"
            f"  tokens (new)    : {r['new_prompt_tokens']} in / "
            f"{r['new_completion_tokens']} out\n"
            f"  tokens (cached) : {r['total_prompt_tokens']} in / "
            f"{r['total_completion_tokens']} out\n"
        )
