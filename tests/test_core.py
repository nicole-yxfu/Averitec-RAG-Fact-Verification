"""
Minimal unit tests for the AVeriTeC advanced-approaches pipeline.

Covers the three pure, high-traffic components whose correctness every reported
number depends on:
  1. normalize_llm_output()  — maps raw LLM text to a 0-3 label (or -1)
  2. _write_json_atomic()    — crash-safe cache writes (tmp + os.replace)
  3. load_fewshot()          — strict schema validation of few-shot files

Runs with either:
    python tests/test_core.py        # no pytest needed
    pytest tests/test_core.py
"""
import json
import os
import sys
import tempfile

# ── make src/ importable ────────────────────────────────────────────────
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(TEST_DIR), "src")
sys.path.insert(0, SRC_DIR)

# bm25_baseline builds a stopword set at import time; ensure the NLTK data
# is present so importing the pipeline never fails on a clean machine.
import nltk
for pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from reprompt_eval import normalize_llm_output, load_fewshot
from bm25_baseline import _write_json_atomic


# ── 1. normalize_llm_output ─────────────────────────────────────────────
def test_normalize_exact_labels():
    assert normalize_llm_output("supported") == 0
    assert normalize_llm_output("refuted") == 1
    assert normalize_llm_output("not enough evidence") == 2
    assert normalize_llm_output("conflicting evidence/cherrypicking") == 3


def test_normalize_case_and_whitespace():
    assert normalize_llm_output("  REFUTED  ") == 1
    assert normalize_llm_output("Supported.") == 0
    assert normalize_llm_output('"not enough evidence"') == 2
    assert normalize_llm_output("*Conflicting Evidence/Cherrypicking*") == 3


def test_normalize_substring_fallback():
    # verbose LLM output should still resolve via substring matching
    assert normalize_llm_output("The claim is refuted by the evidence.") == 1
    assert normalize_llm_output("This is conflicting and cherrypicked") == 3
    assert normalize_llm_output("there is not enough evidence here") == 2


def test_normalize_unparseable_returns_minus_one():
    assert normalize_llm_output("") == -1
    assert normalize_llm_output("banana") == -1
    assert normalize_llm_output("CONTENT_FILTERED") == -1


# ── 2. _write_json_atomic ───────────────────────────────────────────────
def test_atomic_write_roundtrip():
    payload = {"claim_0": {"true_label": 1, "raw_output": "refuted"}}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cache.json")
        _write_json_atomic(path, payload)
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == payload


def test_atomic_write_overwrite_leaves_no_tmp():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cache.json")
        _write_json_atomic(path, {"a": 1})
        _write_json_atomic(path, {"a": 2, "b": 3})
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"a": 2, "b": 3}
        # no leftover *.tmp sibling from the atomic swap
        assert [f for f in os.listdir(d) if f.endswith(".tmp")] == []


# ── 3. load_fewshot schema validation ───────────────────────────────────
def _write(tmpdir, obj):
    path = os.path.join(tmpdir, "fs.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return path


def test_load_fewshot_valid():
    good = [{"claim": "c", "evidence_text": "e", "label": "refuted"}]
    with tempfile.TemporaryDirectory() as d:
        assert load_fewshot(_write(d, good)) == good


def test_load_fewshot_rejects_bad_input():
    cases = [
        [],                                                  # empty list
        [{"claim": "c", "evidence_text": "e"}],              # missing label
        [{"claim": "c", "label": "refuted"}],                # missing evidence
        [{"claim": "c", "evidence_text": "e", "label": "x"}],  # invalid label
    ]
    with tempfile.TemporaryDirectory() as d:
        for bad in cases:
            try:
                load_fewshot(_write(d, bad))
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for: {bad}")


# ── 4. ranking cache must not outlive its corpus ────────────────────────
# Replays the June 2026 incident: ranking caches written against one corpus
# build kept being served after the corpus was rebuilt under the same name.
def _ks(path, docs):
    import json as _json
    with open(path, "w", encoding="utf-8") as f:
        for url, text in docs:
            f.write(_json.dumps({"url": url, "url2text": text.split()}) + "\n")


def _retrieve(ks_file, corpus_dir, rank_dir, query):
    from collections import defaultdict
    from bm25_baseline import retrieve_bm25_cached
    counters = defaultdict(int)
    ranked, _ = retrieve_bm25_cached(
        query_text=query, claim_idx=0, ks_file=ks_file, split_ks_name="dev",
        top_k=2, chunk_size=200, corpus_cache_dir=corpus_dir,
        ranking_cache_dir=rank_dir, ranking_depth=2,
        timings=defaultdict(float), counters=counters)
    return [u for _, u, _ in ranked], counters


def test_ranking_cache_is_invalidated_when_corpus_is_rebuilt():
    with tempfile.TemporaryDirectory() as d:
        ks = os.path.join(d, "0.json")
        corpus_dir, rank_dir = os.path.join(d, "c"), os.path.join(d, "r")
        os.makedirs(corpus_dir), os.makedirs(rank_dir)
        filler = [(f"u{i}", f"unrelated filler text number {i}") for i in range(5)]

        _ks(ks, [("old", "volcano eruption ash cloud")] + filler)
        first, _ = _retrieve(ks, corpus_dir, rank_dir, "volcano eruption")
        assert first[0] == "old"
        again, c = _retrieve(ks, corpus_dir, rank_dir, "volcano eruption")
        assert again == first and c["ranking_cache_hits"] == 1

        # Rebuild: different content, same corpus NAME. The stale ranking
        # file is still on disk under the same name. (The new text differs
        # in LENGTH on purpose: the corpus cache compares KS file size only,
        # because the NTFS drive's mtime is unstable — see _file_fingerprint.)
        _ks(ks, [("new", "volcano eruption lava flow from the summit")] + filler)
        after, c = _retrieve(ks, corpus_dir, rank_dir, "volcano eruption")
        assert c["corpus_cache_misses"] == 1          # corpus was rebuilt
        assert c["ranking_cache_misses"] == 1         # and the ranking was NOT reused
        assert after[0] == "new"


def test_unstamped_legacy_ranking_cache_is_not_trusted():
    import gzip as _gzip, json as _json
    with tempfile.TemporaryDirectory() as d:
        ks = os.path.join(d, "0.json")
        corpus_dir, rank_dir = os.path.join(d, "c"), os.path.join(d, "r")
        os.makedirs(corpus_dir), os.makedirs(rank_dir)
        _ks(ks, [("real", "volcano eruption")] +
            [(f"u{i}", f"filler {i}") for i in range(5)])
        _retrieve(ks, corpus_dir, rank_dir, "volcano eruption")
        # Forge a pre-fix entry: right name, no stamp, wrong answer.
        (path,) = [os.path.join(rank_dir, f) for f in os.listdir(rank_dir)]
        with _gzip.open(path, "rt", encoding="utf-8") as f:
            payload = _json.load(f)
        payload.pop("corpus_stamp")
        payload["ranked_passages"] = [[99.0, "forged", "x"], [98.0, "forged2", "y"]]
        with _gzip.open(path, "wt", encoding="utf-8") as f:
            _json.dump(payload, f)
        got, c = _retrieve(ks, corpus_dir, rank_dir, "volcano eruption")
        assert "forged" not in got and c["ranking_cache_misses"] == 1


# ── 5. a read failure must never become an overwrite ────────────────────
# Replays 2026-11-13: an unreadable corpus cache plus an unmounted KS drive
# turned into 479 corpus caches replaced with empty payloads.
def _corpus(ks_file, corpus_dir):
    from collections import defaultdict
    from bm25_baseline import get_or_build_tokenized_corpus
    return get_or_build_tokenized_corpus(
        claim_idx=0, ks_file=ks_file, split_ks_name="dev", chunk_size=200,
        corpus_cache_dir=corpus_dir, timings=defaultdict(float),
        counters=defaultdict(int))


def test_unreadable_cache_with_missing_ks_raises_and_is_untouched():
    from bm25_baseline import _corpus_cache_name, _safe_cache_path
    with tempfile.TemporaryDirectory() as d:
        corpus_dir = os.path.join(d, "c")
        os.makedirs(corpus_dir)
        ks = os.path.join(d, "0.json")
        _ks(ks, [("u", "volcano eruption ash")] * 3)
        _corpus(ks, corpus_dir)                       # build a real cache
        os.remove(ks)                                 # drive unmounted
        path = _safe_cache_path(corpus_dir, _corpus_cache_name("dev", 0, 200))
        with open(path, "rb") as f:
            good = f.read()
        with open(path, "wb") as f:                   # simulate a failed read
            f.write(b"\x1f\x8b not really gzip")
        damaged = open(path, "rb").read()
        try:
            _corpus(ks, corpus_dir)
        except RuntimeError as e:
            assert "Refusing to rebuild" in str(e)
        else:
            raise AssertionError("unreadable cache was silently rebuilt")
        assert open(path, "rb").read() == damaged     # not overwritten
        assert good != damaged


def test_missing_ks_and_no_cache_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        corpus_dir = os.path.join(d, "c")
        os.makedirs(corpus_dir)
        corpus, tok, n = _corpus(os.path.join(d, "absent.json"), corpus_dir)
        assert (corpus, tok, n) == ([], [], 0)
        assert os.listdir(corpus_dir) == []           # no empty cache written


def test_present_but_empty_ks_is_still_cached():
    # The guard must not over-reach: a KS file that exists and is empty is
    # a real, cacheable fact (5 dev claims have no KS documents at all).
    with tempfile.TemporaryDirectory() as d:
        corpus_dir = os.path.join(d, "c")
        os.makedirs(corpus_dir)
        ks = os.path.join(d, "0.json")
        open(ks, "w").close()
        _corpus(ks, corpus_dir)
        assert len(os.listdir(corpus_dir)) == 1


# ── plain-python runner (no pytest required) ────────────────────────────
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
