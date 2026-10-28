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
