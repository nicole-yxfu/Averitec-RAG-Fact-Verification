"""
Unit tests for the FastAPI service layer (src/service.py).

All tests run fully offline — the open-web retrieval and the LLM call are
monkeypatched, so no network or Azure credentials are needed. Covers:
  1. _split_passage()   — parsing stored 'Title: .. / Snippet: ..' passages
  2. VerifyRequest      — Pydantic input validation (min length, top_k range)
  3. verify_claim()     — end-to-end wiring, verdict mapping, cost arithmetic
  4. health()           — liveness response shape

Runs with either:
    python tests/test_service.py     # no pytest needed
    pytest tests/test_service.py
"""
import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(TEST_DIR), "src")
sys.path.insert(0, SRC_DIR)

# bm25_baseline (imported transitively by service) builds a stopword set at
# import time; ensure NLTK data is present on a clean machine.
import nltk
for pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from pydantic import ValidationError

import service


# ── 1. _split_passage ────────────────────────────────────────────────────
def test_split_passage_title_snippet():
    title, snippet = service._split_passage("Title: Foo Bar\nSnippet: the body here")
    assert title == "Foo Bar"
    assert snippet == "the body here"


def test_split_passage_without_title():
    title, snippet = service._split_passage("plain text, no prefix")
    assert title is None
    assert snippet == "plain text, no prefix"


# ── 2. VerifyRequest validation ──────────────────────────────────────────
def test_request_accepts_valid():
    req = service.VerifyRequest(claim="The sky is blue on a clear day.")
    assert req.top_k == 5  # default


def test_request_rejects_bad_input():
    bad_cases = [
        {"claim": "x"},                                   # too short (<3)
        {"claim": "A perfectly fine claim.", "top_k": 0},   # top_k < 1
        {"claim": "A perfectly fine claim.", "top_k": 99},  # top_k > 10
    ]
    for bad in bad_cases:
        try:
            service.VerifyRequest(**bad)
        except ValidationError:
            continue
        raise AssertionError(f"expected ValidationError for: {bad}")


# ── 3. verify_claim (retrieval + LLM monkeypatched) ──────────────────────
def test_verify_claim_mocked_pipeline():
    orig_ddg, orig_call = service.ddg_search_cached, service.call_langchain
    try:
        # One fake open-web hit; hits_to_passages (real) packs it into the
        # 'Title: ..\nSnippet: ..' shape the service then parses back out.
        service.ddg_search_cached = lambda q, d, top_k=5: [
            {"title": "Eiffel height", "snippet": "324 m tall",
             "url": "https://example.com/eiffel"}]
        service.call_langchain = lambda client, system, user: ("supported", 500, 1)

        resp = service.verify_claim(client=object(),
                                    claim="The Eiffel Tower is 324 m tall.",
                                    top_k=3)

        assert resp.verdict == "supported"
        assert resp.verdict_label == "Supported"
        assert resp.verdict_id == 0
        assert resp.n_evidence == 1
        assert resp.evidence[0].url == "https://example.com/eiffel"
        assert resp.evidence[0].title == "Eiffel height"
        assert resp.evidence[0].snippet == "324 m tall"
        assert resp.usage.prompt_tokens == 500
        # cost = 500/1e6 * 0.40 + 1/1e6 * 1.60
        expected = round(500 / 1e6 * service.COST_PER_1M_INPUT
                         + 1 / 1e6 * service.COST_PER_1M_OUTPUT, 6)
        assert resp.usage.cost_usd == expected
        assert resp.latency.total_ms == round(
            resp.latency.retrieval_ms + resp.latency.llm_ms, 1)
    finally:
        service.ddg_search_cached, service.call_langchain = orig_ddg, orig_call


def test_verify_claim_unparseable_verdict():
    orig_ddg, orig_call = service.ddg_search_cached, service.call_langchain
    try:
        service.ddg_search_cached = lambda q, d, top_k=5: []
        service.call_langchain = lambda client, system, user: ("banana", 10, 1)
        resp = service.verify_claim(client=object(), claim="A vague claim.", top_k=5)
        assert resp.verdict_id is None
        assert resp.verdict == "unparseable"
        assert resp.n_evidence == 0  # no evidence retrieved
    finally:
        service.ddg_search_cached, service.call_langchain = orig_ddg, orig_call


# ── 4. health ────────────────────────────────────────────────────────────
def test_health_shape():
    h = service.health()
    assert h.status == "ok"
    assert h.model  # non-empty
    assert "open-web" in h.evidence_source


# ── plain-python runner (no pytest required) ─────────────────────────────
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
