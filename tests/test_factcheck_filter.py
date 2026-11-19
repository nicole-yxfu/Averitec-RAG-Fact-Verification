"""
Tests for filter_factcheck.py, the leakage control for open-web evidence.

What matters: fact-check pages are removed, ordinary news pages are kept,
order is preserved, the input cache is never modified, and a claim that
loses every passage ends up with an empty list (which build_user_input turns
into an explicit "no evidence" note) rather than disappearing.

Runs with either:
    python tests/test_factcheck_filter.py
    pytest tests/test_factcheck_filter.py
"""
import copy
import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TEST_DIR)
sys.path.insert(0, os.path.join(REPO, "src"))

from filter_factcheck import filter_cache, is_factcheck_url  # noqa: E402


def test_known_factcheck_sites_match():
    for url in ["https://www.politifact.com/factchecks/2020/x/",
                "https://www.snopes.com/fact-check/x/",
                "https://www.factcheck.org/2021/01/x/",
                "https://factcheck.afp.com/x",
                "https://www.reuters.com/fact-check/x",
                "https://www.usatoday.com/story/news/factcheck/2020/x/",
                "https://leadstories.com/hoax-alert/2020/x.html"]:
        assert is_factcheck_url(url), url


def test_ordinary_news_is_kept():
    for url in ["https://www.reuters.com/world/us/x",
                "https://www.bbc.com/news/x",
                "https://en.wikipedia.org/wiki/X",
                ""]:
        assert not is_factcheck_url(url), url


def _cache():
    return {
        "claim_0": {"claim": "c0", "top_passages": [
            [0.0, "https://www.bbc.com/news/a", "a"],
            [0.0, "https://www.politifact.com/x", "b"],
            [0.0, "https://en.wikipedia.org/wiki/C", "c"]]},
        "claim_1": {"claim": "c1", "top_passages": [
            [0.0, "https://www.snopes.com/fact-check/x", "d"]]},
        "claim_2": {"claim": "c2", "top_passages": [
            [0.0, "https://www.bbc.com/news/e", "e"]]},
    }


def test_filter_removes_keeps_order_and_counts():
    out, stats = filter_cache(_cache())
    assert [p[2] for p in out["claim_0"]["top_passages"]] == ["a", "c"]
    assert out["claim_0"]["factcheck_removed"] == 1
    assert out["claim_0"]["n_passages"] == 2
    assert stats["claims_with_factcheck_hit"] == 2
    assert stats["passages_before"] == 5 and stats["passages_after"] == 3


def test_claim_left_empty_is_kept_with_empty_list():
    out, stats = filter_cache(_cache())
    assert "claim_1" in out and out["claim_1"]["top_passages"] == []
    assert stats["claims_left_without_evidence"] == 1


def test_input_not_modified():
    c = _cache()
    before = copy.deepcopy(c)
    filter_cache(c)
    assert c == before


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
