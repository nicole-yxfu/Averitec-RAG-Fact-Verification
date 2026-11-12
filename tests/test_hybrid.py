"""
Unit tests for the Week-4 hybrid-retrieval and comparison code.

Covers the parts where a silent bug would corrupt a reported number rather
than crash:
  1. rrf_fuse()              — fusion maths, determinism, tie-breaking
  2. rank_bm25_pool()        — lexical ranking, and the all-stopword pool
                               that makes BM25Okapi divide by zero
  3. rerank_cross_encoder()  — only the head is re-ordered; the tail and the
                               candidate set are preserved
  4. mcnemar()               — the significance test, against hand-computed
                               exact binomial p-values
  5. to_gold_int()           — gold labels arrive as ints OR dataset strings

No network and no model downloads: the cross-encoder is replaced by a stub,
since what is under test is the re-ranking bookkeeping, not the model.

Runs with either:
    python tests/test_hybrid.py        # no pytest needed
    pytest tests/test_hybrid.py
"""
import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(TEST_DIR), "src")
sys.path.insert(0, SRC_DIR)

import nltk
for pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords"
                       else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from hybrid_retrieval import (RRF_K, assert_cross_encoder_works,
                              rank_bm25_pool, rank_engine, rrf_fuse,
                              rerank_cross_encoder)
from compare_runs import mcnemar, to_gold_int


# ── 1. rrf_fuse ─────────────────────────────────────────────────────────
def test_rrf_single_ranking_is_identity():
    assert rrf_fuse([[2, 0, 1]]) == [2, 0, 1]


def test_rrf_unanimous_agreement_preserved():
    assert rrf_fuse([[0, 1, 2], [0, 1, 2], [0, 1, 2]]) == [0, 1, 2]


def test_rrf_scores_match_formula():
    # Doc 1 is 1st in A and 3rd in B; doc 0 is 2nd in both.
    #   doc1 = 1/61 + 1/63 = 0.032271
    #   doc0 = 1/62 + 1/62 = 0.032258   -> doc1 edges ahead
    a, b = [1, 0, 2], [2, 0, 1]
    assert 1 / (RRF_K + 1) + 1 / (RRF_K + 3) > 2 / (RRF_K + 2)
    assert rrf_fuse([a, b])[0] == 1


def test_rrf_rewards_broad_consensus_over_one_top_hit():
    # Doc 0 is 2nd in all three rankers; docs 9/8/7 are each 1st in exactly
    # one. Consensus must beat a single top hit:
    #   doc0 = 3/62 = 0.04839   vs   doc9 = 1/61 = 0.01639
    fused = rrf_fuse([[9, 0, 1], [8, 0, 1], [7, 0, 1]])
    assert fused[0] == 0
    assert fused[1] == 1          # 3rd everywhere, 3/63, still beats the 1sts


def test_rrf_a_unanimous_second_place_beats_a_split_first():
    # The converse guard: being 1st in TWO of three rankers is enough to
    # beat being 2nd in all three, so RRF is not merely a rank-average that
    # always favours breadth.
    #   doc1 = 1/63 + 2/61 = 0.04866   vs   doc0 = 3/62 = 0.04839
    assert rrf_fuse([[9, 0, 1], [1, 0, 9], [1, 0, 9]])[0] == 1


def test_rrf_ties_break_on_pool_position():
    # Mirror-image rankings make docs 0 and 2 exactly tied
    # (1/61 + 1/63 each); the lower pool index must come first so the
    # output is deterministic run to run. Note doc 1, 2nd in both, scores
    # 2/62 = 0.032258, just BELOW the tied pair's 0.032266 — reciprocal
    # ranks are convex, so the extremes beat the middle.
    assert rrf_fuse([[0, 1, 2], [2, 1, 0]]) == [0, 2, 1]


def test_rrf_keeps_docs_missing_from_some_rankings():
    fused = rrf_fuse([[0, 1], [2]])
    assert sorted(fused) == [0, 1, 2]


def test_rrf_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


# ── 2. rank_bm25_pool ───────────────────────────────────────────────────
def test_bm25_ranks_term_overlap_first():
    passages = ["the weather in Berlin is mild",
                "quarterly earnings beat expectations",
                "Berlin weather forecast mild and dry"]
    order = rank_bm25_pool("Berlin weather is mild", passages)
    assert order[0] in (0, 2)          # both share the query terms
    assert order[-1] == 1              # the finance passage shares none
    assert sorted(order) == [0, 1, 2]  # a ranking, not a filter


def test_bm25_all_stopword_pool_does_not_crash():
    # _tokenize strips stopwords and punctuation, so these passages reduce
    # to empty token lists. BM25Okapi would divide by a zero average
    # document length; the guard must return the pool order instead.
    assert rank_bm25_pool("the and of", ["the and of", "a an the"]) == [0, 1]


def test_bm25_stemming_matches_inflections():
    # _tokenize Porter-stems, so 'directed' must match 'directing'.
    # The pool is padded to 6 distinct passages on purpose: see
    # test_bm25_tiny_pool_has_no_lexical_signal for why 2 would not work.
    passages = ["unrelated financial report",
                "she was directing the film",
                "quarterly earnings beat expectations",
                "the weather forecast for tomorrow",
                "a recipe for sourdough bread",
                "local election results announced"]
    assert rank_bm25_pool("who directed the film", passages)[0] == 1


def test_bm25_tiny_pool_has_no_lexical_signal():
    # Documents the small-pool degeneracy rather than papering over it:
    # rank_bm25's Okapi idf is log(N-n+0.5) - log(n+0.5), which is exactly
    # 0 at N=2, n=1, so EVERY passage scores 0 and the ranking falls back
    # to pool order even though passage 1 is a perfect lexical match.
    # Real pools here are 10-20 hits, where idf is well behaved.
    passages = ["unrelated financial report", "she was directing the film"]
    assert rank_bm25_pool("who directed the film", passages) == [0, 1]


def test_bm25_empty_pool():
    assert rank_bm25_pool("anything", []) == []


# ── 3. rank_engine / rerank_cross_encoder ───────────────────────────────
def test_rank_engine_is_pool_order():
    assert rank_engine([{}, {}, {}]) == [0, 1, 2]


class _StubCE:
    """Scores a pair by the passage's integer content, so the expected
    re-ordering is known exactly without loading a transformer."""

    def predict(self, pairs):
        return [float(p[1]) for p in pairs]


def test_cross_encoder_reorders_only_the_head():
    passages = [str(i) for i in range(6)]          # "0".."5"
    fused = [5, 4, 3, 2, 1, 0]
    out = rerank_cross_encoder(_StubCE(), "q", passages, fused, depth=3)
    # head [5,4,3] re-sorted by descending stub score -> unchanged
    # tail [2,1,0] must be untouched
    assert out == [5, 4, 3, 2, 1, 0]

    out = rerank_cross_encoder(_StubCE(), "q", passages, [3, 5, 4, 0, 1, 2],
                               depth=3)
    assert out[:3] == [5, 4, 3]       # head re-ranked by the model
    assert out[3:] == [0, 1, 2]       # tail keeps its fused order


def test_cross_encoder_preserves_the_candidate_set():
    passages = [str(i) for i in range(6)]
    fused = [2, 0, 5, 1, 4, 3]
    out = rerank_cross_encoder(_StubCE(), "q", passages, fused, depth=4)
    assert sorted(out) == sorted(fused)
    assert len(out) == len(fused)


def test_cross_encoder_depth_beyond_pool_reranks_everything():
    passages = [str(i) for i in range(4)]
    out = rerank_cross_encoder(_StubCE(), "q", passages, [0, 1, 2, 3],
                               depth=99)
    assert out == [3, 2, 1, 0]


def test_cross_encoder_empty_order():
    assert rerank_cross_encoder(_StubCE(), "q", [], [], depth=5) == []


class _NaNCE:
    """Reproduces the real ms-marco-MiniLM-L-6-v2 failure in this
    environment: finite embeddings, NaN logits."""

    def predict(self, pairs):
        return [float("nan")] * len(pairs)


class _ConstantCE:
    """A model that runs but discriminates nothing — the other way a
    re-ranker becomes the identity without raising."""

    def predict(self, pairs):
        return [1.0] * len(pairs)


def test_cross_encoder_nan_raises_instead_of_silently_doing_nothing():
    # NaN compares False against everything, so an unguarded sorted() would
    # return the input order and the variant would look like a clean null
    # result. It must raise instead.
    try:
        rerank_cross_encoder(_NaNCE(), "q", ["a", "b", "c"], [0, 1, 2],
                             depth=3)
    except RuntimeError as e:
        assert "NaN" in str(e)
        return
    raise AssertionError("expected RuntimeError on NaN scores")


def test_cross_encoder_self_test_accepts_a_working_model():
    class Good:
        def predict(self, pairs):
            # Rank by whether the passage mentions the director.
            return [10.0 if "Spielberg" in p[1] else -10.0 for p in pairs]

    assert_cross_encoder_works(Good()) is None


def test_cross_encoder_self_test_rejects_nan_and_constant_models():
    for model, why in ((_NaNCE(), "NaN"), (_ConstantCE(), "constant")):
        try:
            assert_cross_encoder_works(model)
        except RuntimeError:
            continue
        raise AssertionError(f"self-test should reject a {why} model")


def test_cross_encoder_self_test_rejects_an_inverted_model():
    class Inverted:
        def predict(self, pairs):
            return [-10.0 if "Spielberg" in p[1] else 10.0 for p in pairs]

    try:
        assert_cross_encoder_works(Inverted())
    except RuntimeError as e:
        assert "irrelevant" in str(e)
        return
    raise AssertionError("self-test should reject an inverted model")


# ── 4. mcnemar ──────────────────────────────────────────────────────────
def _run(pairs):
    """pairs: list of (gold, pred) -> {claim_i: (gold, pred)}"""
    return {f"c{i}": p for i, p in enumerate(pairs)}


def test_mcnemar_identical_runs_are_not_significant():
    a = _run([(0, 0), (1, 1), (2, 3)])
    m = mcnemar(a, dict(a))
    assert m["n_discordant"] == 0
    assert m["p_value"] == 1.0
    assert m["both_right"] == 2 and m["both_wrong"] == 1


def test_mcnemar_counts_discordant_pairs_in_the_right_direction():
    # c0: only A right.  c1: only B right.  c2: both right.  c3: both wrong.
    a = _run([(0, 0), (1, 2), (3, 3), (0, 1)])
    b = _run([(0, 1), (1, 1), (3, 3), (0, 2)])
    m = mcnemar(a, b)
    assert (m["only_a_right"], m["only_b_right"]) == (1, 1)
    assert (m["both_right"], m["both_wrong"]) == (1, 1)
    assert m["p_value"] == 1.0            # 1 vs 1 is a fair coin


def test_mcnemar_exact_p_values():
    # 6 discordant claims, all favouring B: exact two-sided p = 2*(1/2)^6.
    a = _run([(0, 1)] * 6)
    b = _run([(0, 0)] * 6)
    m = mcnemar(a, b)
    assert (m["only_a_right"], m["only_b_right"]) == (0, 6)
    assert abs(m["p_value"] - 2 * 0.5 ** 6) < 1e-12

    # 5 vs 1 -> p = 2 * P(X<=1) = 2 * (1+6)/2^6
    a = _run([(0, 1)] * 5 + [(0, 0)])
    b = _run([(0, 0)] * 5 + [(0, 1)])
    m = mcnemar(a, b)
    assert abs(m["p_value"] - 2 * 7 / 64) < 1e-12


def test_mcnemar_ignores_claims_absent_from_one_run():
    # Pairing requires the same claims; an unmatched claim must not count.
    a = _run([(0, 0), (1, 2)])
    b = {"c0": (0, 1)}
    m = mcnemar(a, b)
    assert m["n_common"] == 1
    assert (m["only_a_right"], m["only_b_right"]) == (1, 0)


def test_mcnemar_unparseable_counts_as_wrong():
    # pred == -1 never equals a gold label, so it must land in the
    # 'wrong' column rather than being silently dropped.
    a = _run([(0, 0)])
    b = _run([(0, -1)])
    m = mcnemar(a, b)
    assert (m["only_a_right"], m["only_b_right"]) == (1, 0)


# ── 5. to_gold_int ──────────────────────────────────────────────────────
def test_to_gold_int_accepts_ints_and_dataset_strings():
    assert to_gold_int(2) == 2
    assert to_gold_int("Supported") == 0
    assert to_gold_int("Refuted") == 1
    assert to_gold_int("Not Enough Evidence") == 2
    assert to_gold_int("Conflicting Evidence/Cherrypicking") == 3


def test_to_gold_int_rejects_unknown_labels():
    assert to_gold_int("supported") == -1      # case-sensitive by design
    assert to_gold_int(None) == -1
    assert to_gold_int("nonsense") == -1


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
