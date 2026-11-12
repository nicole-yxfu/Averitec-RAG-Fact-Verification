"""
Unit tests for the selective-prediction (risk-coverage) code.

These numbers go straight into the README's headline figure, and every
failure mode here is silent — a wrong sort order or an off-by-one in the
coverage index produces a *prettier* curve, not a crash. So the tests pin
the maths against hand-computed values:

  1. risk_coverage()  — ordering, accuracy@k, AURC, tie handling
  2. rule_based_scores() — the fitted-nothing baseline
  3. featurize()      — missing signals must stay a third category, not be
                        silently folded into "bad"
  4. crossfit_scores() — no claim may be scored by a model that saw it

Runs with either:
    python tests/test_selective.py
    pytest tests/test_selective.py
"""
import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(TEST_DIR), "src")
sys.path.insert(0, SRC_DIR)

import numpy as np

from selective_prediction import (crossfit_scores, featurize,
                                  risk_coverage, rule_based_scores)


# ── 1. risk_coverage ────────────────────────────────────────────────────
def test_perfect_confidence_puts_all_correct_first():
    # Scores rank the two correct claims above the two wrong ones.
    correct = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    rc = risk_coverage(scores, correct)
    # acc@1 = 1/1, acc@2 = 2/2, acc@3 = 2/3, acc@4 = 2/4
    assert list(np.round(rc["acc_at_k"], 6)) == [1.0, 1.0, round(2 / 3, 6), 0.5]


def test_full_coverage_accuracy_equals_overall_accuracy():
    # The last point of the curve must be the plain accuracy, whatever the
    # score does — otherwise the baseline line in the figure is wrong.
    rng = np.random.default_rng(7)
    for _ in range(5):
        correct = rng.integers(0, 2, 40)
        scores = rng.random(40)
        rc = risk_coverage(scores, correct)
        assert abs(rc["acc_at_k"][-1] - correct.mean()) < 1e-12


def test_aurc_matches_hand_computation():
    correct = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    # risks: 0, 0, 1/3, 1/2  ->  mean = 5/24
    assert abs(risk_coverage(scores, correct)["aurc"] - 5 / 24) < 1e-12


def test_aurc_of_a_useless_score_equals_the_error_rate():
    # A constant score carries no information, so after tie-breaking the
    # expected AURC is the overall error rate — the figure's dashed line.
    correct = np.concatenate([np.ones(300), np.zeros(200)]).astype(int)
    scores = np.zeros(500)
    aurcs = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(500)
        aurcs.append(risk_coverage(scores[perm], correct[perm])["aurc"])
    assert abs(float(np.mean(aurcs)) - 0.4) < 0.03


def test_perfect_score_beats_inverted_score():
    correct = np.array([1, 1, 1, 0, 0, 0])
    good = np.array([6, 5, 4, 3, 2, 1], dtype=float)
    assert (risk_coverage(good, correct)["aurc"]
            < risk_coverage(-good, correct)["aurc"])


def test_ties_are_not_broken_by_input_order():
    # Claim ids come in dataset order and the rule-based score has only
    # three distinct values, so position-based tie-breaking would let
    # dataset ordering pose as confidence. All-tied scores with all the
    # correct claims placed first must NOT yield a perfect curve.
    correct = np.concatenate([np.ones(50), np.zeros(50)]).astype(int)
    rc = risk_coverage(np.zeros(100), correct)
    assert rc["acc_at_k"][49] < 0.95


def test_single_claim():
    assert risk_coverage(np.array([0.5]), np.array([1]))["aurc"] == 0.0
    assert risk_coverage(np.array([0.5]), np.array([0]))["aurc"] == 1.0


# ── 2. rule_based_scores ────────────────────────────────────────────────
def _row(faithful=None, agree=None, pred=0):
    return {"faithful": faithful, "agree": agree, "pred": pred}


def test_rule_based_adds_one_point_per_positive_signal():
    rows = [_row(1, 1), _row(1, 0), _row(0, 1), _row(0, 0), _row(None, None)]
    assert list(rule_based_scores(rows)) == [2.0, 1.0, 1.0, 0.0, 0.0]


def test_rule_based_uses_no_labels():
    # Same features, opposite correctness -> identical scores. If this ever
    # fails, the "nothing fitted" baseline is leaking.
    a = [{"faithful": 1, "agree": 1, "pred": 0, "correct": 1}]
    b = [{"faithful": 1, "agree": 1, "pred": 0, "correct": 0}]
    assert list(rule_based_scores(a)) == list(rule_based_scores(b))


# ── 3. featurize ────────────────────────────────────────────────────────
def test_unknown_signal_is_its_own_category():
    # A content-filtered judge means "no information", which must not look
    # the same as "the judge said ungrounded".
    unknown = featurize([_row(None, None, 0)])[0]
    ungrounded = featurize([_row(0, 0, 0)])[0]
    assert list(unknown[:4]) == [0.0, 0.0, 0.0, 0.0]
    assert list(ungrounded[:4]) == [0.0, 1.0, 0.0, 1.0]


def test_predicted_class_is_one_hot():
    for cls in range(4):
        feats = featurize([_row(1, 1, cls)])[0]
        assert list(feats[4:]) == [1.0 if i == cls else 0.0 for i in range(4)]


def test_featurize_shape():
    assert featurize([_row(1, 1, 0), _row(0, 0, 3)]).shape == (2, 8)


# ── 4. crossfit_scores ──────────────────────────────────────────────────
def test_crossfit_scores_every_claim_exactly_once():
    rng = np.random.default_rng(3)
    X = rng.random((60, 8))
    y = np.array([0, 1] * 30)
    s = crossfit_scores(X, y, n_splits=5, seed=0)
    assert s.shape == (60,)
    assert np.all((s >= 0) & (s <= 1))
    assert len(np.unique(s)) > 1          # not all left at the zero default


def test_crossfit_cannot_memorize_pure_noise():
    # Labels independent of features: an in-sample fit would score high,
    # but out-of-fold scores must not separate the classes. This is the
    # test that would catch the leak the cross-fitting exists to prevent.
    rng = np.random.default_rng(11)
    X = rng.random((200, 8))
    y = rng.integers(0, 2, 200)
    s = crossfit_scores(X, y, n_splits=5, seed=0)
    rc = risk_coverage(s, y)
    # Random ordering would give AURC ~= error rate; allow slack for n=200.
    assert rc["aurc"] > (1 - y.mean()) - 0.12


def test_crossfit_recovers_a_real_signal():
    # Feature 0 perfectly predicts correctness; out-of-fold scores must
    # rank it, or cross-fitting is destroying signal, not just leakage.
    #
    # The bar is the ORACLE AURC — ranking by the labels themselves — not
    # an arbitrary constant. A perfect ranker cannot drive AURC to zero:
    # once every correct claim has been answered, each further claim is an
    # error, so for n = 200 at 50% accuracy even the oracle sits near 0.16.
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 200)
    X = np.zeros((200, 8))
    X[:, 0] = y
    X[:, 1:] = rng.random((200, 7))
    oracle = risk_coverage(y.astype(float), y)["aurc"]
    got = risk_coverage(crossfit_scores(X, y, 5, 0), y)["aurc"]
    assert got <= oracle + 1e-9
    assert got < (1 - y.mean()) - 0.2      # and far better than random


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
