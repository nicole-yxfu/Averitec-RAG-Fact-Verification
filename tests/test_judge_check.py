"""
Tests for the faithfulness-judge check (build_judge_check.py,
score_judge_check.py).

The two properties that matter and fail silently:
  1. Blindness — the labelling page must never be able to see the judge's
     answer or the gold label.
  2. The re-weighting — the sample is 50/50 by judge label but the real
     split is ~75/25, so population numbers must use the real weights.

Runs with either:
    python tests/test_judge_check.py
    pytest tests/test_judge_check.py
"""
import json
import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TEST_DIR)
sys.path.insert(0, os.path.join(REPO, "src"))

from score_judge_check import cohen_kappa, score, wilson  # noqa: E402


def _meta(n_each=2, pop_f=300, pop_u=100):
    key = {}
    for i in range(n_each):
        key[f"f{i}"] = {"judge_faithful": 1, "verdict": "Refuted"}
        key[f"u{i}"] = {"judge_faithful": 0, "verdict": "Supported"}
    return {"key": key, "population": {"faithful": pop_f, "unfaithful": pop_u}}


def _ans(mapping):
    return {cid: {"answer": a} for cid, a in mapping.items()}


def test_wilson_interval():
    lo, hi = wilson(25, 25)
    assert abs(lo - 0.8668) < 1e-3 and hi == 1.0
    lo, hi = wilson(0, 25)
    assert lo == 0.0 and abs(hi - 0.1332) < 1e-3
    assert wilson(0, 0) is None


def test_kappa_extremes():
    assert cohen_kappa([(1, 1), (0, 0), (1, 1), (0, 0)]) == 1.0
    assert cohen_kappa([(1, 0), (0, 1), (1, 0), (0, 1)]) == -1.0


def test_perfect_agreement():
    m = _meta()
    r = score(_ans({"f0": "faithful", "f1": "faithful",
                    "u0": "unfaithful", "u1": "unfaithful"}), m)
    assert r["judge_says_faithful"]["rate"] == 1.0
    assert r["judge_says_unfaithful"]["rate"] == 1.0
    assert r["population_agreement"] == 1.0
    assert abs(r["human_corrected_faithfulness_rate"] - 0.75) < 1e-12


def test_population_reweighting_uses_real_split_not_sample_split():
    # Person agrees with every FAITHFUL call and none of the UNFAITHFUL ones.
    m = _meta(pop_f=300, pop_u=100)
    r = score(_ans({"f0": "faithful", "f1": "faithful",
                    "u0": "faithful", "u1": "faithful"}), m)
    assert r["sample_agreement"] == 0.5          # what a naive average says
    assert abs(r["population_agreement"] - 0.75) < 1e-12   # 0.75*1 + 0.25*0
    assert r["human_corrected_faithfulness_rate"] == 1.0
    assert r["disagreements"] == {"judge_too_lenient": 0, "judge_too_strict": 2}


def test_unsure_and_missing_are_excluded_and_counted():
    m = _meta()
    r = score(_ans({"f0": "faithful", "f1": "unsure", "u0": "unfaithful"}), m)
    assert r["n_scored"] == 2 and r["n_unsure"] == 1 and r["n_missing"] == 1   # u1


def test_labelling_page_data_is_blind_and_matches_key():
    with open(os.path.join(REPO, "judge_check", "items.js"), encoding="utf-8") as f:
        text = f.read()
    items = json.loads(text[text.index("["):text.rindex(";")])
    for it in items:
        assert set(it) == {"n", "id", "claim", "verdict", "rule", "evidence"}, it.keys()
    with open(os.path.join(REPO, "judge_check", "key.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert {it["id"] for it in items} == set(meta["key"])
    labels = [k["judge_faithful"] for k in meta["key"].values()]
    assert labels.count(1) == labels.count(0) == meta["per_stratum"]


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
