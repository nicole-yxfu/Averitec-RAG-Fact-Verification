"""
The static demo (demo/) shows numbers to people who will never open the
code, so it must not drift from the evaluation. These tests rebuild the demo
data from the committed artifacts and check:

  1. the committed demo/examples.js is exactly what build_demo.py produces
     now (someone edited the data or the evaluation without rebuilding);
  2. every example's label ("answered", "caught", ...) agrees with its own
     confidence, threshold, verdict and gold label;
  3. the headline numbers equal the published selective-prediction report.

Runs with either:
    python tests/test_demo.py
    pytest tests/test_demo.py
"""
import json
import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TEST_DIR)
sys.path.insert(0, os.path.join(REPO, "src"))

from build_demo import HEADER, build_payload, render  # noqa: E402

_PAYLOAD = None


def payload():
    global _PAYLOAD
    if _PAYLOAD is None:
        _PAYLOAD = build_payload()
    return _PAYLOAD


def committed():
    with open(os.path.join(REPO, "demo", "examples.js"), encoding="utf-8") as f:
        text = f.read()
    assert text.startswith(HEADER)
    return text, json.loads(text[len(HEADER):].rstrip().rstrip(";"))


def test_committed_demo_data_is_up_to_date():
    text, _ = committed()
    assert text == render(payload()), "run: python src/build_demo.py"


def test_outcomes_agree_with_their_own_numbers():
    p = payload()
    for e in p["examples"]:
        assert e["answered"] == (e["confidence"] >= p["threshold"]), e["id"]
        right = e["verdict"] == e["gold"]
        expected = ("correct" if e["answered"] and right else
                    "miss" if e["answered"] else
                    "caught" if not right else "cautious")
        assert e["outcome"] == expected, e["id"]
        assert len(e["evidence"]) == 5 and e["justification"], e["id"]


def test_headline_numbers_match_published_report():
    with open(os.path.join(REPO, "results", "selective_prediction_dev500.json")) as f:
        pub = json.load(f)["summaries"][0]
    p = payload()
    assert p["accuracy_all"] == pub["overall_accuracy"]
    assert p["aurc"] == pub["aurc"]
    point = next(x for x in pub["points"] if abs(x["coverage"] - p["coverage"]) < 1e-9)
    assert p["accuracy_answered"] == point["accuracy"]


def test_examples_are_not_only_successes():
    # The page claims the examples include mistakes; keep that true.
    outcomes = {e["outcome"] for e in payload()["examples"]}
    assert {"miss", "caught"} <= outcomes


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
