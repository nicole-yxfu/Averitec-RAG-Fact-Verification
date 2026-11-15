"""
Unit tests for the run-artifact contract, the verdict scorer and the CI gate.

Three things are pinned here, because each fails silently:

  1. runs.passage_id — if it ever depends on rank, score or run, the same
     passage stops being recognisable across runs, and cross-run comparisons
     (like the byte-identical retrieval of the two BM25 prompt runs) break
     without any error.
  2. eval_verdict — the label parser's substring ORDER decides how verbose
     replies are scored. Changing it re-labels historical runs.
  3. ci_gate — every breakage check must actually fire on a broken run.

One block is a regression test against the committed run directories: the
published dev500 Macro-F1 scores must re-derive exactly from `runs/`. If
someone edits the parser or the metric, this is what catches it.

Runs with either:
    python tests/test_runs.py
    pytest tests/test_runs.py
"""
import math
import os
import sys
import tempfile

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

import numpy as np

from runs import Run, load_run, passage_id, write_metrics
from eval_verdict import (UNPARSEABLE, gold_to_int, normalize_llm_output,
                          score_predictions)
from export_run import coerce_scalar, split_title


# ── 1. passage identity ─────────────────────────────────────────────────
def test_passage_id_ignores_rank_score_and_run():
    a = Run(run_dir="unused-a")
    b = Run(run_dir="unused-b")
    ids_a = a.add_retrieved("c1", [(9.0, "u1", "alpha"), (3.0, "u2", "beta")])
    ids_b = b.add_retrieved("c1", [(0.1, "u2", "beta"), (0.0, "u1", "alpha")])
    assert ids_a == list(reversed(ids_b))


def test_passage_id_is_sensitive_to_text_and_url():
    base = passage_id("https://x.org/a", "text")
    assert passage_id("https://x.org/b", "text") != base
    assert passage_id("https://x.org/a", "Text") != base


def test_passage_id_strips_only_surrounding_whitespace():
    assert passage_id(" u ", "  t \n") == passage_id("u", "t")


def test_url_text_boundary_cannot_collide():
    # Without a separator, ("ab", "c") and ("a", "bc") would hash alike.
    assert passage_id("ab", "c") != passage_id("a", "bc")


def test_duplicate_passage_stored_once():
    r = Run(run_dir="unused")
    r.add_retrieved("c1", [(1.0, "u", "same")])
    r.add_retrieved("c2", [(1.0, "u", "same")])
    assert len(r.passages) == 1 and len(r.retrieved) == 2


# ── 2. round trip ───────────────────────────────────────────────────────
def test_run_write_load_round_trip():
    with tempfile.TemporaryDirectory() as d:
        r = Run(run_dir=d, config={"retriever": "bm25", "top_k": 5})
        pids = r.add_retrieved("c1", [(2.0, "u1", "p1"), (1.0, "u2", "p2")])
        r.add_prediction("c1", "claim", "Refuted", "refuted", pids)
        r.add_system("c1", latency_ms=12.5, status="ok")
        r.write()

        back = load_run(d)
        assert back.config["retriever"] == "bm25"
        assert back.config["top_k"] == 5
        assert back.config["schema_version"] == 1
        assert back.ranking("c1") == pids
        assert back.rankings() == {"c1": pids}
        assert back.predictions[0]["used_passage_ids"] == pids
        assert back.passages[pids[0]]["text"] == "p1"


def test_write_metrics_merges_sections():
    with tempfile.TemporaryDirectory() as d:
        write_metrics(d, "verdict", {"macro_f1": 0.5})
        write_metrics(d, "retrieval", {"mrr": 0.3})
        import json
        with open(os.path.join(d, "metrics.json")) as f:
            m = json.load(f)
        assert m["verdict"]["macro_f1"] == 0.5
        assert m["retrieval"]["mrr"] == 0.3


# ── 3. eval_verdict ─────────────────────────────────────────────────────
def test_parser_most_specific_label_wins():
    assert normalize_llm_output("Not enough evidence to call it refuted.") == 2
    assert normalize_llm_output("conflicting evidence, partly supported") == 3
    assert normalize_llm_output("**Refuted**") == 1
    assert normalize_llm_output("Conflicting Evidence/Cherrypicking") == 3
    assert normalize_llm_output("I cannot say") == UNPARSEABLE
    assert normalize_llm_output("") == UNPARSEABLE


def test_gold_accepts_string_and_legacy_int_but_not_bool():
    assert gold_to_int("Not Enough Evidence") == 2
    assert gold_to_int(1) == 1
    assert gold_to_int(7) == UNPARSEABLE
    assert gold_to_int(True) == UNPARSEABLE   # bool is an int subclass
    assert gold_to_int(None) == UNPARSEABLE


def _pred(cid, gold, raw):
    return {"claim_id": cid, "gold_label": gold, "raw_output": raw}


def test_perfect_predictions_score_one():
    rows = [_pred("a", "Supported", "supported"),
            _pred("b", "Refuted", "refuted"),
            _pred("c", "Not Enough Evidence", "not enough evidence"),
            _pred("d", "Conflicting Evidence/Cherrypicking", "conflicting")]
    m = score_predictions(rows, n_boot=50)
    assert m["macro_f1"] == 1.0 and m["accuracy"] == 1.0
    assert m["confusion_matrix"]["counts"] == np.eye(4, dtype=int).tolist()


def test_filtered_and_unparseable_are_counted_and_scored_wrong():
    rows = [_pred("a", "Refuted", "refuted"),
            _pred("b", "Refuted", "CONTENT_FILTERED"),
            _pred("c", "Refuted", "no idea")]
    m = score_predictions(rows, n_boot=50)
    assert m["n_content_filtered"] == 1
    assert m["filtered_claim_ids"] == ["b"]
    assert m["n_unparseable"] == 2          # the refusal is also unparseable
    assert abs(m["accuracy"] - 1 / 3) < 1e-12


def test_rows_without_gold_are_dropped_not_scored():
    rows = [_pred("a", "Refuted", "refuted"), _pred("b", None, "refuted")]
    m = score_predictions(rows, n_boot=50)
    assert m["n"] == 1 and m["n_dropped_no_gold"] == 1


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    names = ["Supported", "Refuted", "Not Enough Evidence",
             "Conflicting Evidence/Cherrypicking"]
    replies = ["supported", "refuted", "not enough evidence", "conflicting"]
    rows = []
    for i in range(300):
        g = int(rng.integers(0, 4))
        p = g if rng.random() < 0.6 else int(rng.integers(0, 4))
        rows.append(_pred(f"c{i}", names[g], replies[p]))
    m = score_predictions(rows, n_boot=300)
    ci = m["macro_f1_ci95"]
    assert ci["lo"] < m["macro_f1"] < ci["hi"]
    assert ci["hi"] - ci["lo"] < 0.25


# ── 4. export helpers ───────────────────────────────────────────────────
def test_coerce_scalar():
    assert coerce_scalar("5") == 5 and isinstance(coerce_scalar("5"), int)
    assert coerce_scalar("0.5") == 0.5
    assert coerce_scalar("true") is True
    assert coerce_scalar("gpt-4.1-mini") == "gpt-4.1-mini"


def test_rehydrate_swaps_previews_for_full_text():
    from export_run import rehydrate
    cached = [(2.0, "u1", "The volcano"), (1.0, "u2", "Ash fell")]
    full = {"matches_published_top5": True,
            "ranked": [[2.0, "u1", "The volcano erupted at dawn."],
                       [1.0, "u2", "Ash fell on the town."]]}
    out = rehydrate("c", cached, full)
    assert [t for _, _, t in out] == ["The volcano erupted at dawn.",
                                      "Ash fell on the town."]
    assert passage_id("u1", out[0][2]) != passage_id("u1", cached[0][2])


def _expect_exit(fn):
    try:
        fn()
    except SystemExit:
        return
    raise AssertionError("expected the rehydration guard to refuse")


def test_rehydrate_refuses_a_ranking_that_does_not_match():
    from export_run import rehydrate
    cached = [(2.0, "u1", "The volcano")]
    _expect_exit(lambda: rehydrate("c", cached, {
        "matches_published_top5": False, "problems": ["rank1: url"],
        "ranked": [[2.0, "u1", "The volcano erupted."]]}))
    _expect_exit(lambda: rehydrate("c", cached, {
        "matches_published_top5": True,
        "ranked": [[2.0, "u1", "A different passage entirely."]]}))
    _expect_exit(lambda: rehydrate("c", cached, None))


def test_passages_seen_mirrors_build_user_input_break():
    from rehydrate_bm25_topk import passages_seen
    p = [(1.0, "u", "x" * 1200)] * 5          # entry = "[r] u\n" + 1200 chars
    assert passages_seen(p, 3000) == 2         # third would cross 3000
    short = [(1.0, "u", "x" * 300)] * 5
    assert passages_seen(short, 3000) == 5
    # It BREAKS, it does not skip: a long passage hides shorter ones after it.
    mixed = [(1.0, "u", "x" * 100), (1.0, "u", "x" * 2950), (1.0, "u", "x" * 10)]
    assert passages_seen(mixed, 3000) == 1


def test_split_title_leaves_text_unchanged():
    text = "Title: Foo\nSnippet: bar"
    title, body = split_title(text)
    assert title == "Foo" and body == text
    assert split_title("plain passage") == (None, "plain passage")


# ── 5. regression: published numbers re-derive from runs/ ───────────────
PUBLISHED = {
    "bm25_dev500_plain": 0.3644,
    "bm25_dev500_standard": 0.3241,
    "ddg_dev500_plain": 0.4495,
    "gold_dev500_oracle": 0.5633,
}


def test_committed_runs_reproduce_published_macro_f1():
    checked = 0
    for run_id, expected in PUBLISHED.items():
        run_dir = os.path.join(REPO_DIR, "runs", run_id)
        if not os.path.isdir(run_dir):
            continue
        m = score_predictions(load_run(run_dir).predictions, n_boot=10)
        assert round(m["macro_f1"], 4) == expected, (run_id, m["macro_f1"])
        checked += 1
    assert checked == len(PUBLISHED), "a committed run directory is missing"


def test_prompt_pair_shares_retrieval_byte_for_byte():
    # The point of the decoupled format: same retrieval, two prompts,
    # 0.3241 vs 0.3644. If these ever diverge, the comparison is confounded.
    a = os.path.join(REPO_DIR, "runs", "bm25_dev500_plain", "retrieved.jsonl")
    b = os.path.join(REPO_DIR, "runs", "bm25_dev500_standard", "retrieved.jsonl")
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read() == fb.read()


# ── 6. ci_gate: every check must actually fire ──────────────────────────
from ci_gate import DEFAULT_LIMITS, check  # noqa: E402

_GOLDS = (["Refuted"] * 12 + ["Supported"] * 8 + ["Not Enough Evidence"] * 5
          + ["Conflicting Evidence/Cherrypicking"] * 5)
_REPLY = {"Refuted": "refuted", "Supported": "supported",
          "Not Enough Evidence": "not enough evidence",
          "Conflicting Evidence/Cherrypicking": "conflicting"}
_IDS = [f"g{i}" for i in range(30)]


def _gate_run(d: str, replies: list[str]) -> str:
    r = Run(run_dir=d)
    for cid, gold, reply in zip(_IDS, _GOLDS, replies):
        r.add_prediction(cid, "claim", gold, reply, [])
    r.write()
    return d


def _baseline_replies() -> list[str]:
    # 20/30 right: the first 20 claims correct, the rest called Refuted.
    return [_REPLY[g] if i < 20 else "refuted" for i, g in enumerate(_GOLDS)]


def _gate(cand_replies: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
        return check(_gate_run(c, cand_replies),
                     _gate_run(b, _baseline_replies()), _IDS, DEFAULT_LIMITS)


def _failed(res: dict) -> set[str]:
    return {c["check"] for c in res["checks"] if not c["passed"]}


def test_gate_passes_an_equal_candidate_but_flags_it_identical():
    res = _gate(_baseline_replies())
    assert res["passed"] and res["identical_to_baseline"]


def test_gate_catches_label_collapse():
    res = _gate(["not enough evidence"] * 30)
    assert "label_collapse" in _failed(res)


def test_gate_catches_format_breakage():
    res = _gate(["Verdict: unclear"] * 3 + _baseline_replies()[3:])
    assert "unparseable" in _failed(res)


def test_gate_catches_a_gross_accuracy_drop():
    # Wrong on 10 of the baseline's 20 correct claims, spread across labels
    # so collapse does not fire first.
    flip = {"refuted": "supported", "supported": "refuted",
            "not enough evidence": "refuted", "conflicting": "supported"}
    replies = [flip[r] if i % 2 == 0 and i < 20 else r
               for i, r in enumerate(_baseline_replies())]
    res = _gate(replies)
    assert _failed(res) == {"accuracy"}


def test_gate_fails_on_missing_claims():
    with tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
        base = _gate_run(b, _baseline_replies())
        r = Run(run_dir=c)
        for cid, gold in list(zip(_IDS, _GOLDS))[:25]:
            r.add_prediction(cid, "claim", gold, _REPLY[gold], [])
        r.write()
        res = check(c, base, _IDS, DEFAULT_LIMITS)
    assert not res["passed"] and _failed(res) == {"coverage"}


def test_committed_gate_set_is_stratified_as_documented():
    import json
    with open(os.path.join(REPO_DIR, "data", "ci_gate_30.json")) as f:
        gate = json.load(f)
    from collections import Counter
    counts = Counter(c["gold_label"] for c in gate["claims"])
    assert counts == Counter(gate["quota"])
    assert len({c["claim_id"] for c in gate["claims"]}) == 30


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
            print(f"FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
