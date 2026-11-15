"""
build_demo.py — generate the data behind the static demo page (demo/).

The demo makes NO live model calls. Every number on the page is derived here
from committed artifacts, so the page cannot drift from the README:

  verdicts, evidence     runs/ddg_dev500_plain        (the deployed config)
  confidence             selective_prediction.py, recomputed exactly as
                         published (5-fold cross-fitted, seed 0); the script
                         asserts AURC and accuracy@70% match the committed
                         report before writing anything
  grounded / agreement   the faithfulness judge and the second-prompt run
  gold verdict + reason  data/dev.json (professional fact-checkers)

The examples are hand-picked for READABILITY (short, mostly non-partisan
claims), not for success: the set deliberately includes confident mistakes
and abstentions that would have been right, because a demo that only shows
wins would misrepresent a system that is right 60% of the time.

USAGE
-----
    python src/build_demo.py          # writes demo/examples.js
"""
from __future__ import annotations

import json
import os
import sys
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runs import load_run  # noqa: E402
from selective_prediction import (crossfit_scores, featurize,  # noqa: E402
                                  load_rows, risk_coverage)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(REPO, *p)  # noqa: E731

PREDICTIONS = "results/ddg_direct_dev500_plain_eval_cache.json"
FAITHFULNESS = "results/faithfulness_ddg_plain_dev500.json"
SECOND_PROMPT = "results/ddg_dev500_nee_guard_eval_cache.json"
PUBLISHED = "results/selective_prediction_dev500.json"
COVERAGE = 0.70

# Display order: an easy win first, then a save, then the rest mixed.
EXAMPLES = ["claim_0", "claim_117", "claim_98", "claim_48", "claim_12",
            "claim_41", "claim_31", "claim_26", "claim_27", "claim_243",
            "claim_328"]

LABELS = ["Supported", "Refuted", "Not Enough Evidence",
          "Conflicting Evidence/Cherrypicking"]
SHORT = ["Supported", "Refuted", "Not enough evidence",
         "Conflicting evidence"]


def split_passage(text: str) -> tuple[str, str]:
    """'Title: X\\nSnippet: Y' -> (X, Y)."""
    title, snippet = "", text
    if text.startswith("Title: "):
        first, _, rest = text.partition("\n")
        title = first[len("Title: "):].strip()
        snippet = rest[len("Snippet: "):] if rest.startswith("Snippet: ") else rest
    return title, snippet.strip()


def build_payload() -> dict:
    """Everything the page shows. Asserts it matches the published report."""
    rows = load_rows(R(PREDICTIONS), R(FAITHFULNESS), R(SECOND_PROMPT))
    y = np.array([r["correct"] for r in rows])
    scores = crossfit_scores(featurize(rows), y, n_splits=5, seed=0)
    rc = risk_coverage(scores, y)

    # Refuse to build a page whose numbers differ from the published ones.
    with open(R(PUBLISHED), encoding="utf-8") as f:
        pub = json.load(f)["summaries"][0]
    k = int(round(COVERAGE * len(rows)))
    answered = {rows[i]["cid"] for i in rc["order"][:k]}
    acc_cov = float(np.mean([r["correct"] for r in rows if r["cid"] in answered]))
    pub_cov = next(p for p in pub["points"] if abs(p["coverage"] - COVERAGE) < 1e-9)
    assert round(rc["aurc"], 4) == pub["aurc"], (rc["aurc"], pub["aurc"])
    assert round(acc_cov, 4) == pub_cov["accuracy"], (acc_cov, pub_cov)

    score = {r["cid"]: float(scores[i]) for i, r in enumerate(rows)}
    s_in = min(score[c] for c in answered)
    s_out = max(s for c, s in score.items() if c not in answered)
    threshold = (s_in + s_out) / 2

    # How often each predicted label is right: the strongest single signal.
    by_pred = {}
    for i, name in enumerate(SHORT):
        sel = [r["correct"] for r in rows if r["pred"] == i]
        if sel:
            by_pred[name] = {"n": len(sel), "accuracy": round(float(np.mean(sel)), 3)}

    with open(R("data", "dev.json"), encoding="utf-8") as f:
        dev = json.load(f)
    run = load_run(R("runs", "ddg_dev500_plain"))
    ranking = run.rankings()
    row_by = {r["cid"]: r for r in rows}

    examples = []
    for cid in EXAMPLES:
        rec = dev[int(cid.split("_")[1])]
        r = row_by[cid]
        evidence = []
        for pid in ranking[cid]:
            p = run.passages[pid]
            title, snippet = split_passage(p["text"])
            evidence.append({"title": title, "snippet": snippet, "url": p["url"],
                             "domain": urlparse(p["url"]).netloc.removeprefix("www.")})
        is_answered = cid in answered
        correct = bool(r["correct"])
        outcome = ("correct" if is_answered and correct else
                   "miss" if is_answered else
                   "caught" if not correct else "cautious")
        examples.append({
            "id": cid,
            "claim": rec["claim"],
            "date": rec.get("claim_date"),
            "evidence": evidence,
            "verdict": SHORT[r["pred"]] if r["pred"] >= 0 else "Unparseable",
            "confidence": round(score[cid], 3),
            "answered": is_answered,
            "grounded": None if r["faithful"] is None else bool(r["faithful"]),
            "agree": None if r["agree"] is None else bool(r["agree"]),
            "gold": SHORT[LABELS.index(rec["label"])],
            "justification": (rec.get("justification") or "").strip(),
            "outcome": outcome,
        })

    payload = {
        "source": "Precomputed from runs/ddg_dev500_plain and "
                  "results/selective_prediction_dev500.json by src/build_demo.py",
        "n_claims": len(rows),
        "threshold": round(threshold, 3),
        "coverage": COVERAGE,
        "accuracy_all": round(float(y.mean()), 4),
        "accuracy_answered": round(acc_cov, 4),
        "aurc": round(rc["aurc"], 4),
        "aurc_random": round(1 - float(y.mean()), 4),
        "accuracy_by_verdict": by_pred,
        "examples": examples,
    }
    return payload


HEADER = "// Generated by src/build_demo.py — do not edit by hand.\nwindow.DEMO = "


def render(payload: dict) -> str:
    return HEADER + json.dumps(payload, ensure_ascii=False, indent=1) + ";\n"


def main() -> None:
    payload = build_payload()
    os.makedirs(R("demo"), exist_ok=True)
    with open(R("demo", "examples.js"), "w", encoding="utf-8") as f:
        f.write(render(payload))
    examples = payload["examples"]
    outcomes = {o: sum(e["outcome"] == o for e in examples)
                for o in ("correct", "miss", "caught", "cautious")}
    print(f"Wrote demo/examples.js: {len(examples)} examples {outcomes}, "
          f"threshold {payload['threshold']:.3f}, accuracy "
          f"{payload['accuracy_all']:.4f} -> {payload['accuracy_answered']:.4f} "
          f"at {COVERAGE:.0%} coverage (matches published)")


if __name__ == "__main__":
    main()
