"""
build_judge_check.py — sample verdicts for a human check of the
faithfulness judge (src/rag_eval.py).

THE QUESTION THIS ANSWERS
-------------------------
The faithfulness numbers in the README (0.637 / 0.753 / 0.822 across the
evidence ladder) come from an LLM judge. "How do you know the judge is
right?" had no answer. This sets up a small, honest one: a person applies
the judge's own criterion to 50 verdicts, blind to the judge's answer, and
score_judge_check.py measures agreement.

DESIGN
------
  Same criterion   The labelling page shows the judge's rules verbatim:
                   given ONLY the evidence, is the verdict justified?
  Same evidence    Exactly the passages the judge saw (rag_eval's 3000-char
                   cap applied the same way).
  Blind            The page never loads the judge's label or the gold
                   label. They live in judge_check/key.json, which only the
                   scorer reads.
  Stratified       25 verdicts the judge called faithful + 25 it called
                   unfaithful. A random 50 would hold only ~12 unfaithful
                   ones (the judge says faithful 75% of the time), too few
                   to say anything about that direction. The scorer
                   re-weights by the real 75/25 split when it estimates
                   population-level agreement.

Run: deployed configuration (open-web DDG top-5, plain prompt), dev500.
Claims the judge could not score (content filter) are excluded.

USAGE
-----
    python src/build_judge_check.py     # writes judge_check/items.js + key.json
"""
from __future__ import annotations

import json
import os
import random
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(REPO, *p)  # noqa: E731

FAITHFULNESS = "results/faithfulness_ddg_plain_dev500.json"
RETRIEVAL = "results/ddg_direct_dev500_retrieval_cache.json"
PREDICTIONS = "results/ddg_direct_dev500_plain_eval_cache.json"
PER_STRATUM = 25
SEED = 0
MAX_EVIDENCE_CHARS = 3000          # must equal rag_eval.MAX_EVIDENCE_CHARS

VERDICTS = ["Supported", "Refuted", "Not enough evidence", "Conflicting evidence"]
RULES = {   # the judge's own wording, src/rag_eval.py JUDGE_SYSTEM
    "Supported": "the evidence clearly supports the claim.",
    "Refuted": "the evidence clearly contradicts the claim.",
    "Not enough evidence": "the evidence is insufficient, irrelevant, or absent.",
    "Conflicting evidence": "the evidence both supports and contradicts.",
}


def shown_passages(top_passages: list) -> list:
    """Mirror rag_eval.format_evidence: BREAK at the first block over the cap."""
    out, total = [], 0
    for i, (_, url, passage) in enumerate(top_passages, 1):
        block = f"[{i}] {url}\n{passage}"
        if total + len(block) > MAX_EVIDENCE_CHARS:
            break
        out.append((url, passage))
        total += len(block)
    return out


def split_passage(text: str) -> tuple[str, str]:
    title, snippet = "", text
    if text.startswith("Title: "):
        first, _, rest = text.partition("\n")
        title = first[len("Title: "):].strip()
        snippet = rest[len("Snippet: "):] if rest.startswith("Snippet: ") else rest
    return title, snippet.strip()


def build() -> tuple[list[dict], dict]:
    import rag_eval
    assert rag_eval.MAX_EVIDENCE_CHARS == MAX_EVIDENCE_CHARS

    with open(R(FAITHFULNESS), encoding="utf-8") as f:
        faith = json.load(f)
    with open(R(RETRIEVAL), encoding="utf-8") as f:
        retr = json.load(f)
    with open(R(PREDICTIONS), encoding="utf-8") as f:
        preds = json.load(f)

    strata = {1: [], 0: []}
    for cid, v in faith.items():
        pred = int(v["pred"]) if str(v["pred"]).lstrip("-").isdigit() else -1
        if v["faithful"] in (0, 1) and 0 <= pred <= 3:
            strata[v["faithful"]].append(cid)
    population = {"faithful": len(strata[1]), "unfaithful": len(strata[0])}

    rng = random.Random(SEED)
    chosen = []
    for label in (1, 0):
        pool = sorted(strata[label], key=lambda c: int(c.split("_")[1]))
        chosen += [(c, label) for c in rng.sample(pool, PER_STRATUM)]
    rng.shuffle(chosen)                 # mix the strata: order reveals nothing

    items, key = [], {}
    for n, (cid, judge) in enumerate(chosen, start=1):
        verdict = VERDICTS[int(faith[cid]["pred"])]
        evidence = []
        for url, passage in shown_passages(retr[cid]["top_passages"]):
            title, snippet = split_passage(passage)
            evidence.append({"title": title, "snippet": snippet, "url": url,
                             "domain": urlparse(url).netloc.removeprefix("www.")})
        items.append({"n": n, "id": cid, "claim": preds[cid]["claim"],
                      "verdict": verdict, "rule": RULES[verdict],
                      "evidence": evidence})
        key[cid] = {"judge_faithful": judge, "judge_raw": faith[cid]["judge_raw"],
                    "verdict": verdict, "gold": preds[cid]["true_label"]}

    meta = {"source_faithfulness": FAITHFULNESS, "seed": SEED,
            "per_stratum": PER_STRATUM, "population": population, "key": key}
    return items, meta


def main() -> None:
    items, meta = build()
    os.makedirs(R("judge_check"), exist_ok=True)
    with open(R("judge_check", "items.js"), "w", encoding="utf-8") as f:
        f.write("// Generated by src/build_judge_check.py. Holds NO judge or gold labels.\n")
        f.write("window.ITEMS = ")
        json.dump(items, f, ensure_ascii=False, indent=1)
        f.write(";\n")
    with open(R("judge_check", "key.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    by_verdict = {}
    for it in items:
        by_verdict[it["verdict"]] = by_verdict.get(it["verdict"], 0) + 1
    print(f"Wrote judge_check/items.js ({len(items)} items) and key.json; "
          f"population {meta['population']}; verdicts in sample {by_verdict}")


if __name__ == "__main__":
    main()
