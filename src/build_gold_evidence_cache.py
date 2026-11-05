"""
build_gold_evidence_cache.py — materialize the GOLD (oracle) evidence into the
same retrieval-cache format the rest of the pipeline uses, so that faithfulness
(`rag_eval.py`) can be computed for the oracle config exactly like for the
BM25 (closed-web) and DuckDuckGo (open-web) configs.

The oracle system (`gold_baseline.py`) feeds the LLM AVeriTeC's official,
human-annotated question-answer evidence pairs directly — it never runs
retrieval, so no `top_passages` cache exists for it. This script rebuilds that
gold evidence as `top_passages` entries ([score, url, passage]) keyed
`claim_<idx>`, matching the DDG / BM25 retrieval caches, so the faithfulness
judge sees gold evidence in an identical shape.

Mirrors gold_baseline.load_gold_evidence_map: uses POSITIONAL claim index
(dev.json has duplicate claim texts), reads only questions/answers, and
deliberately excludes the `justification` field (it leaks the verdict).

Usage:
    python src/build_gold_evidence_cache.py \
        --data-file data/dev.json \
        --output-cache results/gold_evidence_dev500_retrieval_cache.json
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from bm25_baseline import _write_json_atomic


def build(data_file: str, output_cache: str) -> None:
    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)

    cache = {}
    for idx, item in enumerate(data):
        claim = item.get("claim", "")
        passages = []
        for q in item.get("questions", []):
            question = q.get("question", "")
            answers = [a.get("answer", "") for a in q.get("answers", [])
                       if a.get("answer")]
            if not (question and answers):
                continue
            # One QA pair -> one evidence passage. score 1.0 (gold), synthetic
            # url marks provenance; format matches DDG/BM25 "Title/Snippet" shape.
            passage = f"Q: {question}\nA: {' '.join(answers)}"
            passages.append([1.0, "gold://averitec-qa", passage])
        cache[f"claim_{idx}"] = {
            "claim": claim,
            "claim_idx": idx,
            "n_passages": len(passages),
            "top_passages": passages,
        }

    _write_json_atomic(output_cache, cache)
    n_with_ev = sum(1 for v in cache.values() if v["top_passages"])
    print(f"Wrote {len(cache)} entries ({n_with_ev} with gold evidence) -> {output_cache}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default="data/dev.json")
    ap.add_argument("--output-cache", required=True)
    args = ap.parse_args()
    build(args.data_file, args.output_cache)


if __name__ == "__main__":
    main()
