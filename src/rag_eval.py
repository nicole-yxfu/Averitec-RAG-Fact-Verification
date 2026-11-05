"""
rag_eval.py — RAG-style evaluation of the fact-verification pipeline.

Beyond answer correctness (Macro-F1), a deployable RAG system must be judged on
whether its outputs are actually *grounded in the evidence it retrieved* rather
than hallucinated from the model's parametric knowledge. This script adds the
**faithfulness / groundedness** axis (the RAGAS-style metric most emphasised in
RAG job postings) via an LLM-as-judge.

For each claim it joins:
  - the retrieved evidence  (from a retrieval cache: `top_passages`)
  - the system's verdict    (from a prediction cache: `raw_output`)
and asks a judge LLM: *given ONLY this evidence, is this verdict justified?*
The faithfulness rate is the fraction of verdicts the judge calls justified.

A verdict can be CORRECT yet UNFAITHFUL (right answer, wrong/absent grounding)
or WRONG yet FAITHFUL (a reasonable read of misleading evidence). Separating
faithfulness from correctness is exactly what a fact-checking system needs:
an unfaithful verdict is dangerous even when it happens to be right.

Judge calls are cached (atomic writes) so runs are resumable and cheap on reruns.

Usage:
    python src/rag_eval.py \
        --retrieval-cache results/ddg_direct_dev500_retrieval_cache.json \
        --prediction-cache results/ddg_direct_dev500_plain_eval_cache.json \
        --output-cache results/faithfulness_ddg_plain.json \
        --sample 150
"""
import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from sklearn.metrics import f1_score  # noqa: F401  (kept for parity / future axes)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from bm25_baseline import (build_native_client, call_native,
                           build_langchain_client, call_langchain,
                           _write_json_atomic)

load_dotenv()

LABEL_MAP = {
    "supported": 0, "refuted": 1,
    "not enough evidence": 2, "conflicting evidence/cherrypicking": 3,
}
NAMES = ["Supported", "Refuted", "NEE", "Conflicting"]
MAX_EVIDENCE_CHARS = 3000

JUDGE_SYSTEM = """You are a strict evaluator of fact-checking FAITHFULNESS. \
You are given a claim, a set of retrieved evidence passages, and a system's \
veracity verdict. Judge ONLY whether the verdict is justified by the evidence \
provided — not whether it is true in the real world.

A verdict is faithful when:
- "supported": the evidence clearly supports the claim.
- "refuted": the evidence clearly contradicts the claim.
- "not enough evidence": the evidence is insufficient, irrelevant, or absent.
- "conflicting evidence/cherrypicking": the evidence both supports and contradicts.

Answer with exactly one word: faithful  or  unfaithful"""


def normalize_verdict(raw: str) -> int:
    t = str(raw).strip().lower().strip('."\'* \n')
    if t in LABEL_MAP:
        return LABEL_MAP[t]
    for k, v in (("conflicting", 3), ("not enough", 2), ("refuted", 1), ("supported", 0)):
        if k in t:
            return v
    return -1


def parse_judge(raw: str) -> int:
    """1 = faithful, 0 = unfaithful, -1 = unparseable. Order matters:
    'unfaithful' contains 'faithful', so test it first."""
    t = str(raw).strip().lower()
    if "unfaithful" in t:
        return 0
    if "faithful" in t:
        return 1
    return -1


def format_evidence(top_passages) -> str:
    lines, total = [], 0
    for i, item in enumerate(top_passages, 1):
        # top_passages entries are [score, url, passage]
        _, url, passage = item
        block = f"[{i}] {url}\n{passage}"
        if total + len(block) > MAX_EVIDENCE_CHARS:
            break
        lines.append(block)
        total += len(block)
    return "\n".join(lines) if lines else "(no evidence retrieved)"


def build_judge_input(claim: str, verdict_name: str, top_passages) -> str:
    return (f"Claim: {claim}\n\n"
            f"Retrieved evidence:\n{format_evidence(top_passages)}\n\n"
            f"System verdict: {verdict_name}\n\n"
            f"Is this verdict justified by the retrieved evidence alone? "
            f"Answer: faithful or unfaithful")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-cache", required=True, help="cache with top_passages (evidence)")
    ap.add_argument("--prediction-cache", required=True, help="cache with raw_output (verdict)")
    ap.add_argument("--output-cache", required=True, help="where judge decisions are cached")
    ap.add_argument("--sample", type=int, default=-1, help="judge only first N claims (-1 = all)")
    ap.add_argument("--backend", default="langchain", choices=["langchain", "native"])
    args = ap.parse_args()

    retrieval = json.load(open(args.retrieval_cache, encoding="utf-8"))
    predictions = json.load(open(args.prediction_cache, encoding="utf-8"))

    cache = {}
    if os.path.exists(args.output_cache):
        cache = json.load(open(args.output_cache, encoding="utf-8"))
        print(f"Resume: {len(cache)} judged already")

    client = build_langchain_client() if args.backend == "langchain" else build_native_client()
    call_fn = call_langchain if args.backend == "langchain" else call_native

    cids = list(retrieval.keys())
    if args.sample > 0:
        cids = cids[:args.sample]

    new_calls = 0
    for i, cid in enumerate(cids, 1):
        if cid in cache:
            continue
        if cid not in predictions:
            continue
        rentry = retrieval[cid]
        pred = normalize_verdict(predictions[cid]["raw_output"])
        if pred < 0:
            continue  # unparseable verdict — skip faithfulness
        user_input = build_judge_input(
            rentry.get("claim", ""), NAMES[pred], rentry.get("top_passages", []))
        # Judge call with bounded exponential backoff on rate limits (HTTP 429).
        # Content-filtered evidence is recorded (faithful=-1) so reruns skip it;
        # any other error after retries is left uncached so a rerun retries it.
        faithful, note, done = None, None, False
        for attempt in range(6):
            try:
                raw, _p, _c = call_fn(client, JUDGE_SYSTEM, user_input)
                faithful, note, done = parse_judge(raw), raw.strip()[:40], True
                break
            except Exception as e:
                err = str(e)
                if "content_filter" in err or "ResponsibleAI" in err:
                    faithful, note, done = -1, "content_filtered", True
                    break
                if "429" in err or "rate" in err.lower():
                    wait = 5 * (2 ** attempt)  # 5,10,20,40,80,160s
                    print(f"  [429] {cid}: backing off {wait}s (attempt {attempt+1}/6)")
                    time.sleep(wait)
                    continue
                print(f"  [error] {cid}: {err[:120]}")
                break
        if not done:
            continue  # exhausted retries / hard error — leave uncached for rerun
        cache[cid] = {"pred": pred, "faithful": faithful, "judge_raw": note}
        new_calls += 1
        try:
            _write_json_atomic(args.output_cache, cache)
        except OSError as e:
            # A transient filesystem hiccup on the atomic replace must not drop
            # all progress; the in-memory cache persists on the next write.
            print(f"  [warn] cache write failed ({e}); will retry next claim")
        if new_calls % 25 == 0:
            print(f"  {i}/{len(cids)} | judged {new_calls} new")

    # ── report ──────────────────────────────────────────────────────────
    judged = [v for v in cache.values() if v["faithful"] in (0, 1)]
    n = len(judged)
    if n == 0:
        print("No judged claims.")
        return
    faithful = sum(v["faithful"] for v in judged)
    print(f"\n{'='*52}")
    print(f"Faithfulness: {faithful}/{n} = {faithful/n:.3f}")
    print(f"(retrieval: {os.path.basename(args.retrieval_cache)})")
    print("\nBy predicted verdict:")
    for lbl in range(4):
        sub = [v for v in judged if v["pred"] == lbl]
        if sub:
            f = sum(v["faithful"] for v in sub)
            print(f"  {NAMES[lbl]:<12} {f:>3}/{len(sub):<3} = {f/len(sub):.3f}")


if __name__ == "__main__":
    main()
