"""
dense_rerank.py — Week 3 Dense Re-ranking (BM25 prefilter + Dense rerank).

Pipeline:
  1. Retrieve BM25 top-N candidates (pool=30 default) per claim from the
     cached BM25 corpus.
  2. Re-rank those N passages with a sentence-transformer (all-mpnet-base-v2)
     by cosine(claim, passage); keep top-k (default 5).
  3. Feed re-ranked top-k to the LLM with a CoT + Conflict-first prompt and
     a token-robust label parser (avoids spurious matches in the CoT
     summary).

History: this file used to be `dense_rerank_v2.py` — the earlier v1 (fixed
top-5 BM25 pool, no real candidate expansion) is gone. For a clean
apples-to-apples comparison against BM25 under an identical plain prompt,
see reprompt_eval.py.
"""
import os
import json
import argparse
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import f1_score, classification_report
from dotenv import load_dotenv

from dataset import read_averitec_json
# Re-use the existing BM25 infrastructure (cache, retrieval, LLM clients)
from bm25_baseline import (
    build_langchain_client, call_langchain,
    build_native_client, call_native,
    build_user_input, _write_json_atomic,
    build_text_id_to_source_idx,
    retrieve_bm25_cached, get_ks_file
)

load_dotenv()

# ==========================================
# CoT prompt: summary-then-label, with explicit Conflicting priority rule.
# (Week 3 ablation found this prompt is OUTPERFORMED by the plain Zero-Shot
# prompt — see Finding 1. This prompt is retained because dense_rerank.py
# is also used in --retrieval-only mode where it is not invoked.)
# ==========================================
SYSTEM_PROMPT = """You are an expert fact-checker. You are given a claim and a set of evidence passages.

First, write a concise one-sentence summary of the core facts in the evidence.
Then, strictly based on your summary, classify the claim's veracity.

CRITICAL RULE: If the evidence contains BOTH clear supporting AND clear refuting facts, you MUST choose 'Conflicting Evidence/Cherrypicking'.

You MUST use the exact output format below:
Summary: <your summary>
Label: <supported / refuted / not enough evidence / conflicting evidence/cherrypicking>"""

LLM_LABEL_MAP = {
    "supported": 0,
    "refuted": 1,
    "not enough evidence": 2,
    "conflicting evidence/cherrypicking": 3,
}
TARGET_NAMES = ["Supported (0)", "Refuted (1)", "NEE (2)", "Conflicting (3)"]


def normalize_llm_output(raw_output: str) -> int:
    """Safe parser: extract only the text AFTER 'Label:' so that words in
    the CoT Summary (which may include 'supported' / 'refuted' in passing)
    don't contaminate the predicted-label match."""
    if "Label:" in raw_output:
        target_text = raw_output.split("Label:")[-1]
    elif "label:" in raw_output.lower():
        target_text = raw_output.lower().split("label:")[-1]
    else:
        target_text = raw_output

    text = target_text.strip().lower().strip('."\'* \n')

    if text in LLM_LABEL_MAP:
        return LLM_LABEL_MAP[text]
    if "conflicting" in text:
        return 3
    if "not enough" in text:
        return 2
    if "refuted" in text:
        return 1
    if "supported" in text:
        return 0
    return -1


def run_v2(data_file, source_idx_file, split_label, ks_root, split_ks_name,
           pool_size, top_k, backend, results_dir, retrieval_only=False):
    texts = read_averitec_json(data_file)
    texts = [t for t in texts if t.label is not None]

    # Force CPU by default. M2's MPS backend hangs in uninterruptible
    # sleep on long inputs from large KS files (claim_35 in dev has 851 docs
    # → ~30 long passages → mpnet on MPS deadlocks inside Metal). CPU on
    # M2 is sub-second for 30 sentences, so no real perf loss.
    # Override with EMBED_DEVICE=mps if you want to test MPS on smaller inputs.
    device = os.environ.get("EMBED_DEVICE", "cpu")
    print(f"Loading SentenceTransformer (all-mpnet-base-v2) on device={device}...")
    st_model = SentenceTransformer('all-mpnet-base-v2', device=device)

    if retrieval_only:
        client = None
        call_fn = None
        print("Mode: --retrieval-only (no LLM calls; use reprompt_eval.py "
              "afterwards on the resulting top_passages cache).")
    else:
        client = build_langchain_client() if backend == "langchain" else build_native_client()
        call_fn = call_langchain if backend == "langchain" else call_native

    os.makedirs(results_dir, exist_ok=True)
    # Dedicated cache file — never overwrites earlier BM25-only results
    cache_file = os.path.join(
        results_dir, f"dense_v2_{split_label}_pool{pool_size}_top{top_k}_cache.json")

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)

    text_id_to_source_idx = build_text_id_to_source_idx(
        data_file, source_idx_file)

    # Dummy timing/counter dicts to satisfy retrieve_bm25_cached's signature
    timings = {"ks_tokenize": 0.0, "bm25_scoring": 0.0}
    counters = {"corpus_cache_hits": 0, "corpus_cache_misses": 0,
                "ranking_cache_hits": 0, "ranking_cache_misses": 0}

    y_true, y_pred = [], []
    unparseable = new_calls = 0

    for i, text in enumerate(texts):
        cid = text.text_id

        if cid in cache:
            entry = cache[cid]
        else:
            claim_idx = text_id_to_source_idx.get(cid)
            if claim_idx is None:
                continue

            ks_file = get_ks_file(ks_root, split_ks_name, claim_idx)

            # Step 1: BM25 Top-30 candidate pool from the cached corpus
            # (warm corpus + cached ranking → fast, no LLM call needed here)
            passages, n_docs = retrieve_bm25_cached(
                query_text=text.original_claim,
                claim_idx=claim_idx,
                ks_file=ks_file,
                split_ks_name=split_ks_name,
                top_k=pool_size,           # Wide net: 30 candidates for rerank
                chunk_size=200,
                corpus_cache_dir=os.path.join(
                    results_dir, "bm25_corpus_cache"),
                ranking_cache_dir=os.path.join(
                    results_dir, "bm25_ranking_cache"),
                ranking_depth=pool_size,
                timings=timings,
                counters=counters,
            )

            # Step 2: Dense rerank — pick top-5 from the 30 by mpnet cosine
            if passages:
                claim_emb = st_model.encode(
                    text.original_claim, convert_to_tensor=True)
                passage_texts = [p[2] for p in passages]
                passage_embs = st_model.encode(
                    passage_texts, convert_to_tensor=True)

                scores = util.cos_sim(claim_emb, passage_embs)[0].cpu().numpy()
                ranked_indices = scores.argsort()[::-1][:top_k]

                # Repack as (score, url, passage) to match the BM25 schema
                top_passages = [(float(scores[idx]), passages[idx][1],
                                 passages[idx][2][:300]) for idx in ranked_indices]
            else:
                top_passages = []

            # 3. Skip LLM step in retrieval-only mode (Protocol Rule #2):
            # save top_passages and defer LLM verification to reprompt_eval.py.
            # This avoids burning API budget on a prompt we know we'll discard,
            # and keeps retrieval reusable across any future prompt design.
            if retrieval_only:
                entry = {
                    "claim": text.original_claim,
                    "claim_idx": claim_idx,
                    "top_passages": top_passages,
                    "true_label": text.label,
                    "raw_output": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
                cache[cid] = entry
                _write_json_atomic(cache_file, cache)
            else:
                # End-to-end: CoT + Metadata
                user_input = build_user_input(
                    text, top_passages, use_metadata=True)
                try:
                    raw_text, p_tok, c_tok = call_fn(
                        client, SYSTEM_PROMPT, user_input)
                    entry = {
                        "claim": text.original_claim,
                        "claim_idx": claim_idx,
                        "top_passages": top_passages,
                        "true_label": text.label,
                        "raw_output": raw_text,
                        "prompt_tokens": p_tok,
                        "completion_tokens": c_tok
                    }
                except Exception as e:
                    if "content_filter" in str(e) or "ResponsibleAI" in str(e):
                        entry = {
                            "claim": text.original_claim, "true_label": text.label,
                            "raw_output": "CONTENT_FILTERED", "prompt_tokens": 0, "completion_tokens": 0
                        }
                    else:
                        raise

                cache[cid] = entry
                new_calls += 1
                _write_json_atomic(cache_file, cache)

        if not retrieval_only:
            pred = normalize_llm_output(entry["raw_output"])
            if pred == -1:
                unparseable += 1
                print(f"  [unparseable] {cid}:\n{entry['raw_output']}")
            y_true.append(entry["true_label"])
            y_pred.append(pred)

        if (i+1) % 50 == 0:
            print(
                f"  ... {i+1}/{len(texts)} retrieved "
                f"({new_calls} new LLM calls)")

    if retrieval_only:
        print(f"\n{'='*60}")
        print(f"Retrieval-only complete. {len(cache)} claims cached to:")
        print(f"  {cache_file}")
        print("Run reprompt_eval.py on this cache to score with a chosen prompt.")
    else:
        macro_f1 = f1_score(y_true, y_pred, average='macro', labels=[0, 1, 2, 3])
        print(f"\n{'='*60}")
        print(f"Dense Rerank v2 (Pool={pool_size} -> Top-{top_k}) | {split_label}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print(classification_report(y_true, y_pred, labels=[
              0, 1, 2, 3], target_names=TARGET_NAMES, zero_division=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--source-file", default="data/train.json")
    parser.add_argument("--split-label", default="internal_dev_sample150")
    parser.add_argument(
        "--ks-root", default="/Volumes/Seagate Backup Plus Drive/AVeriTeC_knowledge_store/data_store/knowledge_store")
    # internal_dev_sample150 is stratified-sampled from train, so its KS sub-dir is "train"
    parser.add_argument("--split-ks-name", default="train")
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backend", default="langchain")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip the LLM verification step; only persist "
                             "BM25→Dense top_passages to cache. Feed the cache "
                             "to reprompt_eval.py to score with any prompt. "
                             "Aligns with Protocol Rule #2 (Decouple & Cache).")
    args = parser.parse_args()

    run_v2(args.data_file, args.source_file, args.split_label, args.ks_root,
           args.split_ks_name, args.pool_size, args.top_k, args.backend,
           args.results_dir, retrieval_only=args.retrieval_only)
