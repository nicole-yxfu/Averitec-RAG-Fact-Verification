"""
gold_baseline.py — LLM + Gold Evidence (Upper Bound)

This is the upper-bound system in the project plan's 8-system comparison.
It uses AVeriTeC's official, human-annotated question-answer evidence pairs
directly from train.json / dev.json — NO Knowledge Store access, NO
retrieval (BM25 or Dense), NO external network calls. The only cost is the
LLM call itself, so 500 claims should take minutes, not hours.

IMPORTANT — what evidence is used and what is deliberately excluded:
  Used      : claim['questions'][i]['question'] + ['answers'][j]['answer']
              These are fact-focused QA pairs (e.g. "Where was the claim
              first published?" -> "It was first published on Scoopertino")
              written to help VERIFY the claim, not to state the verdict.
  Excluded  : claim['justification']
              This field is written AFTER the human annotator already
              determined the label, and often states or implies the verdict
              directly. Using it as "evidence" would leak the answer to the
              LLM and inflate Macro-F1 artificially — the resulting score
              would not measure verification ability, only label leakage.
              This script never reads this field.

Usage:
  # Quick smoke-test (10 claims)
  python src/gold_baseline.py --sample 10

  # Full official_dev evaluation (expected: a few minutes)
  python src/gold_baseline.py --sample -1

  # No-metadata ablation (mirrors llm_baseline.py / bm25_baseline.py)
  python src/gold_baseline.py --sample -1 --no-metadata
"""

import os
import json
import random
import argparse

from dotenv import load_dotenv
from sklearn.metrics import f1_score, classification_report

from dataset import read_averitec_json
from bm25_baseline import build_text_id_to_source_idx, _write_json_atomic

random.seed(42)
load_dotenv()

MAX_EVIDENCE_CHARS = 3000  # same cap used in bm25_baseline.py, for fair comparison

LLM_LABEL_MAP = {
    "supported":                          0,
    "refuted":                            1,
    "not enough evidence":                2,
    "conflicting evidence/cherrypicking": 3,
}
TARGET_NAMES = ["Supported (0)", "Refuted (1)", "NEE (2)", "Conflicting (3)"]

SYSTEM_PROMPT = """You are an expert fact-checker. You are given a claim and a set of official, human-verified question-answer evidence pairs about that claim.

Based ONLY on the provided evidence, determine the claim's veracity.

Output ONLY one of the following four labels, in lowercase, with no additional words, punctuation, or explanation:

supported
refuted
not enough evidence
conflicting evidence/cherrypicking"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Gold evidence extraction
# ═══════════════════════════════════════════════════════════════════════════════

def load_gold_evidence_map(source_json_file: str) -> dict:
    """
    Builds a dict: claim_idx (positional index in source file) -> list of
    (question, [answers]) pairs, read directly from the official AVeriTeC
    JSON (train.json or dev.json).

    Uses POSITIONAL index rather than claim text as the key. dev.json /
    train.json contain duplicate claim texts (confirmed while debugging
    bm25_baseline.py — 500 claims in dev.json produced only 491 unique
    texts). A text-based lookup would silently let duplicate claims
    overwrite each other's evidence entry, misattributing gold QA pairs
    between unrelated claims. Since evidence_map is built from the SAME
    file as the claims being evaluated (no cross-file lookup is needed,
    unlike bm25_baseline.py's internal_dev -> train.json case), positional
    indexing is both simpler and fully safe here.

    Deliberately does NOT read the 'justification' field — see module
    docstring for why.
    """
    with open(source_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    evidence_map = {}
    for idx, item in enumerate(data):
        qa_pairs = []
        for q in item.get('questions', []):
            question_text = q.get('question', '')
            answers = [a.get('answer', '') for a in q.get('answers', [])
                       if a.get('answer')]
            if question_text and answers:
                qa_pairs.append((question_text, answers))
        evidence_map[idx] = qa_pairs

    return evidence_map


def build_user_input(text, qa_pairs: list, use_metadata: bool = True) -> str:
    """
    Constructs the LLM user message: claim (+ optional metadata) + gold
    QA evidence. Mirrors bm25_baseline.py's build_user_input() structure
    so the two systems are as directly comparable as possible.
    """
    lines = [f"Claim: {text.original_claim}"]

    if use_metadata:
        lines.append(f"Speaker: {text.speaker}")
        lines.append(f"Source:  {text.reporting_source}")

    lines.append("")

    if qa_pairs:
        lines.append("Official Verified Evidence (Question-Answer pairs):")
        total_chars = 0
        for rank, (question, answers) in enumerate(qa_pairs, 1):
            answer_text = " / ".join(answers)
            entry = f"[{rank}] Q: {question}\n    A: {answer_text}"
            if total_chars + len(entry) > MAX_EVIDENCE_CHARS:
                break
            lines.append(entry)
            lines.append("")
            total_chars += len(entry)
    else:
        lines.append(
            "Note: No gold evidence question-answer pairs are available "
            "for this claim."
        )

    lines.append(
        "Based on the evidence above, what is the veracity of this claim?"
    )
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LLM clients (identical to bm25_baseline.py / llm_baseline.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_langchain_client():
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0.0,
    )


def call_langchain(client, system_prompt: str, user_text: str):
    from langchain_core.messages import SystemMessage, HumanMessage
    response = client.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ])
    usage = response.response_metadata.get("token_usage", {})
    return (response.content,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0))


def build_native_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def call_native(client, system_prompt: str, user_text: str):
    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ],
        temperature=0.0,
    )
    return (response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens)


def normalize_llm_output(raw_output: str):
    """Identical logic to bm25_baseline.py / llm_baseline.py."""
    text = raw_output.strip().lower().strip('."\'* ')
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
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_gold_baseline(split_name, data_file, source_idx_file, backend,
                      sample, use_metadata, results_dir):
    texts = read_averitec_json(data_file)
    texts = [t for t in texts if t.label is not None]

    mode_suffix = "with_metadata" if use_metadata else "no_metadata"
    mode_label = (f"LLM + Gold Evidence "
                  f"({'w/ metadata' if use_metadata else 'claim only'})")

    print(f"\n{'=' * 60}")
    print(f"Gold Evidence Baseline: {mode_label}")
    print(f"Split: {split_name} | Backend: {backend}")
    print(f"{'=' * 60}")

    if sample is not None:
        texts = texts[:sample]
        print(f"  ⚠️  Debug mode: processing first {sample} claims only.")
    print(f"Total claims to process: {len(texts)}")

    print(
        f"Loading gold evidence from: {os.path.basename(source_idx_file)} ...")
    evidence_map = load_gold_evidence_map(source_idx_file)
    print(f"  Indexed gold evidence for {len(evidence_map)} claims.")

    # Resolves each evaluated claim to its correct position in source_idx_file.
    # For official_dev this is the identity mapping (dev.json == dev.json).
    # For internal_dev, this correctly recovers the original train.json row
    # via full-record / metadata / claim-text fallback matching — reusing the
    # exact same validated logic bm25_baseline.py uses for Knowledge Store
    # lookups, instead of assuming position i is safe (it is NOT, since
    # internal_dev.json is a re-indexed, stratified subset of train.json).
    print(f"Building text-id → source-index map: {os.path.basename(data_file)} "
          f"against {os.path.basename(source_idx_file)} ...")
    text_id_to_source_idx = build_text_id_to_source_idx(
        data_file, source_idx_file)
    print(f"  Resolved {len(text_id_to_source_idx)} data rows.")

    os.makedirs(results_dir, exist_ok=True)
    cache_file = os.path.join(
        results_dir, f"gold_baseline_{split_name}_{mode_suffix}_cache.json"
    )

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached results from {cache_file}")

    client = (build_langchain_client() if backend == "langchain"
              else build_native_client())
    call_fn = call_langchain if backend == "langchain" else call_native

    y_true, y_pred = [], []
    new_prompt_tokens, new_completion_tokens = 0, 0
    total_prompt_tokens, total_completion_tokens = 0, 0
    unparseable = new_calls = no_evidence_count = 0

    for i, text in enumerate(texts):
        cid = text.text_id
        claim_idx = text_id_to_source_idx.get(cid)

        if cid in cache and cache[cid].get("claim_idx") == claim_idx:
            entry = cache[cid]
        else:
            if cid in cache:
                print(
                    f"  [cache refresh] {cid}: claim_idx changed, recomputing")

            qa_pairs = evidence_map.get(
                claim_idx, []) if claim_idx is not None else []
            if claim_idx is None:
                print(f"  [warn] {cid}: not found in source index — "
                      f"claim='{text.original_claim[:60]}...'")
            if not qa_pairs:
                no_evidence_count += 1

            user_input = build_user_input(text, qa_pairs, use_metadata)

            try:
                raw_text, p_tok, c_tok = call_fn(
                    client, SYSTEM_PROMPT, user_input)
                entry = {
                    "claim":             text.original_claim,
                    "claim_idx":         claim_idx,
                    "n_qa_pairs":        len(qa_pairs),
                    "input_mode":        mode_suffix,
                    "true_label":        text.label,
                    "raw_output":        raw_text,
                    "prompt_tokens":     p_tok,
                    "completion_tokens": c_tok,
                }
            except Exception as e:
                err = str(e)
                if "content_filter" in err or "ResponsibleAI" in err:
                    print(f"  [content_filter] {cid} — skipped")
                    entry = {
                        "claim":             text.original_claim,
                        "claim_idx":         claim_idx,
                        "n_qa_pairs":        len(qa_pairs),
                        "input_mode":        mode_suffix,
                        "true_label":        text.label,
                        "raw_output":        "CONTENT_FILTERED",
                        "prompt_tokens":     0,
                        "completion_tokens": 0,
                    }
                else:
                    raise

            cache[cid] = entry
            new_calls += 1
            new_prompt_tokens += entry["prompt_tokens"]
            new_completion_tokens += entry["completion_tokens"]

            # Atomic write — see bm25_baseline._write_json_atomic() for rationale
            # (prevents the previously-observed cache-truncation incident).
            _write_json_atomic(cache_file, cache)

        pred = normalize_llm_output(entry["raw_output"])
        if pred is None:
            unparseable += 1
            print(f"  [unparseable] {cid}  raw={entry['raw_output']!r}")
            pred = -1

        y_true.append(entry["true_label"])
        y_pred.append(pred)
        total_prompt_tokens += entry["prompt_tokens"]
        total_completion_tokens += entry["completion_tokens"]

        if (i + 1) % 50 == 0:
            print(f"  … {i+1}/{len(texts)} processed "
                  f"({new_calls} new API calls so far)")

    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=[0, 1, 2, 3])

    print(f"\n{'─' * 60}")
    print(f"Results: {split_name} | {mode_label}")
    print(f"{'─' * 60}")
    print(f"Macro-F1                   : {macro_f1:.4f}")
    print(f"Claims with no gold evidence: {no_evidence_count}/{len(texts)}")
    print(f"Unparseable outputs        : {unparseable}/{len(texts)}")
    print(f"New API calls this run     : {new_calls}")
    print(f"Token usage (new)          : "
          f"{new_prompt_tokens} prompt + {new_completion_tokens} completion")
    print(f"Token usage (all cached)   : "
          f"{total_prompt_tokens} prompt + {total_completion_tokens} completion")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3],
        target_names=TARGET_NAMES, zero_division=0))

    return {
        "split": split_name, "mode": mode_label, "n_claims": len(texts),
        "macro_f1": macro_f1, "no_evidence_count": no_evidence_count,
        "unparseable": unparseable, "new_calls": new_calls,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="LLM + Gold Evidence (Upper Bound) — no retrieval needed."
    )
    parser.add_argument("--split", choices=["internal_dev", "official_dev", "both"],
                        default="official_dev")
    parser.add_argument(
        "--backend", choices=["langchain", "native"], default="langchain")
    parser.add_argument("--sample", type=int, default=10,
                        help="Claims to process; -1 = full split (default: 10)")
    parser.add_argument("--no-metadata", action="store_true")
    args = parser.parse_args()

    sample = None if args.sample == -1 else args.sample
    use_metadata = not args.no_metadata

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    results_dir = os.path.join(project_dir, 'results')

    split_configs = {
        "internal_dev": (os.path.join(data_dir, "internal_dev.json"),
                         os.path.join(data_dir, "train.json")),
        "official_dev": (os.path.join(data_dir, "dev.json"),
                         os.path.join(data_dir, "dev.json")),
    }

    splits_to_run = (["internal_dev", "official_dev"] if args.split == "both"
                     else [args.split])

    summary = []
    for split_name in splits_to_run:
        data_file, source_idx_file = split_configs[split_name]
        result = run_gold_baseline(
            split_name=split_name, data_file=data_file,
            source_idx_file=source_idx_file, backend=args.backend,
            sample=sample, use_metadata=use_metadata, results_dir=results_dir,
        )
        summary.append(result)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for r in summary:
        print(f"{r['split']:<15} | {r['mode']}\n"
              f"  n={r['n_claims']}  Macro-F1={r['macro_f1']:.4f}  "
              f"no_evidence={r['no_evidence_count']}  "
              f"unparseable={r['unparseable']}  new_calls={r['new_calls']}\n")
