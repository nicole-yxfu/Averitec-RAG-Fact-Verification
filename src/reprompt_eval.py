"""
reprompt_eval.py — apples-to-apples re-prompting on a fixed retrieval cache.

Reads `top_passages` from a previously-saved retrieval cache (BM25 or Dense
re-rank), then re-runs the LLM verification step with a clean plain
zero-shot prompt + metadata. Lets us hold retrieval constant and isolate the
effect of prompt design — e.g. compare BM25 vs Dense fairly under the SAME
prompt, instead of conflating retrieval differences with prompt differences.

This replaces and merges the previous pure_eval.py + pure_eval_week2.py;
those two scripts were identical except for the hard-coded output filename,
which is now a CLI flag.

Few-shot ICL (Week 4):
  Pass `--fewshot-file <path.json>` to prepend worked examples to every
  user input. See data/fewshot_template.json for the expected schema.
  Examples are loaded once at startup; the LLM sees them in every call as
  in-context demonstrations before the target claim.

Usage:
  # Plain Zero-Shot (default) on Dense V2 retrieval cache
  python src/reprompt_eval.py \\
      --retrieval-cache results/dense_v2_internal_dev_sample150_pool30_top5_cache.json \\
      --output-cache results/dense_final_pure_eval_cache.json

  # Plain Zero-Shot on Week 2 BM25 retrieval (apples-to-apples vs Dense)
  python src/reprompt_eval.py \\
      --retrieval-cache results/bm25_baseline_internal_dev_sample150_with_metadata_top5_cache.json \\
      --output-cache results/week2_bm25_pure_eval_cache.json

  # Week 4: Few-shot ICL on Final Main System retrieval (BM25 top-5)
  python src/reprompt_eval.py \\
      --retrieval-cache results/bm25_baseline_internal_dev_sample150_with_metadata_top5_cache.json \\
      --output-cache results/bm25_fewshot8_sample150_cache.json \\
      --fewshot-file data/fewshot_gold_8shot.json
"""
import os
import json
import argparse
from sklearn.metrics import f1_score, classification_report
from dotenv import load_dotenv

from dataset import read_averitec_json
from bm25_baseline import (
    build_langchain_client, call_langchain,
    build_native_client, call_native,
    build_user_input, _write_json_atomic
)

load_dotenv()

# "plain" = the terse Week-3 prompt that beat the standard one by +0.040 on
# BM25 (Finding 1: Prompt > Retriever). "standard" = the more verbose prompt
# from bm25_baseline.py (imported below to guarantee it stays identical).
# --prompt-style selects between them so Finding 1 can be re-tested on any
# retrieval cache (e.g. the open-web DDG cache in Week 5 Task 4).
PLAIN_PROMPT = """You are an expert fact-checker. Given a claim and retrieved evidence, determine the veracity of the claim.

Output ONLY one of the following four labels, in lowercase, with no additional words, punctuation, or explanation:

supported
refuted
not enough evidence
conflicting evidence/cherrypicking"""

from bm25_baseline import SYSTEM_PROMPT as STANDARD_PROMPT

# "nee_guard" = PLAIN_PROMPT plus one paragraph defining when 'not enough
# evidence' is warranted. Motivated by a measured failure, not a hunch: on
# dev500 the open-web system predicts NEE 117 times where only 35 claims
# carry that gold label (3.3x over-prediction), and the faithfulness judge
# scores NEE verdicts lowest of all four classes (0.522 open-web vs 0.861
# Refuted) — i.e. the model reaches for NEE when evidence is merely
# incomplete rather than genuinely silent, and cannot ground that choice.
#
# The added paragraph only sharpens the decision boundary; it adds no
# information about the label distribution and names no class as more or
# less likely, so it is not a prior smuggled in as an instruction. The
# expected cost is NEE recall, and that trade-off is what gets measured.
NEE_GUARD_PROMPT = PLAIN_PROMPT + """

When deciding on 'not enough evidence', apply this standard: choose it only when the evidence is silent on the claim's substance, or addresses a different question entirely. If the evidence does speak to the claim but is incomplete, imprecise, indirect, or covers only part of it, commit to 'supported' or 'refuted' according to where that evidence points. Partial or indirect evidence is evidence, not an absence of it."""

SYSTEM_PROMPT = PLAIN_PROMPT  # default; overridden by --prompt-style at runtime

LLM_LABEL_MAP = {
    "supported": 0,
    "refuted": 1,
    "not enough evidence": 2,
    "conflicting evidence/cherrypicking": 3,
}
TARGET_NAMES = ["Supported (0)", "Refuted (1)", "NEE (2)", "Conflicting (3)"]


VALID_LABELS = set(LLM_LABEL_MAP.keys())


def load_fewshot(path: str) -> list[dict]:
    """
    Load + validate few-shot examples from a JSON file.

    Schema (a list of objects):
        {
            "claim":          "<the claim text>",            # required
            "speaker":        "<optional speaker>",          # optional
            "source":         "<optional reporting source>", # optional
            "evidence_text":  "<pre-formatted evidence>",    # required;
                              # e.g. "[1] passage 1\\n[2] passage 2"
                              # or "Q: ... A: ...\\nQ: ... A: ..."
            "label":          "supported"                    # required;
                              # must be in LLM_LABEL_MAP (lowercase)
        }

    Validation is strict: any malformed example aborts the run so a
    silent prompt corruption never reaches the LLM.
    """
    with open(path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"{path}: expected non-empty JSON list")
    for i, ex in enumerate(examples):
        for k in ("claim", "evidence_text", "label"):
            if k not in ex or not str(ex[k]).strip():
                raise ValueError(f"{path}: example {i} missing/empty key '{k}'")
        if ex["label"].strip().lower() not in VALID_LABELS:
            raise ValueError(
                f"{path}: example {i} label {ex['label']!r} not in "
                f"{sorted(VALID_LABELS)}")
    return examples


def format_fewshot_block(examples: list[dict]) -> str:
    """
    Format few-shot examples as a single text block to PREPEND to every
    user message. Examples are separated by `---` to make boundaries
    visually obvious to the LLM (in-context format matters per Singhal 2024
    and the Week 3 negative finding from our hand-crafted few-shot attempt).
    """
    parts = [f"Here are {len(examples)} worked examples to guide your "
             f"judgment. Each shows a claim, its evidence, and the correct "
             f"label. Apply the same standard to the new claim that follows.\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"--- Example {i} ---")
        parts.append(f"Claim: {ex['claim']}")
        if ex.get("speaker"):
            parts.append(f"Speaker: {ex['speaker']}")
        if ex.get("source"):
            parts.append(f"Source:  {ex['source']}")
        parts.append("")
        parts.append("Evidence:")
        parts.append(ex["evidence_text"].rstrip())
        parts.append("")
        parts.append(f"Label: {ex['label'].strip().lower()}")
        parts.append("")
    parts.append("--- Now classify the following claim using the same standard ---\n")
    return "\n".join(parts)


def normalize_llm_output(raw_output: str) -> int:
    text = raw_output.strip().lower().strip('."\'* \n')
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


def run_reprompt_eval(data_file, retrieval_cache_file, output_cache_file,
                      backend, results_dir, fewshot_file=None,
                      system_prompt=SYSTEM_PROMPT):
    print(f"Loading original texts to restore metadata: {data_file}")
    texts = read_averitec_json(data_file)
    texts_dict = {t.text_id: t for t in texts if t.label is not None}

    print(f"Loading retrieval passages from: {retrieval_cache_file}")
    with open(retrieval_cache_file, 'r', encoding='utf-8') as f:
        retrieval_cache = json.load(f)

    # Few-shot ICL: load + format once, prepend to every user input.
    # The block must be the prefix (not suffix) so the LLM reads examples
    # before the target — putting examples after the target would invite
    # the model to "lookahead-leak" features of the target into example
    # interpretation.
    fewshot_prefix = ""
    if fewshot_file:
        fewshot = load_fewshot(fewshot_file)
        fewshot_prefix = format_fewshot_block(fewshot) + "\n"
        print(f"Few-shot mode: loaded {len(fewshot)} examples from {fewshot_file}")
    else:
        print("Few-shot mode: disabled (plain Zero-Shot)")

    client = build_langchain_client() if backend == "langchain" else build_native_client()
    call_fn = call_langchain if backend == "langchain" else call_native

    os.makedirs(results_dir, exist_ok=True)

    final_cache = {}
    if os.path.exists(output_cache_file):
        with open(output_cache_file, 'r', encoding='utf-8') as f:
            final_cache = json.load(f)
        print(f"  Loaded {len(final_cache)} pre-existing results from output cache.")

    y_true, y_pred = [], []
    new_calls = 0
    total_claims = len(retrieval_cache)

    mode_label = ("Few-shot ICL + metadata"
                  if fewshot_prefix else "plain Zero-Shot + metadata")
    print(f"Re-prompting {total_claims} claims with {mode_label}...")

    for processed_count, (cid, entry) in enumerate(retrieval_cache.items(), 1):
        if cid in final_cache:
            final_entry = final_cache[cid]
        else:
            original_text = texts_dict.get(cid)
            if not original_text:
                continue

            safe_passages = entry.get("top_passages", [])
            target_input = build_user_input(
                original_text, safe_passages, use_metadata=True)
            user_input = fewshot_prefix + target_input

            try:
                raw_text, _p, _c = call_fn(client, system_prompt, user_input)
                final_entry = {
                    "claim": entry.get("claim", original_text.original_claim),
                    "true_label": entry["true_label"],
                    "raw_output": raw_text,
                }
            except Exception as e:
                err = str(e)
                tag = "CONTENT_FILTERED" if (
                    "content_filter" in err or "ResponsibleAI" in err
                ) else "ERROR"
                final_entry = {
                    "claim": entry.get("claim", original_text.original_claim),
                    "true_label": entry["true_label"],
                    "raw_output": tag,
                }

            final_cache[cid] = final_entry
            new_calls += 1
            _write_json_atomic(output_cache_file, final_cache)
            print(f"  {processed_count}/{total_claims} | {cid} | "
                  f"new API calls: {new_calls}")

        pred = normalize_llm_output(final_entry["raw_output"])
        y_true.append(final_entry["true_label"])
        y_pred.append(pred)

    macro_f1 = f1_score(y_true, y_pred,
                        average='macro', labels=[0, 1, 2, 3])

    print(f"\n{'=' * 60}")
    print("Re-prompt Evaluation Summary")
    print(f"Retrieval source : {os.path.basename(retrieval_cache_file)}")
    print(f"Output cache     : {os.path.basename(output_cache_file)}")
    print(f"Macro-F1         : {macro_f1:.4f}")
    print(classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3],
        target_names=TARGET_NAMES, zero_division=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Re-prompt a saved retrieval cache with a clean Zero-Shot prompt."
    )
    parser.add_argument("--data-file",
                        default="data/internal_dev_sample150.json",
                        help="Original AVeriTeC JSON, for restoring metadata "
                             "(speaker/source) onto the LLM input.")
    parser.add_argument("--retrieval-cache", required=True,
                        help="Input cache containing `top_passages` to re-prompt.")
    parser.add_argument("--output-cache", required=True,
                        help="Where to write LLM responses for this re-prompt run.")
    parser.add_argument("--backend", default="langchain",
                        choices=["langchain", "native"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--fewshot-file", default=None,
                        help="Path to a JSON file of few-shot ICL examples. "
                             "If provided, the examples are prepended to "
                             "every user message. See "
                             "data/fewshot_template.json for the schema. "
                             "Omit for plain Zero-Shot (the Week 3 default).")
    parser.add_argument("--prompt-style", default="plain",
                        choices=["plain", "standard", "nee_guard"],
                        help="System prompt: 'plain' (Week 3 terse prompt, "
                             "default), 'standard' (verbose prompt from "
                             "bm25_baseline.py), or 'nee_guard' (plain plus "
                             "a definition of when 'not enough evidence' is "
                             "warranted). Lets Finding 1 (Prompt > Retriever) "
                             "be re-tested on any retrieval cache.")
    args = parser.parse_args()

    chosen_prompt = {"standard": STANDARD_PROMPT,
                     "nee_guard": NEE_GUARD_PROMPT}.get(
                         args.prompt_style, PLAIN_PROMPT)
    print(f"Prompt style: {args.prompt_style}")

    run_reprompt_eval(args.data_file, args.retrieval_cache, args.output_cache,
                      args.backend, args.results_dir,
                      fewshot_file=args.fewshot_file,
                      system_prompt=chosen_prompt)
