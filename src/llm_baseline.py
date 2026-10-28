import os
import json
import random
import argparse

from dotenv import load_dotenv
from sklearn.metrics import f1_score, classification_report

from dataset import read_averitec_json

# Set random seed for strict reproducibility (does not affect LLM API calls,
# but keeps this script consistent with the rest of the pipeline).
random.seed(42)

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt design
#
# The label strings requested from the LLM mirror dataset.LABEL_MAP exactly
# (lowercased), so that normalize_llm_output() can map back to the same
# numeric IDs (0-3) used everywhere else in the project.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert fact-checker. Given a claim, determine its veracity based on your own knowledge.

Output ONLY one of the following four labels, in lowercase, with no additional words, punctuation, or explanation:

supported
refuted
not enough evidence
conflicting evidence/cherrypicking"""

LLM_LABEL_MAP = {
    "supported": 0,
    "refuted": 1,
    "not enough evidence": 2,
    "conflicting evidence/cherrypicking": 3,
}

TARGET_NAMES = ["Supported (0)", "Refuted (1)", "NEE (2)", "Conflicting (3)"]


def normalize_llm_output(raw_output):
    """
    Maps a raw LLM completion string to one of the four numeric label IDs
    (0-3), or None if the output cannot be parsed.

    Exact match (after stripping case/whitespace/quotes/punctuation) is
    tried first. If that fails, a keyword fallback is used, ordered to
    avoid ambiguous substring collisions (e.g. "conflicting" is checked
    before "refuted"/"supported" since a hedged answer might mention
    multiple labels).
    """
    text = raw_output.strip().lower()
    text = text.strip('."\'* ')

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


def build_input_text(text, use_metadata):
    """
    Constructs the user-facing input string for the LLM.

    With --no-metadata (use_metadata=False):
        Only the raw claim text is passed. This matches the "pure" zero-shot
        setting used in AVeriTeC official baselines and competition systems,
        enabling direct comparison with published results.

    With metadata (use_metadata=True, default):
        Speaker and reporting source are appended to the claim. This matches
        the information available to the Averaged Perceptron baseline, making
        the two systems directly comparable. Used for the primary "LLM +
        Metadata (Zero-shot)" entry in the Week 5 results table.
    """
    if not use_metadata:
        return text.original_claim

    return (
        f"Claim: {text.original_claim}\n"
        f"Speaker: {text.speaker}\n"
        f"Source: {text.reporting_source}"
    )


# ---------------------------------------------------------------------------
# LLM client wrappers
#
# Two backends are supported behind the same call_fn(client, input_text)
# interface, per the Week 1 Risk Checkpoint:
#   - "langchain": AzureChatOpenAI (LangChain v0.3, default)
#   - "native"   : AzureOpenAI (official openai SDK, fallback)
# ---------------------------------------------------------------------------

print("DEPLOYMENT =", os.environ["AZURE_OPENAI_DEPLOYMENT"])
print("ENDPOINT   =", os.environ["AZURE_OPENAI_ENDPOINT"])
print("API_VER    =", os.environ.get("AZURE_OPENAI_API_VERSION"))


def build_langchain_client():
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0.0,
    )


print("DEPLOYMENT =", os.environ["AZURE_OPENAI_DEPLOYMENT"])
print("ENDPOINT =", os.environ["AZURE_OPENAI_ENDPOINT"])


def call_langchain(client, input_text):
    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=input_text),
    ]
    response = client.invoke(messages)

    raw_text = response.content
    usage = response.response_metadata.get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    return raw_text, prompt_tokens, completion_tokens


def build_native_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def call_native(client, input_text):
    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ],
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    return raw_text, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Main evaluation loop with result caching
#
# Every API response is cached to:
#   results/llm_baseline_{split}_{mode}_cache.json
# keyed by claim_id. Re-running the script reuses cached responses and only
# calls the API for claims not yet cached -- preventing redundant API costs.
#
# The cache filename includes the mode suffix ("with_metadata" or
# "no_metadata") so the two ablation conditions never overwrite each other.
# ---------------------------------------------------------------------------

def run_baseline(split_name, data_file, backend, sample, use_metadata,
                 results_dir):

    texts = read_averitec_json(data_file)
    texts = [t for t in texts if t.label is not None]

    # Determine display name for this run
    mode_label = "LLM + Metadata (Zero-shot)" if use_metadata else "Pure LLM (Zero-shot, claim only)"
    mode_suffix = "with_metadata" if use_metadata else "no_metadata"

    print(f"\n{'=' * 60}")
    print(f"{mode_label}")
    print(f"Split: {split_name} | Backend: {backend}")
    print(f"{'=' * 60}")

    # Debug mode warning
    if sample is not None:
        texts = texts[:sample]
        print(f"  ⚠️  Debug mode: only processing first {sample} claims. "
              f"Use --sample -1 for full evaluation.")

    print(f"Processing {len(texts)} claims...")

    os.makedirs(results_dir, exist_ok=True)
    cache_file = os.path.join(
        results_dir,
        f"llm_baseline_{split_name}_{mode_suffix}_cache.json"
    )

    # Load existing cache
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached responses from {cache_file}")

    # Build client
    if backend == "langchain":
        client = build_langchain_client()
        call_fn = call_langchain
    elif backend == "native":
        client = build_native_client()
        call_fn = call_native
    else:
        raise ValueError(f"Unknown backend: {backend}")

    y_true, y_pred = [], []

    # Track token usage separately for new calls vs. full cumulative total
    new_prompt_tokens, new_completion_tokens = 0, 0
    total_prompt_tokens, total_completion_tokens = 0, 0

    unparseable = 0
    new_calls = 0

    for i, text in enumerate(texts):
        cid = text.text_id

        if cid in cache:
            entry = cache[cid]
        else:
            input_text = build_input_text(text, use_metadata)
            try:
                raw_text, p_tok, c_tok = call_fn(client, input_text)
                entry = {
                    "claim": text.original_claim,
                    "speaker": text.speaker,
                    "source": text.reporting_source,
                    "input_mode": mode_suffix,
                    "true_label": text.label,
                    "raw_output": raw_text,
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                }
            except Exception as e:
                error_str = str(e)
                # Azure content filter: log and skip, treat as unparseable
                if "content_filter" in error_str or "ResponsibleAI" in error_str:
                    print(f"  [content_filter] claim_id={cid} — skipped")
                    entry = {
                        "claim": text.original_claim,
                        "speaker": text.speaker,
                        "source": text.reporting_source,
                        "input_mode": mode_suffix,
                        "true_label": text.label,
                        "raw_output": "CONTENT_FILTERED",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    }
                else:
                    # Re-raise unexpected errors
                    raise

            cache[cid] = entry
            new_calls += 1
            new_prompt_tokens += entry["prompt_tokens"]
            new_completion_tokens += entry["completion_tokens"]

            # Incremental cache save: partial progress is never lost.
            # Atomic write (tmp + os.replace) avoids the truncation window
            # of `open('w')+json.dump`, which previously wiped a 500-entry
            # cache down to 8 entries when another run started with an
            # empty cache file. See bm25_baseline._write_json_atomic.
            tmp_path = f"{cache_file}.tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, cache_file)

        pred = normalize_llm_output(entry["raw_output"])
        if pred is None:
            unparseable += 1
            print(f"  [unparseable] claim_id={cid} "
                  f"raw_output={entry['raw_output']!r}")
            # Treat as wrong prediction for all classes (safe default)
            pred = -1

        y_true.append(entry["true_label"])
        y_pred.append(pred)
        total_prompt_tokens += entry["prompt_tokens"]
        total_completion_tokens += entry["completion_tokens"]

        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(texts)} "
                  f"({new_calls} new API calls so far)")

    # Macro-F1: labels=[0,1,2,3] ensures the metric always reflects the full
    # 4-class task even when a class is absent from this subset.
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=[0, 1, 2, 3])

    print(f"\n--- Results: {split_name} | {mode_label} ---")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"Unparseable outputs : {unparseable}/{len(texts)}")
    print(f"New API calls this run : {new_calls}")
    print(f"Token usage — new calls this run   : "
          f"{new_prompt_tokens} prompt + {new_completion_tokens} completion")
    print(f"Token usage — cumulative (all cache): "
          f"{total_prompt_tokens} prompt + {total_completion_tokens} completion")

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3],
        target_names=TARGET_NAMES, zero_division=0))

    return {
        "split": split_name,
        "mode": mode_label,
        "n_claims": len(texts),
        "macro_f1": macro_f1,
        "unparseable": unparseable,
        "new_calls": new_calls,
        "new_prompt_tokens": new_prompt_tokens,
        "new_completion_tokens": new_completion_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "LLM Baseline — Zero-shot 4-class fact verification.\n\n"
            "Two modes:\n"
            "  Default : LLM + Metadata (claim + speaker + source)\n"
            "            Comparable to the Averaged Perceptron baseline.\n"
            "  --no-metadata : Pure LLM (claim only)\n"
            "            Comparable to published AVeriTeC baselines.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split", choices=["internal_dev", "official_dev", "both"],
        default="both",
        help="Which split(s) to evaluate on (default: both)")
    parser.add_argument(
        "--backend", choices=["langchain", "native"], default="langchain",
        help="API backend: 'langchain' (default) or 'native' openai SDK")
    parser.add_argument(
        "--sample", type=int, default=20,
        help=(
            "Number of claims to process. Default is 20 for safe debugging. "
            "Use --sample -1 to run on the full split."
        ))
    parser.add_argument(
        "--no-metadata", action="store_true",
        help=(
            "Run in pure claim-only mode (no speaker/source). "
            "Use for ablation: compare against default (with metadata) "
            "to quantify the contribution of metadata to LLM performance."
        ))
    args = parser.parse_args()

    sample = None if args.sample == -1 else args.sample
    use_metadata = not args.no_metadata

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    results_dir = os.path.join(project_dir, 'results')

    splits_to_run = []
    if args.split in ("internal_dev", "both"):
        splits_to_run.append(
            ("internal_dev", os.path.join(data_dir, "internal_dev.json")))
    if args.split in ("official_dev", "both"):
        splits_to_run.append(
            ("official_dev", os.path.join(data_dir, "dev.json")))

    summary = []
    for split_name, data_file in splits_to_run:
        result = run_baseline(
            split_name, data_file, args.backend,
            sample, use_metadata, results_dir
        )
        summary.append(result)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for r in summary:
        print(
            f"{r['split']:<15} | {r['mode']}\n"
            f"  n={r['n_claims']}  Macro-F1={r['macro_f1']:.4f}  "
            f"unparseable={r['unparseable']}  new_calls={r['new_calls']}\n"
            f"  tokens (new): {r['new_prompt_tokens']} in / "
            f"{r['new_completion_tokens']} out\n"
            f"  tokens (total cached): {r['total_prompt_tokens']} in / "
            f"{r['total_completion_tokens']} out\n"
        )
