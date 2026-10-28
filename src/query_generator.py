"""
query_generator.py — Query construction strategies for retrieval (BM25 now,
Dense RAG in Week 3, Multi-hop in Week 4).

Strategy A — Direct Query:
  Use the raw claim text as-is. This is what bm25_baseline.py has been
  using all along (text.original_claim passed straight to BM25), so the
  already-completed 500-claim official_dev BM25 run already IS the
  Strategy A result — it does not need to be re-run.

Strategy B — LLM-Rewritten (Keyword) Query:
  An LLM extracts the key entities, numbers, dates, and core relation from
  the claim, and rewrites it as a short search-engine-style query, e.g.
  "The film was directed by Nolan" -> "Nolan director filmography"

IMPORTANT: both strategies share the exact same claim text in the final
verification prompt (the "Claim: ..." line built by build_user_input() in
bm25_baseline.py / gold_baseline.py). Only the text used to QUERY the
retrieval index changes — switching query strategy never changes what the
LLM is told the claim is, when it makes its final veracity prediction.

Strategy B caches its LLM-rewritten queries to disk, keyed by claim text,
in results/query_strategy_b_cache.json. This cache is intentionally
SEPARATE from bm25_baseline.py's result/corpus/ranking caches: the
rewritten query for a claim does not depend on top_k, chunk_size, or
retrieval method, so it is reused across every future ablation run
(top-k sweep, Dense RAG in Week 3, etc.) without re-calling the LLM.

Usage (as a library):
  from query_generator import generate_query
  query_text = generate_query(claim_text, strategy="keyword",
                              client=client, call_fn=call_langchain,
                              cache=query_cache, cache_path=cache_path)

Usage (standalone preview, no retrieval involved):
  python src/query_generator.py --split internal_dev --preview 10
"""

import os
import json
import argparse

from dotenv import load_dotenv

load_dotenv()

QUERY_CACHE_FILE_DEFAULT = "query_strategy_b_cache.json"
DECOMPOSE_CACHE_FILE_DEFAULT = "query_strategy_c_decompose_cache.json"

KEYWORD_REWRITE_SYSTEM_PROMPT = """You are a search query optimization assistant. Given a factual claim, extract the key named entities, numbers, dates, and core relation, and rewrite it as a concise web search query.

Rules:
- Output ONLY the search query, nothing else (no quotes, no explanation, no trailing punctuation).
- Keep it short: 3-8 words.
- Preserve specific names, numbers, and dates exactly as written in the claim.
- Drop filler words (the, was, is, that, which, etc).

Example:
Claim: The film was directed by Christopher Nolan and grossed over $800 million worldwide.
Query: Christopher Nolan director film $800 million gross

Example:
Claim: Nigeria's current population exceeds 200 million.
Query: Nigeria population 200 million"""


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy A — Direct Query
# ═══════════════════════════════════════════════════════════════════════════════

def generate_query_direct(claim_text: str) -> str:
    """Strategy A: the raw claim text, unmodified."""
    return claim_text


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy B — LLM-Rewritten Keyword Query
# ═══════════════════════════════════════════════════════════════════════════════

def load_query_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_query_cache(cache_path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def generate_query_keyword(claim_text: str, client, call_fn,
                           cache: dict = None, cache_path: str = None) -> str:
    """
    Strategy B: an LLM rewrites the claim as a short keyword search query.

    Falls back to the direct claim text if the LLM call fails or returns
    an empty string, so retrieval never silently receives an empty query.

    cache (optional): an in-memory dict the caller persists across calls,
    avoiding repeated LLM calls for a claim already rewritten earlier
    (within this run, or in a previous run via cache_path).
    """
    if cache is not None and claim_text in cache:
        return cache[claim_text]

    try:
        raw_output, _p_tok, _c_tok = call_fn(
            client, KEYWORD_REWRITE_SYSTEM_PROMPT, claim_text
        )
        rewritten = raw_output.strip().strip('."\'')
        if not rewritten:
            rewritten = claim_text   # never return an empty query
    except Exception as e:
        print(f"  [query_generator warn] keyword rewrite failed for "
              f"claim, falling back to direct query: {e}")
        rewritten = claim_text

    if cache is not None:
        cache[claim_text] = rewritten
        if cache_path:
            save_query_cache(cache_path, cache)

    return rewritten


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy C — Claim Decomposition into Sub-questions
# ═══════════════════════════════════════════════════════════════════════════════
#
# Inspired by UHH's "Question-Retrieve" strategy (FEVER 2024) and Singhal et
# al. (2024). The LLM decomposes a long fact-checking claim into 2-3 short
# sub-questions; each sub-question is then independently retrieved by BM25
# (in bm25_baseline.py), and the union of top-k passages is deduplicated +
# scored to produce the final top-k for the verification LLM.
#
# Unlike Strategy A/B which return ONE query string, Strategy C returns a
# LIST of query strings. The bm25_baseline.py Strategy C branch handles
# multi-query retrieval and dedup; this function only generates the queries.

DECOMPOSE_SYSTEM_PROMPT = """You are an assistant that decomposes a factual claim into the small number of sub-questions a fact-checker would ask to verify or refute it.

Rules:
- Output ONLY a JSON array of 2 to 3 short questions (strings). No prose, no markdown fences, no explanation.
- Each question must be a single sentence ending with "?".
- Each question must be answerable from the kind of evidence found in news articles, government reports, or fact-check archives.
- Preserve specific named entities, numbers, and dates from the claim exactly.
- Together, the questions should cover the verifiable components of the claim. Avoid redundancy.

Example:
Claim: In 2018, NASA's James Webb Space Telescope launched and successfully reached the L2 Lagrange point within 30 days.
Output: ["When was NASA's James Webb Space Telescope launched?", "Did the James Webb Space Telescope reach the L2 Lagrange point?", "How long did the James Webb Space Telescope take to reach L2?"]

Example:
Claim: Nigeria's population exceeds 200 million.
Output: ["What is the current population of Nigeria?"]"""


def _parse_decomposition(raw_output: str, claim_text: str) -> list[str]:
    """Robust JSON parser for decomposition output.

    Handles common LLM-output deviations:
      - Markdown fences ```json ... ```
      - Stray prose before/after the JSON array
      - Single-quote JSON (rare but happens)
      - Output that's just a single string instead of a list

    On any parse failure, falls back to [claim_text] (= Strategy A
    equivalent), so retrieval never silently receives zero queries.
    """
    text = raw_output.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop opening fence (and possible "json" language tag) + closing fence
        text = "\n".join(l for l in lines[1:-1] if not l.strip().startswith("```"))
    text = text.strip()
    # Try to locate the JSON array bracket (in case LLM added explanation)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try single-quote → double-quote (very rough; only safe if no
        # apostrophes in questions, which is true for our examples)
        try:
            parsed = json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            return [claim_text]
    # Normalize: must be a list of non-empty strings
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return [claim_text]
    cleaned = [q.strip() for q in parsed if isinstance(q, str) and q.strip()]
    if not cleaned:
        return [claim_text]
    # Cap at 3 questions (prompt asks for 2-3, but enforce as defense)
    return cleaned[:3]


def generate_query_decomposition(claim_text: str, client, call_fn,
                                 cache: dict = None,
                                 cache_path: str = None) -> list[str]:
    """
    Strategy C: an LLM decomposes the claim into 2-3 sub-questions for
    independent BM25 retrieval.

    Returns a LIST of query strings (vs. Strategy A/B which return one string).
    Falls back to [claim_text] (Strategy A equivalent) if the LLM call fails
    or the output cannot be parsed into a non-empty list — retrieval never
    silently receives zero queries.

    cache (optional): an in-memory dict the caller persists across calls,
    avoiding repeated LLM calls for a claim already decomposed earlier.
    """
    if cache is not None and claim_text in cache:
        return cache[claim_text]

    try:
        raw_output, _p_tok, _c_tok = call_fn(
            client, DECOMPOSE_SYSTEM_PROMPT, claim_text
        )
        questions = _parse_decomposition(raw_output, claim_text)
    except Exception as e:
        print(f"  [query_generator warn] decomposition call failed for "
              f"claim, falling back to direct query: {e}")
        questions = [claim_text]

    if cache is not None:
        cache[claim_text] = questions
        if cache_path:
            save_query_cache(cache_path, cache)

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

def generate_query(claim_text: str, strategy: str, client=None,
                   call_fn=None, cache: dict = None,
                   cache_path: str = None):
    """
    Dispatches to the requested query strategy.

    Returns:
        str  for strategies 'direct' and 'keyword' (single query)
        list[str] for strategy 'decompose' (multi-query)

    Callers must handle both return types — see bm25_baseline.py's
    Strategy C branch for the multi-query retrieval + dedup pattern.
    """
    if strategy == "direct":
        return generate_query_direct(claim_text)
    elif strategy == "keyword":
        if client is None or call_fn is None:
            raise ValueError(
                "strategy='keyword' requires an LLM client and call_fn"
            )
        return generate_query_keyword(claim_text, client, call_fn,
                                      cache=cache, cache_path=cache_path)
    elif strategy == "decompose":
        if client is None or call_fn is None:
            raise ValueError(
                "strategy='decompose' requires an LLM client and call_fn"
            )
        return generate_query_decomposition(claim_text, client, call_fn,
                                            cache=cache, cache_path=cache_path)
    else:
        raise ValueError(
            f"Unknown query strategy: {strategy!r}. "
            "Expected 'direct', 'keyword', or 'decompose'."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone preview — sanity-check Strategy B rewrites without running
# any retrieval. Useful before committing to a full Recall@5 ablation run.
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import read_averitec_json
    from bm25_baseline import build_langchain_client, call_langchain

    parser = argparse.ArgumentParser(
        description="Preview Strategy B (LLM-rewritten) queries for a few claims."
    )
    parser.add_argument("--split", choices=["internal_dev", "official_dev"],
                        default="internal_dev")
    parser.add_argument("--data-file", default=None,
                        help="Override claims file (e.g. a stratified sample)")
    parser.add_argument("--preview", type=int, default=10,
                        help="Number of claims to preview (default: 10)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_file = args.data_file or os.path.join(
        project_dir, 'data', f"{args.split}.json")
    results_dir = os.path.join(project_dir, 'results')
    cache_path = os.path.join(results_dir, QUERY_CACHE_FILE_DEFAULT)

    texts = read_averitec_json(data_file)
    texts = [t for t in texts if t.label is not None][:args.preview]

    client = build_langchain_client()
    cache = load_query_cache(cache_path)

    print(f"{'=' * 70}")
    print(f"Strategy A vs Strategy B — {len(texts)} claims preview")
    print(f"{'=' * 70}\n")

    for text in texts:
        direct = generate_query_direct(text.original_claim)
        keyword = generate_query_keyword(
            text.original_claim, client, call_langchain,
            cache=cache, cache_path=cache_path
        )
        print(f"Claim                : {text.original_claim}")
        print(f"Strategy A (direct)  : {direct}")
        print(f"Strategy B (keyword) : {keyword}")
        print()
