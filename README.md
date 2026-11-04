# What Actually Drives Retrieval-Augmented Fact Verification?

**A controlled study on AVeriTeC showing that *where* evidence comes from matters far more than *how* it is retrieved.**

Fact-checking with retrieval-augmented LLMs has three tunable components — the **evidence source**, the **prompt**, and the **retrieval strategy** — and most research pours effort into the last one (dense retrievers, query reformulation, decomposition). This project asks: *which one actually moves the needle?* By decoupling the pipeline and caching every stage, each experiment changes exactly one component, cleanly isolating its effect on 4-class veracity prediction (Supported / Refuted / Not Enough Evidence / Conflicting).

### Headline result

> **Evidence Source ≫ Prompt Design > Retrieval Strategy**

Climbing an evidence-availability ladder on the official 500-claim dev set (Macro-F1):

| Configuration | Macro-F1 | |
|---|---|---|
| Averaged Perceptron (no evidence) | 0.3546 | traditional baseline |
| LLM zero-shot (no evidence) | 0.3365 | reasoning alone isn't the bottleneck |
| BM25 closed-web + plain prompt | 0.3644 | curated Knowledge Store |
| **DuckDuckGo open-web + plain prompt** | **0.4495** | **+0.085, McNemar p = 2.5e-7** |
| Gold evidence (oracle) | 0.5633 | upper bound |

Switching the **evidence source** (closed → open web) gives the single largest gain (+0.085, statistically significant), closing 43% of the gap to the oracle ceiling. In contrast, swapping the **retrieval architecture** (BM25 → dense re-ranking) moves Macro-F1 by just −0.007 (within noise), and sophisticated query reformulation *hurts* (keyword −0.035, decomposition −0.042). A simpler **prompt** reliably adds a secondary +0.03–0.04 across both evidence sources.

An **oracle-gap error analysis** (quantitative decomposition + 20-case qualitative study, cross-verified by 3 independent LLM annotators) traces the residual error to the Conflicting class (54.6% of the gap) and an annotation-convention mismatch — and shows ~20% of the gap is irreducible annotation ambiguity, i.e. the gold ceiling is not ground truth.

**Stack:** Python · BM25 (rank-bm25) · sentence-transformers (mpnet) · DuckDuckGo · Azure OpenAI (`gpt-4.1-mini`) · scikit-learn · NLTK

## Full results

All configurations on the official 500-claim dev set (Macro-F1 + per-class F1). Regenerate this table from the committed caches with `python src/make_results_table.py` — no API calls needed.

| Evidence | Configuration | Macro-F1 | Supported F1 | Refuted F1 | NEE F1 | Conflicting F1 | n |
|---|---|---|---|---|---|---|---|
| No evidence | Majority baseline | 0.1894 | 0.000 | 0.758 | 0.000 | 0.000 | 500 |
|  | Random baseline | 0.1929 | 0.201 | 0.384 | 0.069 | 0.118 | 500 |
|  | Averaged Perceptron | 0.3546 | 0.357 | 0.758 | 0.218 | 0.085 | 500 |
|  | LLM zero-shot (no retrieval) | 0.3365 | 0.551 | 0.578 | 0.174 | 0.043 | 500 |
| Closed web | BM25 + plain prompt | 0.3644 | 0.558 | 0.596 | 0.150 | 0.154 | 500 |
|  | Dense rerank + plain prompt | 0.3572 | 0.570 | 0.645 | 0.178 | 0.036 | 500 |
|  | BM25 + 8-shot ICL | 0.3754 | 0.541 | 0.632 | 0.102 | **0.226** | 500 |
| Open web | DuckDuckGo + standard prompt | 0.4180 | 0.674 | 0.659 | 0.214 | 0.125 | 500 |
|  | **DuckDuckGo + plain prompt** | **0.4495** | 0.712 | 0.711 | 0.237 | 0.138 | 500 |
| Oracle | Gold evidence | 0.5633 | 0.858 | 0.862 | 0.487 | 0.045 | 500 |

Reading down the three evidence rungs (closed → open → oracle) is the main result; the two prompt rows under *Open web* isolate the prompt effect at a fixed source; *Dense rerank* and *8-shot ICL* are the retrieval- and prompt-side ablations. Even under gold evidence, the **Conflicting** class stays near zero (F1 0.045) — the motivation for the error analysis below.

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

Create a `.env` in this directory with Azure OpenAI credentials (never committed):

```
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
```

## Data

All AVeriTeC splits used by the reported experiments are committed under `data/` (~20 MB total), so the repo reproduces out of the box:

- `train.json`, `dev.json` — official AVeriTeC train / dev
- `internal_train.json`, `internal_dev.json` — our stratified split of train (via `src/split_data.py`)
- `internal_dev_sample150.json` — stratified 150-claim ablation subset
- `fewshot_clean8_pragref.json`, `fewshot_template.json` — few-shot ICL artifacts

Only the AVeriTeC **Knowledge Store** (used for closed-web BM25) is not committed due to size — download it via `src/download_ks.py` (kept local).

## How to reproduce the headline numbers

All results are cached in `results/*.json`; the commands below regenerate them. The pipeline is **decoupled**: retrieval writes a cache, then `reprompt_eval.py` runs the LLM on it — so a prompt ablation never re-retrieves.

```bash
# 1. Traditional baseline (no evidence)
python src/train_perceptron.py            # → averaged perceptron, Macro-F1 0.3546
python src/eval_week1_official_500.py

# 2. LLM zero-shot, no retrieval  (metadata is on by default; --sample -1 = full split)
python src/llm_baseline.py --split official_dev --sample -1

# 3. Closed-web: BM25 over the Knowledge Store (retrieval, then LLM)
python src/warmup_dev_corpus.py           # ~10 h, one-time corpus warm-up
python src/bm25_baseline.py --split official_dev --sample -1 --retrieval-only
python src/reprompt_eval.py \
    --data-file data/dev.json \
    --retrieval-cache results/bm25_baseline_official_dev_with_metadata_top5_cache.json \
    --output-cache results/bm25_official_dev_plain_prompt_cache.json   # 0.3644

# 4. Open-web: DuckDuckGo (the +0.085 result)
python src/open_web_baseline.py \
    --data-file data/dev.json \
    --output-cache results/ddg_direct_dev500_retrieval_cache.json --sample -1
python src/reprompt_eval.py \
    --data-file data/dev.json \
    --retrieval-cache results/ddg_direct_dev500_retrieval_cache.json \
    --output-cache results/ddg_direct_dev500_plain_eval_cache.json     # 0.4495

# 5. Oracle upper bound
python src/gold_baseline.py --split official_dev --sample -1           # 0.5633

# 6. Prompt ablation (Finding: plain > standard, replicates across sources)
python src/reprompt_eval.py --data-file data/dev.json \
    --retrieval-cache results/ddg_direct_dev500_retrieval_cache.json \
    --output-cache results/ddg_direct_dev500_standard_eval_cache.json --prompt-style standard

# 7. Error analysis (oracle gap decomposition + qualitative case dump)
python src/analyze_gold_gap.py
python src/prep_gold_gap_cases.py
```

## Tests

Minimal unit tests cover the pure, high-traffic components every reported number depends on (LLM-output label normalization, atomic cache writes, few-shot schema validation):

```bash
python tests/test_core.py     # no pytest required
# or:  pytest tests/test_core.py
```

## Source layout (`src/`)

**Core pipeline**
- `bm25_baseline.py` — BM25 Okapi retrieval over the Knowledge Store; 3-tier cache, atomic writes, `--retrieval-only` decoupling. The project's backbone.
- `dense_rerank.py` — BM25 prefilter → mpnet bi-encoder rerank → top-5 (retrieval-strategy ablation).
- `open_web_baseline.py` — DuckDuckGo open-web retrieval (direct claim-as-query, top-5); the open-web rung of the evidence ladder.
- `reprompt_eval.py` — apples-to-apples LLM re-prompting on any retrieval cache; `--prompt-style {plain,standard}`, `--fewshot-file`.
- `gold_baseline.py` — gold-evidence oracle upper bound.
- `llm_baseline.py` — LLM zero-shot, no retrieval.
- `query_generator.py` — query strategies A (direct) / B (keyword) / C (decompose).

**Traditional baseline (no-evidence anchor)**
- `perceptron.py`, `train_perceptron.py`, `features.py`, `dataset.py`, `dummy_baselines.py`

**Evaluation & analysis**
- `make_results_table.py` — scores every prediction cache and regenerates the full results table above (no API calls)
- `evaluate_retrieval.py` (URL Recall@5), `evaluate_text_overlap.py` (text-overlap recall + permutation null)
- `analyze_gold_gap.py` — oracle gap decomposition (quantitative)
- `prep_gold_gap_cases.py` — samples gold-wrong cases for qualitative 4-way classification
- `error_analysis.py`, `eval_week1_150.py`, `eval_week1_official_500.py`

**Few-shot ICL**
- `fewshot_picker_clean.py` — Leakage-free few-shot example selector. Draws examples only from `internal_train` and strictly excludes any claims that overlap with the evaluation splits.

**Data prep & infra**
- `split_data.py`, `sample_stratified.py`, `explore_averitec.py`, `create_mini_dev.py`, `download_ks.py`, `download_mini.py`
- `warmup_dev_corpus.py` (parallel corpus tokenization), `warmup_progress.py`
- `run_llm_150.py`, `run_gold_150.py` — sample150 wrappers.

## Notes

- All LLM calls use `gpt-4.1-mini` at temperature 0. Some run-to-run variation (~±0.01 Macro-F1) remains, concentrated in the sparse Conflicting class; differences ≤ 0.01 are treated as noise.
- Ablations follow a **Rule of 150**: piloted on the stratified `internal_dev_sample150.json` before a single confirmation run on dev500.
- Large regeneratable caches (BM25 corpus/rankings, DDG snippets, large retrieval caches) are git-ignored; the small prediction/eval caches that back the reported numbers are committed under `results/`.
