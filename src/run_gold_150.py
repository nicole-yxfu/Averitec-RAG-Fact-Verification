"""
run_gold_150.py — Gold-evidence upper bound on internal_dev_sample150.

Thin wrapper around gold_baseline.run_gold_baseline so the 150-claim subset
uses the IDENTICAL gold-QA-pair extraction and prompt as the official_dev
run. The earlier hand-rolled version of this script hit the duplicate-claim
indexing bug — it assumed sample150[i] == train.json[i], but the 150-claim
file is a stratified sample, so e.g. sample150[0] is actually train[2368].
That fed every claim somebody else's gold QA pairs and dragged Macro-F1 to
0.10. Calling gold_baseline directly reuses build_text_id_to_source_idx,
which resolves each subset row to its correct train.json source index.

Output cache: results/gold_baseline_internal_dev_sample150_with_metadata_cache.json
Usage:
    python src/run_gold_150.py
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

from gold_baseline import run_gold_baseline


if __name__ == "__main__":
    data_file = os.path.join(PROJECT_DIR, "data", "internal_dev_sample150.json")
    # sample150 was drawn from internal_dev (which itself came from train.json),
    # so claim_idx resolution must happen against train.json.
    source_idx_file = os.path.join(PROJECT_DIR, "data", "train.json")
    results_dir = os.path.join(PROJECT_DIR, "results")

    run_gold_baseline(
        split_name="internal_dev_sample150",
        data_file=data_file,
        source_idx_file=source_idx_file,
        backend="langchain",
        sample=None,
        use_metadata=True,
        results_dir=results_dir,
    )
