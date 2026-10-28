"""
run_llm_150.py — Week 1 LLM Zero-shot on internal_dev_sample150.

Thin wrapper around llm_baseline.run_baseline so the 150-claim subset gets
the IDENTICAL prompt + input format as the official_dev run. The earlier
hand-rolled version of this script accidentally produced a fake "no
evidence" zero-shot (it called build_user_input with passages=[], which
appends a "No evidence passages were found in the knowledge store" note,
collapsing every prediction to NEE and dragging Macro-F1 to 0.11). Calling
llm_baseline directly removes that risk and guarantees apples-to-apples
comparison with the 500-claim official_dev result already in the repo.

Output cache: results/llm_baseline_internal_dev_sample150_with_metadata_cache.json
Usage:
    python src/run_llm_150.py
"""
import os
import sys

# Make the project's src/ importable when run from any CWD.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

from llm_baseline import run_baseline


if __name__ == "__main__":
    data_file = os.path.join(PROJECT_DIR, "data", "internal_dev_sample150.json")
    results_dir = os.path.join(PROJECT_DIR, "results")

    run_baseline(
        split_name="internal_dev_sample150",
        data_file=data_file,
        backend="langchain",
        sample=None,
        use_metadata=True,
        results_dir=results_dir,
    )
