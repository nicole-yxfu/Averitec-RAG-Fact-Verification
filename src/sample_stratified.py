"""
sample_stratified.py — Creates a label-stratified subset of internal_dev
or official_dev, for fast ablation experiments (Strategy A vs B comparison,
top-k sweeps, etc.) without re-running the full 500-claim set each time.

Why stratified, not "first N": AVeriTeC's label distribution is skewed
(Refuted ~50%+, Conflicting ~6-8%). A naive prefix slice (texts[:150]) can
under-represent or even drop the rare classes, making Macro-F1 comparisons
between configurations noisy and unreliable. Stratified sampling preserves
the original class proportions at any sample size.

This script does NOT touch internal_dev.json / dev.json — it writes a NEW
file alongside them, so the original 500-claim files stay untouched for
the Week 6 official 8-system comparison table (which must use the full set).

Usage:
  python src/sample_stratified.py --source internal_dev --n 150
  python src/sample_stratified.py --source official_dev --n 150 --seed 42

Then point bm25_baseline.py at the sample via --data-file:
  python src/bm25_baseline.py --split internal_dev \
      --data-file data/internal_dev_sample150.json \
      --split-label internal_dev_sample150 \
      --sample -1
"""

import os
import json
import argparse
from collections import Counter

from sklearn.model_selection import train_test_split


def stratified_sample(data: list, n: int, seed: int) -> list:
    """Selects n records from data, preserving label proportions."""
    if n >= len(data):
        return data
    labels = [item.get('label') for item in data]
    selected, _ = train_test_split(
        data, train_size=n, random_state=seed, stratify=labels
    )
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Create a label-stratified subset of internal_dev or official_dev."
    )
    parser.add_argument("--source", choices=["internal_dev", "official_dev"],
                        required=True)
    parser.add_argument("--n", type=int, default=150,
                        help="Target sample size (default: 150)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')

    source_file = os.path.join(data_dir, f"{args.source}.json")
    output_file = os.path.join(data_dir, f"{args.source}_sample{args.n}.json")

    print(f"Loading: {source_file}")
    with open(source_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Total claims: {len(data)}")

    sampled = stratified_sample(data, args.n, args.seed)

    # Re-order to match the ORIGINAL file order (train_test_split shuffles).
    # Purely cosmetic — easier to eyeball/cross-reference — does not affect
    # which claims were selected or their labels.
    sampled_ids = {id(item) for item in sampled}
    ordered_sampled = [item for item in data if id(item) in sampled_ids]

    # ── Verify the label distribution survived sampling ─────────────────────
    full_dist = Counter(item.get('label') for item in data)
    sample_dist = Counter(item.get('label') for item in ordered_sampled)

    print(f"\nLabel distribution check:")
    print(f"  {'Label':<38} {'Full %':>8} {'Sample %':>10} {'Sample n':>10}")
    for label in full_dist:
        full_pct = 100 * full_dist[label] / len(data)
        sample_pct = (100 * sample_dist.get(label, 0) / len(ordered_sampled)
                      if ordered_sampled else 0)
        print(f"  {label:<38} {full_pct:>7.1f}% {sample_pct:>9.1f}% "
              f"{sample_dist.get(label, 0):>10}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ordered_sampled, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(ordered_sampled)} claims to: {output_file}")
    print(f"\nUse with bm25_baseline.py via:")
    print(f"  python src/bm25_baseline.py --split {args.source} \\")
    print(f"      --data-file {output_file} \\")
    print(f"      --split-label {args.source}_sample{args.n} \\")
    print(f"      --sample -1")


if __name__ == '__main__':
    main()
