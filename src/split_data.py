import os
import json
import random
from collections import Counter
from sklearn.model_selection import train_test_split

# Set random seed for strict reproducibility
random.seed(42)

INTERNAL_DEV_SIZE = 500


def split_official_train():
    """
    Splits the official AVeriTeC train.json into:
      - internal_train.json : used for Perceptron training (Week 1),
                               Few-Shot ICL example retrieval (Week 3-4)
      - internal_dev.json   : used for tuning/ablation during Weeks 2-4
                               (query strategy, chunking, top-k, prompt design)

    The split is stratified by label to preserve the original class
    distribution (especially important for the rare Conflicting Evidence/
    Cherrypicking class, which makes up only 6.4% of the data).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    train_file = os.path.join(project_dir, 'data', 'train.json')

    print(f"Loading official train set from: {train_file}")
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total claims in official train.json: {len(data)}")

    labels = [item.get('label') for item in data]

    internal_train_data, internal_dev_data = train_test_split(
        data,
        test_size=INTERNAL_DEV_SIZE,
        random_state=42,
        stratify=labels
    )

    print(f"\nInternal Train: {len(internal_train_data)} claims")
    print(f"Internal Dev:   {len(internal_dev_data)} claims")

    print("\nLabel distribution check:")
    full_counts = Counter(labels)
    train_counts = Counter([x.get('label') for x in internal_train_data])
    dev_counts = Counter([x.get('label') for x in internal_dev_data])

    label_order = ["Refuted", "Supported",
                   "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]

    print(f"  {'Label':<38} {'Full':>8} {'Internal Train':>16} {'Internal Dev':>14}")
    for label in label_order:
        full_n = full_counts.get(label, 0)
        train_n = train_counts.get(label, 0)
        dev_n = dev_counts.get(label, 0)
        print(f"  {label:<38} {full_n:>8} {train_n:>16} {dev_n:>14}")

    internal_train_file = os.path.join(
        project_dir, 'data', 'internal_train.json')
    internal_dev_file = os.path.join(
        project_dir, 'data', 'internal_dev.json')

    with open(internal_train_file, 'w', encoding='utf-8') as f:
        json.dump(internal_train_data, f, indent=2, ensure_ascii=False)

    with open(internal_dev_file, 'w', encoding='utf-8') as f:
        json.dump(internal_dev_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {internal_train_file}")
    print(f"Saved: {internal_dev_file}")


if __name__ == '__main__':
    split_official_train()
