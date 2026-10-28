import json
import os
import random

# Set random seed consistent with the project to ensure reproducibility
random.seed(42)


def create_mini_dev():
    input_file = "data/internal_dev.json"
    output_file = "data/internal_dev_mini.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group by label to ensure the sampled distribution remains balanced (stratified sampling)
    labels_dict = {}
    for item in data:
        label = item.get("label")
        if label not in labels_dict:
            labels_dict[label] = []
        labels_dict[label].append(item)

    mini_data = []
    # Sample 50 out of 500 records, i.e., extract a 10% proportion
    for label, items in labels_dict.items():
        sample_size = max(1, int(len(items) * 0.10))
        mini_data.extend(random.sample(items, sample_size))

    # Truncate to exactly 50 records (if there's an overflow due to rounding)
    if len(mini_data) > 50:
        mini_data = mini_data[:50]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mini_data, f, indent=4, ensure_ascii=False)

    print(
        f"Success! Extracted {len(mini_data)} records from {len(data)} total records, saved to {output_file}")
    for label, items in labels_dict.items():
        count = sum(1 for x in mini_data if x.get("label") == label)
        print(f"  - Label '{label}': {count} records")


if __name__ == "__main__":
    create_mini_dev()
