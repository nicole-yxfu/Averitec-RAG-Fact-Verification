import os
import json
from collections import Counter


def explore_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    train_file = os.path.join(project_dir, 'data', 'train.json')

    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("1. Dataset Marco-scale")

    print(f"Total claims: {len(data)}")

    print("2. Available Feature Fields")

    if len(data) > 0:
        keys = list(data[0].keys())
        for key in keys:
            print(f"  - {key}")

    print("3. Raw Label Distribution")

    labels = [item.get('label', 'MISSING') for item in data]
    label_counts = Counter(labels)
    for label, count in label_counts.most_common():
        print(f"  - {label}: {count} ({count/len(data)*100:.1f}%)")

    print("4. Speaker and Reporting Source Analysis")

    # Analyze Speakers
    speakers = [item.get('speaker') for item in data if item.get('speaker')]
    speaker_counts = Counter(speakers)
    print(
        f"[Speaker]: Total {len(speaker_counts)} unique speakers/organizations.")
    print("Top 3 most active speakers:")
    for speaker, count in speaker_counts.most_common(3):
        print(f"  * {speaker}: {count}")

    # Analyze Reporting Sources
    sources = [item.get('reporting_source')
               for item in data if item.get('reporting_source')]
    source_counts = Counter(sources)
    print(f"\n[Reporting Source]: Total {len(source_counts)} unique sources.")
    print("Top 3 most active reporting sources:")
    for source, count in source_counts.most_common(3):
        print(f"  * {source}: {count}")

    print("5. Complete Data Structure Sample")

    sample = data[0]
    print(f"  [Claim]: {sample.get('claim')}")
    print(f"  [Label]: {sample.get('label')}")
    print(f"  [Date]: {sample.get('claim_date')}")
    print(f"  [Speaker]: {sample.get('speaker')}")
    print(f"  [Source]: {sample.get('reporting_source')}")
    print(
        f"  [Justification (First 100 chars)]: {sample.get('justification', '')[:100]}...")


if __name__ == '__main__':
    explore_data()
