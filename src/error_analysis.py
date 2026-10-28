import os
import csv
import random

from perceptron import train_averaged, predict
from features import FeatureMapping, get_instances
from dataset import read_averitec_json

# Set random seed for reproducibility
random.seed(42)


def run_deep_error_analysis():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    train_file = os.path.join(project_dir, 'data', 'train.json')
    dev_file = os.path.join(project_dir, 'data', 'dev.json')

    print("Loading data and extracting features (including metadata)...")
    train_texts = read_averitec_json(train_file)
    dev_texts = read_averitec_json(dev_file)

    fm = FeatureMapping()
    train_instances = get_instances(train_texts, fm)
    fm.frozen = True
    dev_instances = get_instances(dev_texts, fm)

    print(f"Feature vocabulary size: {len(fm.map)}")

    print("Training Averaged Perceptron (10 iterations)...")
    weights = train_averaged(train_instances, fm, num_iterations=10)

    # Reverse mapping to retrieve string feature names from IDs
    reverse_map = {v: k for k, v in fm.map.items()}
    TARGET_NAMES = ["Supported (0)", "Refuted (1)",
                    "NEE (2)", "Conflicting (3)"]

    print("1. Error Analysis: Feature Weight Inspection")

    # Inspect top features for each class
    for class_id in range(4):
        class_name = TARGET_NAMES[class_id]
        class_weights = weights[class_id]

        word_weights = []
        for feature_index, weight in enumerate(class_weights):
            if feature_index in reverse_map:
                word_weights.append((reverse_map[feature_index], weight))

        # Sort by weight descending
        word_weights.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 10 strongest features for class '{class_name}':")
        for word, weight in word_weights[:10]:
            print(f"  {word:35} : {weight:.4f}")

    print("2. Qualitative Error Sampling")

    conflicting_failed = []

    # Collect severe misclassifications
    for i in range(len(dev_instances)):
        instance = dev_instances[i]
        text = dev_texts[i]
        prediction = predict(instance, weights)

        # Target instances where True=Conflicting(3) but Pred=Refuted(1) or Supported(0)
        if instance.label == 3 and prediction in [0, 1]:
            wrong_score = sum(weights[prediction][f_idx]
                              for f_idx in instance.features)
            conflicting_failed.append(
                (wrong_score, text.original_claim, TARGET_NAMES[prediction]))

    # Sort by confidence score descending
    conflicting_failed.sort(key=lambda x: x[0], reverse=True)

    print("Top 5 most confident misclassifications (True: Conflicting, Pred: Refuted/Supported):")
    for score, claim, wrong_pred in conflicting_failed[:5]:
        print(
            f"\n  [Misclassified as {wrong_pred} | Confidence Score: {score:.2f}]")
        print(f"  Claim: {claim}")

    # Save instance-level predictions
    results_dir = os.path.join(project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    perceptron_file = os.path.join(results_dir, 'perceptron_predictions.csv')

    with open(perceptron_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['claim_id', 'true_label', 'predicted_label'])
        for i in range(len(dev_instances)):
            instance = dev_instances[i]
            text = dev_texts[i]
            pred = predict(instance, weights)
            writer.writerow([text.text_id, instance.label, pred])

    print(f"\nPredictions successfully saved to: {perceptron_file}")


if __name__ == '__main__':
    run_deep_error_analysis()
