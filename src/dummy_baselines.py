import os
import csv
import random
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from dataset import read_averitec_json

# Set random seeds for strict reproducibility
random.seed(42)
np.random.seed(42)


def run_dummy_baselines():
    """
    Calculates and saves lower-bound statistical baselines (Majority and Random)
    to establish a performance floor for the AVeriTeC classification task.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    dev_file = os.path.join(project_dir, 'data', 'dev.json')

    print("Loading Dev dataset for baseline evaluation...")
    dev_texts = read_averitec_json(dev_file)

    # Extract ground truth labels, ignoring instances without labels
    y_true = [text.label for text in dev_texts if text.label is not None]

    print("Baseline:  Majority Class (Always Predict 'Refuted')")

    # Class ID 1 corresponds to 'Refuted' (the majority class)
    y_pred_majority = [1] * len(y_true)

    maj_macro_f1 = f1_score(y_true, y_pred_majority, average='macro', labels=[
                            0, 1, 2, 3], zero_division=0)
    maj_accuracy = accuracy_score(y_true, y_pred_majority)

    # Explaining the metric discrepancy professionally
    print(
        f"Accuracy : {maj_accuracy:.4f} (Artificially high due to class imbalance)")
    print(f"Macro-F1 : {maj_macro_f1:.4f} (Reflects true structural failure)")

    print("Baseline: Random Guess (4-class)")

    random_f1_scores = []

    # Run 1000 iterations to calculate a stable mean and standard deviation
    for _ in range(1000):
        y_pred_random = [random.choice([0, 1, 2, 3])
                         for _ in range(len(y_true))]
        random_f1_scores.append(f1_score(
            y_true, y_pred_random, average='macro', labels=[0, 1, 2, 3], zero_division=0))

    rand_macro_f1_mean = np.mean(random_f1_scores)
    rand_macro_f1_std = np.std(random_f1_scores)

    print(f"Macro-F1 : {rand_macro_f1_mean:.4f} ± {rand_macro_f1_std:.4f}")

    # ==========================================
    # Data Export Module
    # ==========================================
    results_dir = os.path.join(project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save Majority Baseline Predictions
    majority_file = os.path.join(results_dir, 'majority_predictions.csv')
    with open(majority_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['claim_id', 'true_label', 'predicted_label'])
        valid_texts = [t for t in dev_texts if t.label is not None]
        for text, pred in zip(valid_texts, y_pred_majority):
            writer.writerow([text.text_id, text.label, pred])
    print(f"\nMajority predictions successfully saved to: {majority_file}")

    # Save Random Baseline Predictions
    random_file = os.path.join(results_dir, 'random_predictions.csv')
    with open(random_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['claim_id', 'true_label', 'predicted_label'])
        for text, pred in zip(valid_texts, y_pred_random):
            writer.writerow([text.text_id, text.label, pred])
    print(f"Random predictions successfully saved to: {random_file}")


if __name__ == '__main__':
    run_dummy_baselines()
