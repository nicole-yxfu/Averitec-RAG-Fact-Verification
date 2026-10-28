"""
eval_week1_official_500.py — Week 1 traditional baseline on the 500-claim
Official Dev split. Produces the canonical Averaged Perceptron Macro-F1
number for the Week 6 final results table (Row 3).
"""
import os
import random
from sklearn.metrics import f1_score, classification_report

from dataset import read_averitec_json
from features import FeatureMapping, get_instances
from perceptron import train_averaged, predict

random.seed(42)

TARGET_NAMES = ["Supported (0)", "Refuted (1)", "NEE (2)", "Conflicting (3)"]


def run_week1_on_official_500():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Official dev (500 claims) is a fully held-out test set, so it is safe
    # to train on the full train.json (3068 claims) here — gives the
    # Perceptron its strongest learning signal.
    train_file = os.path.join(project_dir, 'data', 'train.json')

    # Evaluate on the official 500-claim dev split
    dev_file = os.path.join(project_dir, 'data', 'dev.json')

    print("Loading data...")
    train_texts = read_averitec_json(train_file)
    dev_texts = read_averitec_json(dev_file)

    print("Extracting features (Unigrams, Bigrams, Metadata)...")
    fm = FeatureMapping()
    train_instances = get_instances(train_texts, fm)
    fm.frozen = True

    # Filter out any dev.json rows that lack a label (defensive)
    dev_texts_filtered = [t for t in dev_texts if t.label is not None]
    dev_instances = get_instances(dev_texts_filtered, fm)

    print("Training Averaged Perceptron on FULL train.json (10 iterations)...")
    weights = train_averaged(train_instances, fm, num_iterations=10)

    print("Evaluating on official dev.json (500 claims)...")
    y_true, y_pred = [], []
    for instance in dev_instances:
        y_true.append(instance.label)
        y_pred.append(predict(instance, weights))

    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=[0, 1, 2, 3])

    print(f"\n{'=' * 60}")
    print("Week 1: Traditional Machine Learning (Averaged Perceptron)")
    print(f"Tested on: official dev.json (500)")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=[
          0, 1, 2, 3], target_names=TARGET_NAMES, zero_division=0))


if __name__ == '__main__':
    run_week1_on_official_500()
