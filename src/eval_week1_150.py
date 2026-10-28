"""
eval_week1_150.py — Week 1 traditional baseline (Averaged Perceptron),
evaluated on the 150-claim stratified ablation subset so its Macro-F1
number is comparable to the Week 2+ ablation results that all run on s150.
"""
import os
import random
from sklearn.metrics import f1_score, classification_report

from dataset import read_averitec_json
from features import FeatureMapping, get_instances
from perceptron import train_averaged, predict

# Lock the RNG seed for strict reproducibility
random.seed(42)

TARGET_NAMES = ["Supported (0)", "Refuted (1)", "NEE (2)", "Conflicting (3)"]


def run_week1_on_150():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Train on internal_train (2568 claims, disjoint from s150)
    train_file = os.path.join(project_dir, 'data', 'internal_train.json')
    # Evaluate on the 150-claim stratified ablation subset
    dev_file = os.path.join(project_dir, 'data', 'internal_dev_sample150.json')

    print("Loading data...")
    train_texts = read_averitec_json(train_file)
    dev_texts = read_averitec_json(dev_file)

    print("Extracting features (Unigrams, Bigrams, Metadata)...")
    fm = FeatureMapping()
    train_instances = get_instances(train_texts, fm)
    fm.frozen = True
    dev_instances = get_instances(dev_texts, fm)

    print("Training Averaged Perceptron (10 iterations)...")
    weights = train_averaged(train_instances, fm, num_iterations=10)

    print("Evaluating on internal_dev_sample150...")
    y_true, y_pred = [], []
    for instance in dev_instances:
        y_true.append(instance.label)
        y_pred.append(predict(instance, weights))

    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=[0, 1, 2, 3])

    print(f"\n{'=' * 60}")
    print("Week 1: Traditional Machine Learning (Averaged Perceptron)")
    print(f"Tested on: internal_dev_sample150")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=[
          0, 1, 2, 3], target_names=TARGET_NAMES, zero_division=0))


if __name__ == '__main__':
    run_week1_on_150()
