import os
import random
from sklearn.metrics import f1_score, classification_report

from dataset import read_averitec_json
from features import FeatureMapping, get_instances
from perceptron import train_averaged, predict

# Set random seed for strict reproducibility
random.seed(42)

TARGET_NAMES = ["Supported (0)", "Refuted (1)", "NEE (2)", "Conflicting (3)"]


def evaluate(instances, weights, split_name):
    """Evaluates a trained model on a set of instances and prints results."""
    y_true, y_pred = [], []
    for instance in instances:
        y_true.append(instance.label)
        y_pred.append(predict(instance, weights))

    # labels=[0,1,2,3] ensures true 4-class Macro-F1 even if a class is
    # absent from a small subset (e.g. Conflicting at ~6.4% prevalence).
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=[0, 1, 2, 3])

    print(f"\n{'=' * 55}")
    print(f"Results: {split_name}")
    print(f"{'=' * 55}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3],
        target_names=TARGET_NAMES, zero_division=0))

    return macro_f1


def run_training(train_file, label, dev_splits):
    """
    Trains one Averaged Perceptron model on train_file and evaluates it
    on every (name, path) pair in dev_splits.

    Parameters
    ----------
    train_file : str   Path to the training JSON file.
    label      : str   Human-readable label for this training run.
    dev_splits : list  [(split_name, dev_file_path), ...]
    """
    print(f"\n{'#' * 55}")
    print(f"Training run: {label}")
    print(f"{'#' * 55}")

    # 1. Load training data
    print(f"Loading training data from: {os.path.basename(train_file)}")
    train_texts = read_averitec_json(train_file)

    # 2. Extract features
    print("Extracting features (unigram + bigram + speaker + source)...")
    fm = FeatureMapping()
    train_instances = get_instances(train_texts, fm)
    fm.frozen = True   # freeze to prevent leakage into dev sets
    print(f"Feature vocabulary size: {len(fm.map)} unique features.")

    # 3. Train
    # Re-seed immediately before each training run. train_averaged() calls
    # random.shuffle() once per iteration, so when this function is invoked
    # more than once in a single process (internal_train run, then full-train
    # run), the second run would otherwise start from an RNG state already
    # advanced by the first — making the averaged-perceptron result depend on
    # execution order (full-train scored 0.3515 here vs 0.3546 from a clean
    # single run / error_analysis.py). Re-seeding makes each run reproducible
    # and harness-independent: full train → official_dev now stably yields
    # 0.3546, matching the saved perceptron_predictions.csv.
    random.seed(42)
    print("\nTraining Multi-Class Averaged Perceptron (10 iterations)...")
    weights = train_averaged(train_instances, fm, num_iterations=10)

    # 4. Evaluate on each dev split
    results = {}
    for split_name, dev_file in dev_splits:
        print(f"\nLoading dev split: {split_name}...")
        dev_texts = read_averitec_json(dev_file)
        dev_instances = get_instances(dev_texts, fm)
        macro_f1 = evaluate(dev_instances, weights, split_name)
        results[split_name] = macro_f1

    return results


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')

    # -----------------------------------------------------------------------
    # Run 1: Train on internal_train → evaluate on internal_dev + official_dev
    #
    # Purpose: fills in the "Internal Dev" column of the Week 5 results table,
    # and provides a fair internal/official comparison on the same model.
    # internal_train has 2568 claims (500 held out as internal_dev).
    # -----------------------------------------------------------------------
    results_internal = run_training(
        train_file=os.path.join(data_dir, 'internal_train.json'),
        label='internal_train (2568 claims) → internal_dev + official_dev',
        dev_splits=[
            ('Internal Dev  (500)', os.path.join(data_dir, 'internal_dev.json')),
            ('Official Dev  (500)', os.path.join(data_dir, 'dev.json')),
        ]
    )

    # -----------------------------------------------------------------------
    # Run 2: Train on full train.json → evaluate on official_dev
    #
    # Purpose: the formal Traditional Baseline for the Week 5 results table.
    # This is the previously reported 0.3546 result, re-run with labels=[0,1,2,3]
    # and random.seed(42) for strict reproducibility.
    # -----------------------------------------------------------------------
    results_full = run_training(
        train_file=os.path.join(data_dir, 'train.json'),
        label='Full train (3068 claims) → official_dev  [formal Week 5 baseline]',
        dev_splits=[
            ('Official Dev  (500)', os.path.join(data_dir, 'dev.json')),
        ]
    )

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 55}")
    print("Summary — Averaged Perceptron Macro-F1")
    print(f"{'=' * 55}")
    print(f"{'Split':<30} {'Internal train':>16} {'Full train':>12}")
    print(f"{'-'*55}")
    print(f"{'Internal Dev (500)':<30} "
          f"{results_internal.get('Internal Dev  (500)', float('nan')):>16.4f} "
          f"{'—':>12}")
    print(f"{'Official Dev (500)':<30} "
          f"{results_internal.get('Official Dev  (500)', float('nan')):>16.4f} "
          f"{results_full.get('Official Dev  (500)', float('nan')):>12.4f}")
    print(f"{'=' * 55}")
    print("Note: 'Full train → Official Dev' is the formal Week 5 baseline.")
