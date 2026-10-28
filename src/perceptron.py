import random
import pickle

# The AVeriTeC dataset is formulated as a 4-class classification task
NUM_CLASSES = 4


def predict(instance, weights):
    """
    Predicts the class label for a given instance using the current weights.
    Calculates the score (dot product) for each class and returns the argmax.
    """
    best_class = 0
    max_score = float('-inf')

    for c in range(NUM_CLASSES):
        score = 0
        for feature_index in instance.features:
            score += weights[c][feature_index]

        if score > max_score:
            max_score = score
            best_class = c

    return best_class


def train_averaged(instances, feature_mapping, num_iterations):
    """
    Trains a multi-class Averaged Perceptron model.
    Uses the survival time trick for efficient calculation of averaged weights.
    """
    num_features = len(feature_mapping.map)

    # Initialize weight matrices: weights[class_id][feature_index]
    weights = [[0] * num_features for _ in range(NUM_CLASSES)]
    accumulated_weights = [[0] * num_features for _ in range(NUM_CLASSES)]

    total_steps = num_iterations * len(instances)
    current_step = 0

    for iteration in range(num_iterations):
        random.shuffle(instances)
        correct_predictions = 0

        for instance in instances:
            current_step += 1
            prediction = predict(instance, weights)

            # Update weights if the prediction is incorrect
            if prediction != instance.label:
                true_class = instance.label
                pred_class = prediction

                # Calculate survival time for the efficient averaging trick
                survival_time = total_steps - current_step + 1

                for feature_index in instance.features:
                    # Reward the correct class
                    weights[true_class][feature_index] += 1
                    accumulated_weights[true_class][feature_index] += survival_time

                    # Penalize the incorrectly predicted class
                    weights[pred_class][feature_index] -= 1
                    accumulated_weights[pred_class][feature_index] -= survival_time
            else:
                correct_predictions += 1

        accuracy = 100 * correct_predictions / len(instances)
        print(
            f"Iteration {iteration + 1}/{num_iterations} | Training Accuracy: {accuracy:.2f}%")

    # Calculate the final averaged weights
    final_averaged_weights = []
    for c in range(NUM_CLASSES):
        class_weights = [w / total_steps for w in accumulated_weights[c]]
        final_averaged_weights.append(class_weights)

    return final_averaged_weights


def save_model(filename, weights, feature_mapping):
    """Saves the trained model weights and feature mapping to disk."""
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)
        pickle.dump(feature_mapping, f)


def load_model(filename):
    """Loads a trained model and its feature mapping from disk."""
    with open(filename, 'rb') as f:
        weights = pickle.load(f)
        feature_mapping = pickle.load(f)
    return weights, feature_mapping
