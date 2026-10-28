class FeatureMapping:
    """Maintains a mapping from string features to numerical indices."""

    def __init__(self):
        self.map = {}
        self.frozen = False

    def get_index(self, feature):
        """Returns the index of a feature. Adds it to the map if not frozen."""
        if feature in self.map:
            return self.map[feature]
        if not self.frozen:
            number = len(self.map)
            self.map[feature] = number
            return number
        return None


class Instance:
    """Represents a machine learning instance with a label and numerical features."""

    def __init__(self, label, features):
        self.label = label        # Numerical label (0, 1, 2, or 3)
        self.features = features  # List of numerical feature indices


def extract_features(text):
    """Extracts linguistic and metadata features from a Text object."""
    string_features = []

    # Extract unigrams
    for token in text.tokens:
        string_features.append(f"unigram:{token.form.lower()}")

    # Extract bigrams
    for i in range(len(text.tokens) - 1):
        bigram_str = f"bigram:{text.tokens[i].form.lower()}_{text.tokens[i+1].form.lower()}"
        string_features.append(bigram_str)

    # Incorporate metadata features (Speaker and Reporting Source)
    string_features.append(f"speaker:{text.speaker.lower()}")
    string_features.append(f"source:{text.reporting_source.lower()}")

    return string_features


def get_instance(text, feature_mapping):
    """Converts a Text object into an Instance object using the feature mapping."""
    label = text.label
    string_features = extract_features(text)

    features = []
    for feature in string_features:
        number = feature_mapping.get_index(feature)
        if number is not None:
            features.append(number)

    return Instance(label, features)


def get_instances(texts, feature_mapping):
    """Converts a list of Text objects into a list of Instance objects."""
    instances = []
    for text in texts:
        if text.label is None:
            continue
        instances.append(get_instance(text, feature_mapping))
    return instances
