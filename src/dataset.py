import os
import json
import string

# Dictionary mapping string labels to numerical IDs
LABEL_MAP = {
    "Supported": 0,
    "Refuted": 1,
    "Not Enough Evidence": 2,
    "Conflicting Evidence/Cherrypicking": 3
}


class Token:
    """Represents a single word/token."""

    def __init__(self, form):
        self.form = form


class Text:
    """Represents a single claim instance with text, labels, and metadata."""

    def __init__(self, text_id, raw_label, mapped_label, tokens, original_claim, speaker, reporting_source):
        self.text_id = text_id
        self.raw_label = raw_label
        self.label = mapped_label
        self.tokens = tokens
        self.original_claim = original_claim
        self.predicted_label = None

        # Extended AVeriTeC metadata features
        self.speaker = speaker
        self.reporting_source = reporting_source


def read_averitec_json(filename):
    """
    Reads the AVeriTeC JSON dataset, cleans text into tokens, 
    extracts metadata, and converts instances into Text objects.
    """
    texts = []

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i, item in enumerate(data):
        text_id = f"claim_{i}"
        raw_label = item.get("label", "")
        original_claim = item.get("claim", "")

        # Extract metadata, assigning a default value if missing
        speaker = item.get("speaker") or "none"
        reporting_source = item.get("reporting_source") or "none"

        mapped_label = LABEL_MAP.get(raw_label, None)

        tokens = []
        # Defensive programming: process tokens only if claim is not empty
        if original_claim:
            for word in original_claim.split():
                word = word.strip(string.punctuation)
                if not word:
                    continue
                tokens.append(Token(word))

        text_instance = Text(
            text_id, raw_label, mapped_label, tokens,
            original_claim, speaker, reporting_source
        )
        texts.append(text_instance)

    return texts


if __name__ == '__main__':
    # Locate data file relative to the script directory
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(script_dir, 'data', 'train.json')

    print("Loading AVeriTeC dataset for preprocessing checks...")
    texts = read_averitec_json(train_file)
    print(f"Successfully loaded {len(texts)} instances.")

    # Initialize count dictionaries
    raw_counts = {}
    mapped_counts = {}

    for text in texts:
        raw_counts[text.raw_label] = raw_counts.get(text.raw_label, 0) + 1
        if text.label is not None:
            mapped_counts[text.label] = mapped_counts.get(text.label, 0) + 1

    print("1. Raw Label Distribution")

    for label, count in raw_counts.items():
        print(f"  - '{label}': {count}")

    print("2. Mapped 4-class Distribution")

    for label_id, count in mapped_counts.items():
        print(f"  - Class ID {label_id}: {count}")

    print("3. Sample Validation (first instance)")

    print(f"  [Claim]: {texts[0].original_claim}")
    print(
        f"  [Raw Label]: {texts[0].raw_label} -> [Mapped ID]: {texts[0].label}")
