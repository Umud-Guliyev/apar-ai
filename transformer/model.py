from transformers import AutoModelForSequenceClassification

from transformer.config import MODEL_NAME, NUM_LABELS


def create_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    return model