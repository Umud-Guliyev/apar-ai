import torch
import pandas as pd
import joblib

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from transformer.config import (
    MAX_LENGTH,
)

from transformer.dataset import FeedbackDataset


TEST_FILE = "data/test.csv"
OUTPUT_FILE = "submission.csv"


def predict():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading test data...")

    test_df = pd.read_csv(TEST_FILE)

    test_df["tag"] = test_df["tag"].fillna("")

    test_df["text"] = (
        test_df["feedback"]
        + " "
        + test_df["tag"]
    )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        "artifacts/tokenizer"
    )

    print("Loading label encoder...")

    encoder = joblib.load(
        "artifacts/label_encoder.pkl"
    )

    dataset = FeedbackDataset(
        test_df,
        tokenizer,
        MAX_LENGTH,
        inference=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
    )

    print("Loading ensemble models...")

    ensemble_logits = None

    for fold in range(1, 6):

        print(f"Fold {fold}")

        model = AutoModelForSequenceClassification.from_pretrained(
            f"artifacts/fold_{fold}"
        )

        model.to(device)
        model.eval()

        fold_logits = []

        with torch.no_grad():

            for batch in loader:

                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                }

                outputs = model(**batch)

                fold_logits.append(
                    outputs.logits.cpu()
                )

        fold_logits = torch.cat(fold_logits)

        if ensemble_logits is None:

            ensemble_logits = fold_logits

        else:

            ensemble_logits += fold_logits

    ensemble_logits /= 5

    predictions = torch.argmax(
        ensemble_logits,
        dim=1,
    ).numpy()

    labels = encoder.inverse_transform(
        predictions
    )

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "label": labels,
        }
    )

    submission.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSubmission created successfully!")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    predict()