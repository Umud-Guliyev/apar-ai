import torch
import pandas as pd
import joblib

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformer.config import (
    MAX_LENGTH,
    MODEL_NAME
)

from transformer.dataset import FeedbackDataset
from transformer.model import create_model


TEST_FILE = "data/test.csv"
OUTPUT_FILE = "submission.csv"


def predict():

    device = torch.device("cpu")

    print("Loading test data...")

    test_df = pd.read_csv(TEST_FILE)


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
        inference=True
    )


    loader = DataLoader(
        dataset,
        batch_size=32
    )


    print("Loading model...")

    model = create_model()

    model.load_state_dict(
        torch.load(
            "artifacts/best_model.pt",
            map_location=device
        )
    )

    model.to(device)

    model.eval()


    predictions = []


    with torch.no_grad():

        for batch in loader:

            batch = {
                k:v.to(device)
                for k,v in batch.items()
            }

            outputs = model(**batch)

            preds = torch.argmax(
                outputs.logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )


    labels = encoder.inverse_transform(
        predictions
    )


    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "label": labels
        }
    )


    submission.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print("Submission created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    predict()