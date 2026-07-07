import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.optim import AdamW

from transformer.config import (
    MAX_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)

from transformer.dataset import load_data, FeedbackDataset
from transformer.tokenizer import create_tokenizer
from transformer.model import create_model


def train():

    df, encoder = load_data()

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label_id"]
    )

    tokenizer = create_tokenizer()

    train_dataset = FeedbackDataset(
        train_df,
        tokenizer,
        MAX_LENGTH
    )

    val_dataset = FeedbackDataset(
        val_df,
        tokenizer,
        MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = create_model()
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for batch in train_loader:

            optimizer.zero_grad()

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            outputs = model(**batch)

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1} loss:",
            total_loss / len(train_loader)
        )

        evaluate(model, val_loader, device)


def evaluate(model, loader, device):

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for batch in loader:

            batch = {
                k:v.to(device)
                for k,v in batch.items()
            }

            outputs = model(**batch)

            prediction = torch.argmax(
                outputs.logits,
                dim=1
            )

            preds.extend(
                prediction.cpu().numpy()
            )

            labels.extend(
                batch["labels"].cpu().numpy()
            )

    score = f1_score(
        labels,
        preds,
        average="macro"
    )

    print(
        "Validation Macro F1:",
        score
    )


if __name__ == "__main__":
    train()