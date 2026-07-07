import os
import joblib
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

from transformer.evaluate import evaluate
from transformer.config import (
    MAX_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    WARMUP_RATIO,
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
        stratify=df["label_id"],
    )

    tokenizer = create_tokenizer()

    train_dataset = FeedbackDataset(
        train_df,
        tokenizer,
        MAX_LENGTH,
    )

    val_dataset = FeedbackDataset(
        val_df,
        tokenizer,
        MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model()
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * EPOCHS

    warmup_steps = int(
        total_steps * WARMUP_RATIO
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_score = 0.0
    patience_counter = 0

    os.makedirs(
        "artifacts",
        exist_ok=True,
    )

    joblib.dump(
        encoder,
        "artifacts/label_encoder.pkl",
    )

    tokenizer.save_pretrained(
        "artifacts/tokenizer"
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

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            scheduler.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1} loss: {total_loss / len(train_loader):.6f}"
        )

        score = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Validation Macro F1: {score:.6f}"
        )

        if score > best_score:

            best_score = score
            patience_counter = 0

            torch.save(
                model.state_dict(),
                "artifacts/best_model.pt",
            )

            model.save_pretrained(
                "artifacts/model"
            )

            print("✅ Best model saved.")

        else:

            patience_counter += 1

            print(
                f"No improvement ({patience_counter}/{PATIENCE})"
            )

            if patience_counter >= PATIENCE:

                print("\n🛑 Early stopping triggered.")
                break

    print("\nTraining Finished!")
    print(f"Best Validation Macro F1: {best_score:.6f}")


if __name__ == "__main__":
    train()