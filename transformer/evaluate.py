import torch
from sklearn.metrics import f1_score


def evaluate(model, loader, device):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for batch in loader:

            batch = {
                k: v.to(device)
                for k, v in batch.items()
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

    return f1_score(
        labels,
        preds,
        average="macro"
    )