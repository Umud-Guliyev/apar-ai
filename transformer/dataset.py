import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from transformer.config import TRAIN_FILE, TEXT_COLUMN, LABEL_COLUMN

import torch

class FeedbackDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length,
        inference=False
    ):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inference = inference

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        row = self.data.iloc[index]

        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )


        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }


        if not self.inference:
            item["labels"] = torch.tensor(
                row["label_id"],
                dtype=torch.long
            )


        return item



def load_data():

    df = pd.read_csv(TRAIN_FILE)

    df["tag"] = df["tag"].fillna("")


    df["text"] = (
        df[TEXT_COLUMN]
        + " "
        + df["tag"]
    )


    encoder = LabelEncoder()


    df["label_id"] = encoder.fit_transform(
        df[LABEL_COLUMN]
    )


    return df, encoder