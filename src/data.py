import pandas as pd

from config import TRAIN_PATH, TEST_PATH


def load_train():
    return pd.read_csv(TRAIN_PATH)


def load_test():
    return pd.read_csv(TEST_PATH)