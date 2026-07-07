from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from config import MODEL_NAME, CLASS_WEIGHT


def build_model():
    if MODEL_NAME == "LinearSVC":
        return LinearSVC(
            class_weight=CLASS_WEIGHT
        )

    if MODEL_NAME == "LogisticRegression":
        return LogisticRegression(
            max_iter=3000,
            random_state=42,
        )

    raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")