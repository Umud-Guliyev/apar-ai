from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.svm import LinearSVC

from config import TEST_SIZE, SEED
from data import load_train
from features import build_vectorizer

from models import build_model

def main():

    train = load_train()

    X = train["feedback"]
    y = train["label"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    vectorizer = build_vectorizer()

    X_train = vectorizer.fit_transform(X_train)

    X_valid = vectorizer.transform(X_valid)

    model = build_model()

    model.fit(X_train, y_train)

    pred = model.predict(X_valid)

    score = f1_score(
        y_valid,
        pred,
        average="macro",
    )

    print(f"Macro F1: {score:.6f}")


if __name__ == "__main__":
    main()