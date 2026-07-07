from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data import load_train
from features import build_vectorizer
from config import SEED


def main():

    train = load_train()

    X = train["feedback"]

    y = train["label"]

    pipeline = Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("model", LinearSVC()),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    print("=" * 50)

    print("5 Fold Cross Validation")

    print("=" * 50)

    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: {score:.6f}")

    print()

    print(f"Mean : {scores.mean():.6f}")

    print(f"Std  : {scores.std():.6f}")


if __name__ == "__main__":
    main()