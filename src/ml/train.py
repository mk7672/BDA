"""Train and validate the initial TF-IDF + Logistic Regression baseline."""
from pathlib import Path
import sys
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

# Support both `python src/ml/train.py` and `python -m src.ml.train`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.preprocess import LABEL_NEGATIVE, normalize_text  # noqa: E402

TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")
MODEL_PATH = Path("models/sentiment_pipeline.joblib")
SEED = 42


def main() -> None:
    # Review text can legitimately be values like "NA" or "N/A". Prevent
    # pandas' default NA-token list from converting those strings to np.nan.
    train_df = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    validation_df = pd.read_csv(VALIDATION_PATH, keep_default_na=False)
    x_train = train_df["review_text"]
    y_train = train_df["label"]
    x_validation = validation_df["review_text"]
    y_validation = validation_df["label"]

    # Vectorizer performs the same normalization for training and future inference.
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    fit_started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_started

    inference_started = time.perf_counter()
    predictions = pipeline.predict(x_validation)
    inference_seconds = time.perf_counter() - inference_started

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Training time: {fit_seconds:.2f} seconds")
    print(
        f"Validation inference time: {inference_seconds:.2f} seconds "
        f"({len(validation_df) / inference_seconds:.0f} reviews/second)"
    )
    print(f"Accuracy: {accuracy_score(y_validation, predictions):.4f}")
    print(
        f"Negative precision: "
        f"{precision_score(y_validation, predictions, pos_label=LABEL_NEGATIVE, zero_division=0):.4f}"
    )
    print(
        f"Negative recall: "
        f"{recall_score(y_validation, predictions, pos_label=LABEL_NEGATIVE, zero_division=0):.4f}"
    )
    print(
        f"Negative F1: "
        f"{f1_score(y_validation, predictions, pos_label=LABEL_NEGATIVE, zero_division=0):.4f}"
    )
    print("Confusion matrix (rows=actual, columns=predicted; labels 0=positive, 1=negative):")
    print(confusion_matrix(y_validation, predictions, labels=[0, 1]))
    print("Classification report:")
    print(
        classification_report(
            y_validation,
            predictions,
            labels=[0, 1],
            target_names=["positive", "negative"],
            zero_division=0,
            digits=4,
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved fitted pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
