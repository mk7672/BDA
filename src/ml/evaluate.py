"""Evaluate the selected fitted pipeline once on the held-out test split."""
from pathlib import Path
import sys
import time

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Support both `python src/ml/evaluate.py` and module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.predict import DEFAULT_NEGATIVE_THRESHOLD, MODEL_PATH  # noqa: E402
from src.ml.preprocess import LABEL_NEGATIVE  # noqa: E402

TEST_PATH = Path("data/processed/test.csv")


def main() -> None:
    test_df = pd.read_csv(TEST_PATH, keep_default_na=False)
    x_test = test_df["review_text"]
    y_test = test_df["label"].astype(int)
    pipeline = joblib.load(MODEL_PATH)

    inference_started = time.perf_counter()
    negative_scores = pipeline.predict_proba(x_test)[:, list(pipeline.classes_).index(LABEL_NEGATIVE)]
    predictions = (negative_scores >= DEFAULT_NEGATIVE_THRESHOLD).astype(int)
    inference_seconds = time.perf_counter() - inference_started

    print("Final held-out test evaluation; no tuning is performed here.")
    print(f"Threshold: {DEFAULT_NEGATIVE_THRESHOLD:.2f}")
    print(f"Test rows: {len(test_df)}")
    print(f"Inference time: {inference_seconds:.2f} seconds")
    print(f"Inference throughput: {len(test_df) / inference_seconds:.0f} reviews/second")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Negative precision: "
        f"{precision_score(y_test, predictions, pos_label=LABEL_NEGATIVE, zero_division=0):.4f}"
    )
    print(
        "Negative recall: "
        f"{recall_score(y_test, predictions, pos_label=LABEL_NEGATIVE, zero_division=0):.4f}"
    )
    print(
        "Negative F1: "
        f"{f1_score(y_test, predictions, pos_label=LABEL_NEGATIVE, zero_division=0):.4f}"
    )
    print("Confusion matrix (rows=actual, columns=predicted; labels 0=positive, 1=negative):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["positive", "negative"],
            zero_division=0,
            digits=4,
        )
    )


if __name__ == "__main__":
    main()
