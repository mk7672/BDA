"""Compare negative-class decision thresholds on validation data only."""
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Support both `python src/ml/evaluate_thresholds.py` and module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.preprocess import LABEL_NEGATIVE  # noqa: E402

VALIDATION_PATH = Path("data/processed/validation.csv")
MODEL_PATH = Path("models/sentiment_pipeline.joblib")
THRESHOLDS = [round(value, 2) for value in (0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80)]


def main() -> None:
    validation_df = pd.read_csv(VALIDATION_PATH, keep_default_na=False)
    x_validation = validation_df["review_text"]
    y_validation = validation_df["label"].astype(int)

    pipeline = joblib.load(MODEL_PATH)
    class_index = list(pipeline.classes_).index(LABEL_NEGATIVE)
    negative_scores = pipeline.predict_proba(x_validation)[:, class_index]

    print("Validation-only threshold comparison")
    print("threshold  neg_precision  neg_recall  neg_f1  false_alarms  missed_negatives")
    for threshold in THRESHOLDS:
        predictions = (negative_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_validation, predictions, labels=[0, 1]).ravel()
        precision = precision_score(
            y_validation, predictions, pos_label=LABEL_NEGATIVE, zero_division=0
        )
        recall = recall_score(y_validation, predictions, pos_label=LABEL_NEGATIVE, zero_division=0)
        f1 = f1_score(y_validation, predictions, pos_label=LABEL_NEGATIVE, zero_division=0)
        print(
            f"{threshold:9.2f}  {precision:13.4f}  {recall:10.4f}  {f1:6.4f}"
            f"  {fp:12d}  {fn:16d}"
        )


if __name__ == "__main__":
    main()
