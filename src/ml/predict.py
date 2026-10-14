"""Reusable review prediction interface for the fitted sentiment pipeline."""
from pathlib import Path
import sys
from typing import Any

import joblib

# Support imports and direct execution from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.preprocess import LABEL_NEGATIVE, SENTIMENT_NAMES, normalize_text  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
DEFAULT_NEGATIVE_THRESHOLD = 0.45
_MODEL_CACHE: dict[str, Any] = {}


def load_model(model_path: str | Path = MODEL_PATH):
    """Load a fitted pipeline once and reuse it for subsequent predictions."""
    resolved_path = str(Path(model_path).resolve())
    if resolved_path not in _MODEL_CACHE:
        _MODEL_CACHE[resolved_path] = joblib.load(resolved_path)
    return _MODEL_CACHE[resolved_path]


def predict_review(
    review_text: str,
    pipeline=None,
    threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
) -> dict[str, Any]:
    """Predict one review and return evidence fields for downstream modules.

    `confidence` is the model's probability for the predicted class; it is not
    guaranteed to be a calibrated real-world probability. `negative_score`
    always reports the model's score for the negative class.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    if not isinstance(review_text, str):
        return {
            "sentiment": None,
            "confidence": None,
            "negative_score": None,
            "threshold": threshold,
            "keywords": [],
            "error": {"code": "invalid_review_text", "message": "Review text must be a string."},
            "keyword_error": None,
        }

    cleaned_text = normalize_text(review_text)
    if not cleaned_text:
        return {
            "sentiment": None,
            "confidence": None,
            "negative_score": None,
            "threshold": threshold,
            "keywords": [],
            "error": {"code": "empty_review_text", "message": "Review text is empty after normalization."},
            "keyword_error": None,
        }

    try:
        fitted_pipeline = pipeline if pipeline is not None else load_model()
        probabilities = fitted_pipeline.predict_proba([cleaned_text])[0]
        class_probabilities = dict(zip(fitted_pipeline.classes_, probabilities))
        negative_score = float(class_probabilities[LABEL_NEGATIVE])
        predicted_label = (
            LABEL_NEGATIVE if negative_score >= threshold else 1 - LABEL_NEGATIVE
        )
        confidence = (
            negative_score if predicted_label == LABEL_NEGATIVE else 1.0 - negative_score
        )
        keywords = []
        keyword_error = None
        if predicted_label == LABEL_NEGATIVE:
            try:
                # Lazy import avoids a module cycle; keyword extraction remains
                # independently callable and cannot cancel a sentiment result.
                from src.nlp.keyword_extractor import extract_keywords

                keywords = extract_keywords(cleaned_text, pipeline=fitted_pipeline)
            except Exception as exc:
                keyword_error = {
                    "code": "keyword_extraction_failed",
                    "message": str(exc),
                }
        return {
            "sentiment": SENTIMENT_NAMES[int(predicted_label)],
            "confidence": confidence,
            "negative_score": negative_score,
            "threshold": threshold,
            "keywords": keywords,
            "error": None,
            "keyword_error": keyword_error,
        }
    except Exception as exc:
        return {
            "sentiment": None,
            "confidence": None,
            "negative_score": None,
            "threshold": threshold,
            "keywords": [],
            "error": {"code": "prediction_failed", "message": str(exc)},
            "keyword_error": None,
        }
