"""Explainable TF-IDF keyword extraction for an individual review."""
from pathlib import Path
import sys

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Support imports when called from the repository root or another working dir.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.predict import load_model  # noqa: E402
from src.ml.preprocess import normalize_text  # noqa: E402

# Keep negation words so phrases such as "not working" remain extractable.
_FILTERED_WORDS = ENGLISH_STOP_WORDS.difference({"no", "nor", "not"})


def extract_keywords(review_text: str, pipeline=None, top_k: int = 5) -> list[str]:
    """Return up to `top_k` informative unigrams/bigrams from one review.

    Scores come from the fitted TF-IDF vectorizer. They rank terms within this
    review; they do not determine whether a phrase is truly a complaint.
    """
    if top_k < 0:
        raise ValueError("top_k must be zero or greater")
    if top_k == 0 or not isinstance(review_text, str):
        return []

    cleaned_text = normalize_text(review_text)
    if not cleaned_text:
        return []

    fitted_pipeline = pipeline if pipeline is not None else load_model()
    vectorizer = fitted_pipeline.named_steps["tfidf"]
    row = vectorizer.transform([cleaned_text]).tocsr()
    feature_names = vectorizer.get_feature_names_out()

    candidates = []
    for index, score in zip(row.indices, row.data):
        phrase = str(feature_names[index])
        words = phrase.split()
        # Suppress function words while preserving no/nor/not in phrases.
        if len(words) == 1 and words[0] in _FILTERED_WORDS:
            continue
        if len(words) > 1 and any(word in _FILTERED_WORDS for word in words):
            continue
        candidates.append((phrase, float(score)))

    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [phrase for phrase, _ in candidates[:top_k]]
