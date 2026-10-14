"""Text normalization and label preparation for the review sentiment module.

Keep these functions at module top level with stable names: the saved model
pipeline will reference normalize_text by import path.
"""
import html
import re

# Negative is encoded as 1 so scikit-learn's default positive class (and the default
# precision/recall/F1) describe complaint detection.
LABEL_NEGATIVE = 1
LABEL_POSITIVE = 0
SENTIMENT_NAMES = {LABEL_NEGATIVE: "negative", LABEL_POSITIVE: "positive"}

NEGATIVE_MAX_RATING = 2  # ratings <= 2 are negative
POSITIVE_MIN_RATING = 4  # ratings >= 4 are positive; ratings in between are excluded

_MEDIA_MARKER = re.compile(r"\[\[[A-Za-z_]+:[^\]]*\]\]")  # e.g. [[VIDEOID:abc123]]
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")  # <br />, <b>, </i> ... but not "a < b"
_WHITESPACE = re.compile(r"\s+")
_CANT = re.compile(r"\bcan't\b", re.IGNORECASE)
_CANNOT = re.compile(r"\bcannot\b", re.IGNORECASE)
_WONT = re.compile(r"\bwon't\b", re.IGNORECASE)
_NT = re.compile(r"(\w)n't\b", re.IGNORECASE)


def normalize_text(text) -> str:
    """Clean one review. Returns "" for None, NaN or any non-string input.

    Does not lowercase (the TF-IDF vectorizer does that) and does not remove
    stop words or punctuation, so negation such as "not working" is preserved.
    """
    if not isinstance(text, str):
        return ""
    text = _MEDIA_MARKER.sub(" ", text)
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = _CANT.sub("can not", text)
    text = _CANNOT.sub("can not", text)
    text = _WONT.sub("will not", text)
    text = _NT.sub(r"\1 not", text)
    return _WHITESPACE.sub(" ", text).strip()


def rating_to_label(rating):
    """Map a 1-5 star rating to LABEL_NEGATIVE / LABEL_POSITIVE, or None to exclude.

    Neutral (3-star), out-of-range and invalid ratings return None. Nothing is
    silently relabeled.
    """
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return None
    if value != value or not 1 <= value <= 5:  # NaN or out of range
        return None
    if value <= NEGATIVE_MAX_RATING:
        return LABEL_NEGATIVE
    if value >= POSITIVE_MIN_RATING:
        return LABEL_POSITIVE
    return None
