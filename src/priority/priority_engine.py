"""Rule-based priority assignment for customer complaints."""

CRITICAL_KEYWORDS = {
    "fire",
    "smoke",
    "exploded",
    "electric shock",
    "injury",
    "injured",
    "dangerous",
    "unsafe",
    "overheated",
    "overheats",
}

HIGH_PRIORITY_KEYWORDS = {
    "broken",
    "defective",
    "stopped working",
    "not working",
    "damaged",
    "refund",
    "fraud",
    "scam",
    "failed",
    "not delivered",
}


def assign_priority(sentiment, rating=None, keywords=None):
    """Return Critical, High, Medium, or Low for a review."""

    if sentiment is None:
        raise ValueError("sentiment must not be None")

    sentiment = str(sentiment).strip().lower()

    if sentiment not in {"positive", "negative"}:
        raise ValueError("sentiment must be 'positive' or 'negative'")

    if rating is not None:
        if isinstance(rating, bool) or not isinstance(rating, int):
            raise ValueError("rating must be an integer from 1 to 5")
        if not 1 <= rating <= 5:
            raise ValueError("rating must be from 1 to 5")

    if keywords is None:
        keywords = []
    elif isinstance(keywords, str):
        keywords = [keywords]
    elif not isinstance(keywords, (list, tuple, set)):
        raise ValueError("keywords must be a string or a list of strings")

    normalized_keywords = {
        str(word).strip().lower()
        for word in keywords
        if word is not None
    }

    if sentiment == "positive":
        return "Low"

    if normalized_keywords & CRITICAL_KEYWORDS:
        return "Critical"

    if rating == 1 or normalized_keywords & HIGH_PRIORITY_KEYWORDS:
        return "High"

    return "Medium"
