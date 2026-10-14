import pytest

from src.ml.predict import DEFAULT_NEGATIVE_THRESHOLD, predict_review


class FakeRow:
    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def tocsr(self):
        return self


class FakeVectorizer:
    def transform(self, texts):
        return FakeRow([0, 1, 2, 3], [0.8, 0.7, 0.2, 0.1])

    def get_feature_names_out(self):
        return ["stopped working", "battery", "stopped", "working"]


class FakePipeline:
    classes_ = [0, 1]
    named_steps = {"tfidf": FakeVectorizer()}

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, texts):
        return [self.probabilities]


def test_negative_prediction_includes_phrase_keywords():
    result = predict_review(
        "The battery stopped working.", pipeline=FakePipeline([0.1, 0.9])
    )

    assert result["sentiment"] == "negative"
    assert result["confidence"] == pytest.approx(0.9)
    assert result["negative_score"] == pytest.approx(0.9)
    assert result["threshold"] == DEFAULT_NEGATIVE_THRESHOLD
    assert result["keywords"] == ["stopped working", "battery"]
    assert result["error"] is None
    assert result["keyword_error"] is None


def test_positive_prediction_returns_empty_complaint_keywords():
    result = predict_review(
        "This is excellent.", pipeline=FakePipeline([0.95, 0.05])
    )

    assert result["sentiment"] == "positive"
    assert result["confidence"] == pytest.approx(0.95)
    assert result["negative_score"] == pytest.approx(0.05)
    assert result["keywords"] == []
    assert result["error"] is None


@pytest.mark.parametrize(
    "review_text, expected_code",
    [(None, "invalid_review_text"), ("  \n ", "empty_review_text")],
)
def test_invalid_or_empty_review_is_reported(review_text, expected_code):
    result = predict_review(review_text, pipeline=FakePipeline([0.5, 0.5]))

    assert result["sentiment"] is None
    assert result["keywords"] == []
    assert result["error"]["code"] == expected_code


def test_threshold_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="threshold"):
        predict_review("Review text", pipeline=FakePipeline([0.5, 0.5]), threshold=1.1)
