import pytest

from src.nlp.keyword_extractor import extract_keywords


class FakeRow:
    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def tocsr(self):
        return self


class FakeVectorizer:
    def __init__(self, features, scores):
        self.features = features
        self.scores = scores

    def transform(self, texts):
        return FakeRow(list(range(len(self.features))), self.scores)

    def get_feature_names_out(self):
        return self.features


class FakePipeline:
    def __init__(self, features, scores):
        self.named_steps = {
            "tfidf": FakeVectorizer(features, scores),
        }


def test_extracts_phrases_preserves_negation_and_removes_redundant_words():
    pipeline = FakePipeline(
        ["not working", "poor battery", "working", "after two", "the"],
        [0.9, 0.8, 0.7, 0.6, 0.5],
    )

    keywords = extract_keywords("It is not working; poor battery.", pipeline=pipeline)

    assert keywords == ["not working", "poor battery"]


def test_empty_and_non_string_review_return_no_keywords():
    pipeline = FakePipeline(["battery"], [1.0])

    assert extract_keywords("  ", pipeline=pipeline) == []
    assert extract_keywords(None, pipeline=pipeline) == []
    assert extract_keywords("battery", pipeline=pipeline, top_k=0) == []


def test_negative_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k"):
        extract_keywords("battery", pipeline=FakePipeline(["battery"], [1.0]), top_k=-1)
