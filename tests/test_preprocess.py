import pytest

from src.ml.preprocess import (
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    SENTIMENT_NAMES,
    normalize_text,
    rating_to_label,
)


def test_missing_and_non_string_input_become_empty_string():
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
    assert normalize_text(123) == ""


def test_empty_and_whitespace_only():
    assert normalize_text("") == ""
    assert normalize_text("   \n\t ") == ""


def test_html_and_entities_are_cleaned():
    raw = "(Bad) It had to be cut.<br />(GOOD)One caught on fire &#34;quickly&#34;"
    assert normalize_text(raw) == '(Bad) It had to be cut. (GOOD)One caught on fire "quickly"'


def test_media_markers_are_removed():
    raw = "[[VIDEOID:2bbb8a46d9]] This is a really good pitcher"
    assert normalize_text(raw) == "This is a really good pitcher"


def test_negation_is_made_explicit_and_preserved():
    assert normalize_text("The product isn't good") == "The product is not good"
    assert normalize_text("It DOESN'T work") == "It DOES not work"
    assert normalize_text("I can't fit it") == "I can not fit it"
    assert normalize_text("I cannot fit it") == "I can not fit it"
    assert normalize_text("It won't turn on") == "It will not turn on"
    assert normalize_text("Didn\u2019t fit") == "Did not fit"
    assert "not" in normalize_text("not working").split()


def test_normalization_is_idempotent():
    for raw in ["Didn't fit<br />at all", "Works great", "  spaced   out  "]:
        once = normalize_text(raw)
        assert normalize_text(once) == once


@pytest.mark.parametrize(
    "rating, expected",
    [
        (1, LABEL_NEGATIVE),
        (2, LABEL_NEGATIVE),
        (3, None),
        (4, LABEL_POSITIVE),
        (5, LABEL_POSITIVE),
        (2.5, None),
        (0, None),
        (6, None),
        (None, None),
        (float("nan"), None),
        ("5", LABEL_POSITIVE),
        ("abc", None),
    ],
)
def test_rating_to_label(rating, expected):
    assert rating_to_label(rating) == expected


def test_label_encoding_and_names():
    assert LABEL_NEGATIVE == 1
    assert SENTIMENT_NAMES == {1: "negative", 0: "positive"}
