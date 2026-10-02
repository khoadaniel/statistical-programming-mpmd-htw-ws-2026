"""Tests of the title features. The skipped tests belong to the pull-request exercise."""

import pytest

from listingtools.text_features import (
    has_exclamation,
    mentions_size,
    n_chars,
    n_digits,
    n_words,
    upper_share,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Cosy flat", 2), ("Cosy  flat", 2), ("Cosy flat, 45 m²", 4), ("", 0)],
)
def test_n_words(text, expected):
    assert n_words(text) == expected


def test_n_chars_counts_everything():
    assert n_chars("Ruhig, 45 m²") == 12


@pytest.mark.parametrize(("text", "expected"), [("Wow!", 1), ("Quiet room", 0), ("", 0)])
def test_has_exclamation(text, expected):
    assert has_exclamation(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Loft 75 m² in Mitte", 1),
        ("75m2 flat", 1),
        ("Riesig (110qm) zentral", 1),
        ("130 sqm with a view", 1),
        ("2 rooms near the park", 0),
        ("Room 3, quiet", 0),
        ("", 0),
    ],
)
def test_mentions_size(text, expected):
    assert mentions_size(text) == expected


@pytest.mark.parametrize("function", [n_chars, n_words, has_exclamation, mentions_size])
def test_features_reject_missing_text(function):
    with pytest.raises(TypeError):
        function(None)


@pytest.mark.skip(reason="exercise 6: delete this line in your pull request")
@pytest.mark.parametrize(
    ("text", "expected"),
    [("Loft 110 qm", 3), ("Fläche 45 m²", 2), ("Cosy flat", 0), ("", 0)],
)
def test_n_digits(text, expected):
    assert n_digits(text) == expected


@pytest.mark.skip(reason="exercise 6: delete this line in your pull request")
@pytest.mark.parametrize(
    ("text", "expected"),
    [("ABC", 1.0), ("abc", 0.0), ("Ab", 0.5), ("ÄÖ äö", 0.5), ("WG 30 m²", 2 / 3), ("123", 0.0)],
)
def test_upper_share(text, expected):
    assert upper_share(text) == pytest.approx(expected)
