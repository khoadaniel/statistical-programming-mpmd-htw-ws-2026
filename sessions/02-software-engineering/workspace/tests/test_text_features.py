"""Tests of the text features. The skipped tests belong to the pull-request exercise."""

import pytest

from reviewtools.text_features import count_negations, n_exclamations, n_words


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Great product", 2), ("Great  product", 2), ("Works\nas described", 3), ("", 0)],
)
def test_n_words(text, expected):
    assert n_words(text) == expected


def test_n_words_rejects_missing_text():
    with pytest.raises(TypeError):
        n_words(None)


@pytest.mark.skip(reason="exercise 6: delete this line in your pull request")
@pytest.mark.parametrize(("text", "expected"), [("Great!", 1), ("No!!!", 3), ("ok", 0), ("", 0)])
def test_n_exclamations(text, expected):
    assert n_exclamations(text) == expected


@pytest.mark.skip(reason="exercise 6: delete this line in your pull request")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("not good", 1),
        ("Don't buy, never again", 2),
        ("It doesn’t work", 1),  # typographic apostrophe
        ("Nothing wrong with it", 1),
        ("Excellent", 0),
        ("", 0),
    ],
)
def test_count_negations(text, expected):
    assert count_negations(text) == expected
