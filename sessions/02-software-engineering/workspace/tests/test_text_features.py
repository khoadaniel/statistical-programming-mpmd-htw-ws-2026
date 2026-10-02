"""Tests of the text features. The skipped tests belong to the pull-request exercise."""

import pytest

from btitools.text_features import has_code, n_chars, n_digits, n_lines, n_words, upper_share


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Plush toy", 2), ("Plush  toy", 2), ("Plush\ntoy, 30 cm", 4), ("", 0)],
)
def test_n_words(text, expected):
    assert n_words(text) == expected


def test_n_chars_counts_everything():
    assert n_chars("Bär\n30 cm") == 9


@pytest.mark.parametrize(
    ("text", "expected"),
    [("one line", 1), ("first\nsecond", 2), ("first\r\n\r\nsecond\n", 2), ("", 0)],
)
def test_n_lines_ignores_empty_lines(text, expected):
    assert n_lines(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"), [("Classified under <CODE>.", 1), ("CODE", 0), ("", 0)]
)
def test_has_code(text, expected):
    assert has_code(text) == expected


@pytest.mark.parametrize("function", [n_chars, n_words, n_lines, has_code])
def test_features_reject_missing_text(function):
    with pytest.raises(TypeError):
        function(None)


@pytest.mark.skip(reason="exercise 6: delete this line in your pull request")
@pytest.mark.parametrize(
    ("text", "expected"),
    [("Höhe 30 cm", 2), ("Fläche 2 m²", 1), ("Plush toy", 0), ("", 0)],
)
def test_n_digits(text, expected):
    assert n_digits(text) == expected


@pytest.mark.skip(reason="exercise 6: delete this line in your pull request")
@pytest.mark.parametrize(
    ("text", "expected"),
    [("ABC", 1.0), ("abc", 0.0), ("Ab", 0.5), ("ÄÖ äö", 0.5), ("PVC 30 %", 1.0), ("123", 0.0)],
)
def test_upper_share(text, expected):
    assert upper_share(text) == pytest.approx(expected)
