"""Simple features of a description of goods (the pull-request exercise of Session 2).

Reference solution, for self-checking only: compare with your own version after
you have tried the exercise. Copy it over src/btitools/text_features.py to check it.

Each function takes one text and returns one number. The functions are used again
in Session 8, where simple features feed the first leaderboard model (L1). They
describe the *form* of a description (how long, how many numbers, how many lines),
not its content; Session 8 shows how far that gets.

Work on a branch, implement one function with its tests, and open a pull request:

    git switch -c feature/n-digits
"""

from __future__ import annotations

#: Names of the finished features. Add the name of your function here in your pull request.
FEATURE_NAMES = [
    "n_chars",
    "n_words",
    "n_lines",
    "has_code",
    "n_digits",
    "upper_share",
]

#: Placeholder that replaces numbers repeating the decision's own tariff code.
CODE_PLACEHOLDER = "<CODE>"

#: The characters counted by ``n_digits``.
DIGITS = frozenset("0123456789")


def _check(text: object) -> str:
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    return text


def n_chars(text: str) -> int:
    """Number of characters, including spaces and line breaks.

    >>> n_chars("Plush toy")
    9
    """
    return len(_check(text))


def n_words(text: str) -> int:
    """Number of whitespace-separated words.

    >>> n_words("Plush  toy, 30 cm")
    4
    """
    return len(_check(text).split())


def n_lines(text: str) -> int:
    """Number of non-empty lines (descriptions often list properties line by line).

    >>> n_lines("Plush toy\n\nHeight: 30 cm")
    2
    """
    return sum(1 for line in _check(text).splitlines() if line.strip())


def has_code(text: str) -> int:
    """1 if the description quotes its own tariff code (masked as ``<CODE>``), else 0.

    >>> has_code("Classified under <CODE> as a toy")
    1
    """
    return int(CODE_PLACEHOLDER in _check(text))


def n_digits(text: str) -> int:
    """Number of digit characters (0-9) in the text.

    ``str.isdigit`` would also count superscripts such as the ² in "m²".
    """
    return sum(char in DIGITS for char in _check(text))


def upper_share(text: str) -> float:
    """Share of upper-case letters among all letters; 0.0 if the text has no letters."""
    letters = [char for char in _check(text) if char.isalpha()]
    if not letters:
        return 0.0
    return sum(char.isupper() for char in letters) / len(letters)
