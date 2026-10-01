"""Simple features of a review text (the pull-request exercise of Session 2).

Reference solution, for self-checking only: compare with your own version after
you have tried the exercise. Copy it over src/reviewtools/text_features.py to check it.

Each function takes one text and returns one number. The functions are used again
in Session 8, where seven simple features feed the first leaderboard model.

Work on a branch, implement one function with its tests, and open a pull request:

    git switch -c feature/n-exclamations
"""

from __future__ import annotations

import re

#: Names of the finished features. Add the name of your function here in your pull request.
FEATURE_NAMES = ["n_words", "n_exclamations", "count_negations"]

#: Negation words counted by ``count_negations`` (lower case).
NEGATIONS = frozenset(
    {"not", "no", "never", "nothing", "don't", "didn't", "doesn't", "isn't", "wasn't", "can't"}
)


def n_words(text: str) -> int:
    """Number of whitespace-separated words.

    >>> n_words("Works  as described")
    3
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    return len(text.split())


def n_exclamations(text: str) -> int:
    """Number of exclamation marks in the text."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    return text.count("!")


def count_negations(text: str) -> int:
    """Number of negation words from ``NEGATIONS``, ignoring case.

    Typographic apostrophes (’) are replaced by ' so that "doesn’t" counts.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    words = re.findall(r"[a-z']+", text.lower().replace("\u2019", "'"))
    return sum(word in NEGATIONS for word in words)
