"""Simple features of a review text (the pull-request exercise of Session 2).

Each function takes one text and returns one number. The functions are used again
in Session 8, where seven simple features feed the first leaderboard model.

Work on a branch, implement one function with its tests, and open a pull request:

    git switch -c feature/n-exclamations
"""

from __future__ import annotations

import re

#: Names of the finished features. Add the name of your function here in your pull request.
FEATURE_NAMES = ["n_words"]

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
    """Number of exclamation marks in the text.

    TODO (exercise 6, pull request): implement with ``str.count``; raise ``TypeError``
    for non-text input like ``n_words`` does. Remove the skip marker of the tests.
    """
    raise NotImplementedError("exercise 6: n_exclamations")


def count_negations(text: str) -> int:
    """Number of negation words from ``NEGATIONS``, ignoring case.

    Hint: ``re.findall(r"[a-z']+", text.lower())`` splits into words and keeps "don't".
    Typographic apostrophes (’) should count as well: replace them with ' first.

    TODO (exercise 6, pull request).
    """
    _ = re  # remove this line when you use `re`
    raise NotImplementedError("exercise 6: count_negations")
