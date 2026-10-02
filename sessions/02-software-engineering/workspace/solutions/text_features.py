"""Simple features of a listing title (the pull-request exercise of Session 2).

Reference solution, for self-checking only: compare with your own version after
you have tried the exercise. Copy it over src/listingtools/text_features.py to check it.

Each function takes one title and returns one number. The features describe the *form* of a
title (how long, how loud, whether it states the size of the flat), not the flat itself.
Hosts of large flats often put the size into the title ("Loft 110 qm, Mitte"), so such
features can carry information; whether they improve a price model is a question for the
feature-engineering session (Session 9).

Work on a branch, implement one function with its tests, and open a pull request:

    git switch -c feature/n-digits
"""

from __future__ import annotations

import re

#: Names of the finished features. Add the name of your function here in your pull request.
FEATURE_NAMES = [
    "n_chars",
    "n_words",
    "has_exclamation",
    "mentions_size",
    "n_digits",
    "upper_share",
]

#: The characters counted by ``n_digits``.
DIGITS = frozenset("0123456789")

#: A number followed by a unit of floor area: 75 m², 75m2, 60 qm, 130 sqm, 40 sq m.
SIZE = re.compile(r"\b\d+\s?(?:m²|m2|qm|sqm|sq\.?\s?m|square met(?:er|re)s?)(?![a-z0-9])", re.I)


def _check(text: object) -> str:
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    return text


def n_chars(text: str) -> int:
    """Number of characters, including spaces.

    >>> n_chars("Cosy flat")
    9
    """
    return len(_check(text))


def n_words(text: str) -> int:
    """Number of whitespace-separated words.

    >>> n_words("Cosy  flat, 45 m²")
    4
    """
    return len(_check(text).split())


def has_exclamation(text: str) -> int:
    """1 if the title contains an exclamation mark, else 0.

    >>> has_exclamation("Best view in town!")
    1
    """
    return int("!" in _check(text))


def mentions_size(text: str) -> int:
    """1 if the title states a floor area such as "75 m²", "60qm" or "130 sqm", else 0.

    >>> mentions_size("Riesig (110qm) zentral")
    1
    """
    return int(SIZE.search(_check(text)) is not None)


def n_digits(text: str) -> int:
    """Number of digit characters (0-9) in the title.

    ``str.isdigit`` would also count superscripts such as the ² in "m²".
    """
    return sum(char in DIGITS for char in _check(text))


def upper_share(text: str) -> float:
    """Share of upper-case letters among all letters; 0.0 if the title has no letters."""
    letters = [char for char in _check(text) if char.isalpha()]
    if not letters:
        return 0.0
    return sum(char.isupper() for char in letters) / len(letters)
