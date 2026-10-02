"""Exercise 4: out-of-vocabulary rate (replace the stub in src/tariff_service/drift.py)."""

from collections.abc import Sequence


def oov_rate(texts: Sequence[str], vocabulary: set[str]) -> float:
    import re

    token = re.compile(r"(?u)\b\w\w+\b")  # the default token pattern of TfidfVectorizer
    words = [w for t in texts for w in token.findall(t.lower())]
    if not words:
        return 0.0
    return sum(w not in vocabulary for w in words) / len(words)
