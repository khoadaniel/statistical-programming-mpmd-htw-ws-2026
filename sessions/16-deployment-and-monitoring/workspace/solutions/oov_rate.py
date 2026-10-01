"""Exercise 4: out-of-vocabulary rate (replace the stub in src/sentiment_service/drift.py)."""

import re
from collections.abc import Sequence

TOKEN = re.compile(r"(?u)\b\w\w+\b")  # the default token pattern of TfidfVectorizer


def oov_rate(texts: Sequence[str], vocabulary: set[str]) -> float:
    words = [w for t in texts for w in TOKEN.findall(t.lower())]
    if not words:
        return 0.0
    return sum(w not in vocabulary for w in words) / len(words)
