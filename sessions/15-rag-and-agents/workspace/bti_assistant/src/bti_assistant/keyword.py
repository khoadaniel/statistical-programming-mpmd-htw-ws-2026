"""BM25 keyword search (Robertson and Zaragoza 2009), in plain Python."""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    """Score documents by the query words they contain, weighted by rarity (idf) and length."""

    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.ids, self.k1, self.b = ids, k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> list[float]:
        words = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for w in words:
                tf = doc.get(w, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    s += self.idf[w] * tf * (self.k1 + 1) / norm
            out.append(s)
        return out

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        ranked = sorted(zip(self.ids, self.scores(query), strict=True), key=lambda x: -x[1])
        return [(i, s) for i, s in ranked[:k] if s > 0]
