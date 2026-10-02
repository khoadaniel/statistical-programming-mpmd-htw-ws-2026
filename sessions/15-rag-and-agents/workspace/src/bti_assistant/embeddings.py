"""Embedders turn a list of texts into a matrix of unit-length vectors (one row per text).

`HashingEmbedder` works offline and needs no model download. It captures shared words, not
meaning, so it is a stand-in for tests and quick experiments. `SentenceTransformerEmbedder`
uses a real multilingual embedding model (intfloat/multilingual-e5-small, 384 dimensions) and is the
course default. e5 models expect the prefix "query: " for search queries and "passage: " for stored
documents; `encode(texts, kind=...)` adds them.
"""

from __future__ import annotations

import re
import zlib
from typing import Literal, Protocol

import numpy as np

TOKEN = re.compile(r"\w+", re.UNICODE)
STOP_WORDS = frozenset(
    # a few very frequent function words of the main languages of the case study
    "a an and are as at be by for from in is it of on or the to with "
    "aus der die das und mit von für ein eine einem einer den des im "
    "de la le les et en un une du des pour avec "
    "een het van met voor".split()
)
Kind = Literal["query", "passage"]


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str], kind: Kind = "passage") -> np.ndarray: ...


def normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, so that the dot product equals the cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class HashingEmbedder:
    """Bag of words and word pairs without stop words, hashed into `dim` buckets (deterministic, offline)."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        return zlib.crc32(token.encode()) % self.dim

    def encode(self, texts: list[str], kind: Kind = "passage") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = [w for w in TOKEN.findall(text.lower()) if w not in STOP_WORDS]
            for token in words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]:
                out[i, self._bucket(token)] += 1.0
        return normalise(np.log1p(out))


class SentenceTransformerEmbedder:
    """A sentence-transformers model; downloads the weights on first use (about 470 MB for e5-small)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", use_prefixes: bool = True) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.use_prefixes = use_prefixes and "e5" in model_name

    def encode(self, texts: list[str], kind: Kind = "passage") -> np.ndarray:
        if self.use_prefixes:
            texts = [f"{kind}: {t}" for t in texts]
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True)
