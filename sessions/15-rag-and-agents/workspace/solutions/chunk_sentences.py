"""Exercise 2: sentence-based chunking (replace the stub in src/review_assistant/chunking.py)."""

import re

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_sentences(text: str, max_words: int = 120) -> list[str]:
    sentences = [s for s in SENTENCE_END.split(text.strip()) if s]
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        n_current = sum(len(s.split()) for s in current)
        if current and n_current + len(sentence.split()) > max_words:
            chunks.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks
