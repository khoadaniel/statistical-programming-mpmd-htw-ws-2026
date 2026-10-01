"""Evaluation metrics for retrieval and for answers with citations. All are pure functions."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence


def _dedupe(ranked: Sequence[str]) -> list[str]:
    """Keep the first occurrence of each id (several chunks of one review count once)."""
    return list(dict.fromkeys(ranked))


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the relevant ids that appear among the first k retrieved ids.

    >>> recall_at_k(["r3", "r1", "r7", "r4"], {"r1", "r4"}, k=3)
    0.5
    """
    relevant = set(relevant)
    if not relevant:
        raise ValueError("a test question needs at least one relevant id")
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(_dedupe(ranked)[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the first k retrieved ids that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(_dedupe(ranked)[:k]) & set(relevant)) / k


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if at least one relevant id is among the first k, else 0.0."""
    return float(bool(set(_dedupe(ranked)[:k]) & set(relevant)))


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant id; 0.0 if none was retrieved.

    >>> reciprocal_rank(["r3", "r1", "r7", "r4"], {"r1", "r4"})
    0.5
    """
    relevant = set(relevant)
    for rank, doc_id in enumerate(_dedupe(ranked), start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def evaluate_retrieval(
    test_set: dict[str, set[str]], rankings: dict[str, Sequence[str]], k: int = 5
) -> dict[str, float]:
    """Mean recall@k, hit@k and MRR over a labelled test set {question: relevant ids}."""
    missing = set(test_set) - set(rankings)
    if missing:
        raise KeyError(f"no ranking for {len(missing)} question(s)")
    return {
        f"recall@{k}": mean(recall_at_k(rankings[q], rel, k) for q, rel in test_set.items()),
        f"hit@{k}": mean(hit_at_k(rankings[q], rel, k) for q, rel in test_set.items()),
        "mrr": mean(reciprocal_rank(rankings[q], rel) for q, rel in test_set.items()),
    }


CITATION = re.compile(r"\[([A-Za-z0-9_#-]+)\]")


def extract_citations(answer: str) -> list[str]:
    """Ids cited in square brackets, in order of first appearance: '... [r001] [r002]'."""
    return list(dict.fromkeys(CITATION.findall(answer)))


def citation_precision(cited: Iterable[str], retrieved: Iterable[str]) -> float:
    """Share of cited ids that were actually shown to the model. 1.0 means no invented sources.

    Returns 0.0 for an answer without citations: it cannot be checked.
    """
    cited = set(cited)
    if not cited:
        return 0.0
    return len(cited & set(retrieved)) / len(cited)


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Agreement of two raters beyond chance, for labels such as grounded 1/0."""
    if len(a) != len(b) or not a:
        raise ValueError("ratings must be non-empty and of equal length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    labels = set(a) | set(b)
    expected = sum((list(a).count(c) / n) * (list(b).count(c) / n) for c in labels)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
