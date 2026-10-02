"""Evaluation metrics for retrieval and for answers with citations. All are pure functions.

Two kinds of relevance occur in the case study:
- document relevance: which retrieved decisions (BTI references) are relevant to a question (recall@k);
- heading relevance: does a retrieved decision carry the heading of the query decision (heading hit@k)?
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence


def _dedupe(ranked: Sequence[str]) -> list[str]:
    """Keep the first occurrence of each id (several chunks of one decision count once)."""
    return list(dict.fromkeys(ranked))


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the relevant ids that appear among the first k retrieved ids.

    >>> recall_at_k(["d3", "d1", "d7", "d4"], {"d1", "d4"}, k=3)
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

    >>> reciprocal_rank(["d3", "d1", "d7", "d4"], {"d1", "d4"})
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


def heading_hit_at_k(ranked_headings: Sequence[str], true_heading: str, k: int) -> float:
    """1.0 if the true heading is the heading of at least one of the first k retrieved decisions.

    >>> heading_hit_at_k(["3926", "6404", "6403"], "6403", k=2)
    0.0
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return float(true_heading in list(ranked_headings)[:k])


def evaluate_heading_retrieval(
    truth: dict[str, str], rankings: dict[str, Sequence[str]], k: int = 5
) -> dict[str, float]:
    """Mean heading hit@1, hit@k and MRR over labelled queries {query id: true heading}.

    `rankings` holds, per query, the headings of the retrieved decisions in rank order.
    """
    missing = set(truth) - set(rankings)
    if missing:
        raise KeyError(f"no ranking for {len(missing)} query(ies)")
    return {
        "hit@1": mean(heading_hit_at_k(rankings[q], h, 1) for q, h in truth.items()),
        f"hit@{k}": mean(heading_hit_at_k(rankings[q], h, k) for q, h in truth.items()),
        "mrr": mean(reciprocal_rank(list(rankings[q]), {h}) for q, h in truth.items()),
    }


# BTI references contain letters, digits, '/', '.', '-' and sometimes spaces (EEBTIEE BTI 030296-1/166/17)
CITATION = re.compile(r"\[([^\[\]\n]{2,60})\]")


def extract_citations(answer: str) -> list[str]:
    """Ids cited in square brackets, in order of first appearance: '... [DE-12345/21-1] [FR2019-0042]'."""
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
