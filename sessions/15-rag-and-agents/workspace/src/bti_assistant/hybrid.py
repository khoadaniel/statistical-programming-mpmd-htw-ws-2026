"""Hybrid search: combine a keyword ranking and a vector ranking with reciprocal rank fusion."""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Merge rankings: every id gets the sum of 1 / (k + rank) over the rankings it appears in.

    k = 60 is the constant of Cormack, Clarke and Buettcher (2009); it damps the influence of the
    very first ranks. Ids are returned from best to worst; ties keep first-seen order.

    >>> reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]])
    ['a', 'c', 'b']
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])
