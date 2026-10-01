import math

import pytest

from review_assistant.hybrid import reciprocal_rank_fusion
from review_assistant.metrics import (
    citation_precision,
    cohen_kappa,
    evaluate_retrieval,
    extract_citations,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED, RELEVANT = ["r3", "r1", "r7", "r4"], {"r1", "r4"}


def test_recall_at_k_worked_example():
    assert recall_at_k(RANKED, RELEVANT, k=1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, k=3) == 0.5
    assert recall_at_k(RANKED, RELEVANT, k=4) == 1.0


def test_recall_counts_a_document_once_even_with_several_chunks():
    assert recall_at_k(["r1", "r1", "r1", "r4"], RELEVANT, k=2) == 1.0


def test_precision_hit_and_reciprocal_rank():
    assert precision_at_k(RANKED, RELEVANT, k=2) == 0.5
    assert hit_at_k(RANKED, RELEVANT, k=1) == 0.0 and hit_at_k(RANKED, RELEVANT, k=2) == 1.0
    assert reciprocal_rank(RANKED, RELEVANT) == 0.5
    assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), k=3)
    with pytest.raises(ValueError):
        recall_at_k(RANKED, RELEVANT, k=0)


def test_evaluate_retrieval_averages_over_questions():
    test_set = {"q1": {"a"}, "q2": {"b", "c"}}
    rankings = {"q1": ["a", "x"], "q2": ["x", "b"]}
    scores = evaluate_retrieval(test_set, rankings, k=2)
    assert scores == {"recall@2": 0.75, "hit@2": 1.0, "mrr": 0.75}
    with pytest.raises(KeyError):
        evaluate_retrieval(test_set, {"q1": ["a"]}, k=2)


def test_citations_are_extracted_and_checked():
    answer = "One pad failed after a month [r6], another after a year [r9] [r6]."
    assert extract_citations(answer) == ["r6", "r9"]
    assert citation_precision(["r6", "r9"], retrieved=["r6", "r7"]) == 0.5  # r9 was never shown
    assert citation_precision([], retrieved=["r6"]) == 0.0


def test_cohen_kappa():
    human = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    judge = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    assert math.isclose(cohen_kappa(human, judge), 0.4286, abs_tol=1e-3)
    assert cohen_kappa([1, 0, 1], [1, 0, 1]) == 1.0


def test_reciprocal_rank_fusion():
    assert reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]]) == ["a", "c", "b"]
    assert reciprocal_rank_fusion([]) == []
