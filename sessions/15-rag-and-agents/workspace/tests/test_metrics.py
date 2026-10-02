import math

import pytest

from bti_assistant.hybrid import reciprocal_rank_fusion
from bti_assistant.metrics import (
    citation_precision,
    cohen_kappa,
    evaluate_heading_retrieval,
    evaluate_retrieval,
    extract_citations,
    heading_hit_at_k,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED, RELEVANT = ["d3", "d1", "d7", "d4"], {"d1", "d4"}


def test_recall_at_k_worked_example():
    assert recall_at_k(RANKED, RELEVANT, k=1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, k=3) == 0.5
    assert recall_at_k(RANKED, RELEVANT, k=4) == 1.0


def test_recall_counts_a_document_once_even_with_several_chunks():
    assert recall_at_k(["d1", "d1", "d1", "d4"], RELEVANT, k=2) == 1.0


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
    with pytest.raises(ValueError):
        heading_hit_at_k(["6403"], "6403", k=0)


def test_evaluate_retrieval_averages_over_questions():
    test_set = {"q1": {"a"}, "q2": {"b", "c"}}
    rankings = {"q1": ["a", "x"], "q2": ["x", "b"]}
    scores = evaluate_retrieval(test_set, rankings, k=2)
    assert scores == {"recall@2": 0.75, "hit@2": 1.0, "mrr": 0.75}
    with pytest.raises(KeyError):
        evaluate_retrieval(test_set, {"q1": ["a"]}, k=2)


def test_heading_hit_at_k_worked_example():
    # headings of the retrieved decisions, in rank order; the same heading may occur several times
    ranked = ["3926", "6404", "6403", "6403"]
    assert heading_hit_at_k(ranked, "6403", k=2) == 0.0
    assert heading_hit_at_k(ranked, "6403", k=3) == 1.0
    truth = {"q1": "6403", "q2": "9503"}
    scores = evaluate_heading_retrieval(truth, {"q1": ranked, "q2": ["9503", "9503"]}, k=3)
    assert scores == {"hit@1": 0.5, "hit@3": 1.0, "mrr": (1 / 3 + 1) / 2}


def test_citations_are_extracted_and_checked():
    answer = "Similar boots were classified in 6403 [DE-001/21], see also [EEBTIEE BTI 030296-1/166/17] [DE-001/21]."
    assert extract_citations(answer) == ["DE-001/21", "EEBTIEE BTI 030296-1/166/17"]
    assert citation_precision(["DE-001/21", "XX-9"], retrieved=["DE-001/21", "FR-2022-01"]) == 0.5
    assert citation_precision([], retrieved=["DE-001/21"]) == 0.0


def test_cohen_kappa():
    human = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    judge = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    assert math.isclose(cohen_kappa(human, judge), 0.4286, abs_tol=1e-3)
    assert cohen_kappa([1, 0, 1], [1, 0, 1]) == 1.0


def test_reciprocal_rank_fusion():
    assert reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]]) == ["a", "c", "b"]
    assert reciprocal_rank_fusion([]) == []
