import pytest

from recsys.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

REC = [1, 2, 3, 4]
REL = {2, 4}


def test_precision_at_k():
    assert precision_at_k(REC, REL, 4) == pytest.approx(0.5)


def test_recall_at_k():
    assert recall_at_k(REC, REL, 4) == pytest.approx(1.0)


def test_ndcg_at_k():
    assert ndcg_at_k(REC, REL, 4) == pytest.approx(0.6509, abs=1e-3)


def test_average_precision_at_k():
    assert average_precision_at_k(REC, REL, 4) == pytest.approx(0.5)


def test_empty_relevant_is_zero():
    assert recall_at_k(REC, set(), 4) == 0.0
    assert ndcg_at_k(REC, set(), 4) == 0.0
    assert average_precision_at_k(REC, set(), 4) == 0.0
