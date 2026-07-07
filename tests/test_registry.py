"""Testes da regra de promoção Staging -> Production do Model Registry."""

from __future__ import annotations

from recsys.registry import should_promote


def test_promotes_when_candidate_beats_baseline() -> None:
    candidate = {"ndcg_at_k": 0.021, "map_at_k": 0.010}
    baseline = {"ndcg_at_k": 0.015, "map_at_k": 0.006}
    assert should_promote(candidate, baseline, "ndcg_at_k") is True


def test_rejects_when_candidate_worse() -> None:
    candidate = {"ndcg_at_k": 0.010}
    baseline = {"ndcg_at_k": 0.015}
    assert should_promote(candidate, baseline, "ndcg_at_k") is False


def test_rejects_when_metrics_are_equal() -> None:
    candidate = {"ndcg_at_k": 0.015}
    baseline = {"ndcg_at_k": 0.015}
    assert should_promote(candidate, baseline, "ndcg_at_k") is False


def test_promotes_when_no_incumbent_baseline_metric() -> None:
    # Sem baseline para comparar, o candidato assume Production.
    assert should_promote({"ndcg_at_k": 0.01}, {}, "ndcg_at_k") is True
