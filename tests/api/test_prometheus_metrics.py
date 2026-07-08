"""Testes das métricas Prometheus."""

from __future__ import annotations

from prometheus_client import generate_latest

from recsys.api import prometheus_metrics as pm


def test_recommendations_counter_labels_and_render() -> None:
    pm.RECOMMENDATIONS.labels(source="fallback").inc()
    output = generate_latest().decode()
    assert "recsys_recommendations_total" in output
    assert 'source="fallback"' in output


def test_model_loaded_gauge_sets() -> None:
    pm.MODEL_LOADED.set(1)
    assert "recsys_model_loaded" in generate_latest().decode()
