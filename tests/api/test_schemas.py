"""Testes dos schemas Pydantic da API."""

from __future__ import annotations

from recsys.api.schemas import HealthResponse, RecommendationResponse


def test_recommendation_response_fields() -> None:
    r = RecommendationResponse(
        user_id=1, k=3, items=[10, 20, 30], model_version="2", source="embedding"
    )
    assert r.items == [10, 20, 30]
    assert r.source == "embedding"


def test_health_response_allows_null_version() -> None:
    h = HealthResponse(status="ok", model_loaded=False, model_version=None)
    assert h.model_loaded is False
    assert h.model_version is None
