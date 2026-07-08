"""Testes da API FastAPI (via TestClient com ModelService fake)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from recsys.api.main import _require_service, create_app


class _FakeService:
    model_version = "2"

    def recommend(self, user_id: int, k: int) -> tuple[list[int], str]:
        if user_id == 1:
            return [10, 20, 30][:k], "embedding"
        return [9, 8, 7][:k], "fallback"


def _client() -> TestClient:
    return TestClient(create_app(service=_FakeService()))


def test_health_ok() -> None:
    body = _client().get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "2"


def test_recommendations_known_user_uses_embedding() -> None:
    body = _client().get("/recommendations/1?k=2").json()
    assert body == {
        "user_id": 1,
        "k": 2,
        "items": [10, 20],
        "model_version": "2",
        "source": "embedding",
    }


def test_recommendations_cold_start_uses_fallback() -> None:
    body = _client().get("/recommendations/999").json()
    assert body["source"] == "fallback"
    assert body["items"] == [9, 8, 7]


def test_invalid_k_returns_422() -> None:
    assert _client().get("/recommendations/1?k=0").status_code == 422


def test_metrics_endpoint_prometheus_text() -> None:
    client = _client()
    client.get("/recommendations/999")  # gera contador de fallback
    text = client.get("/metrics").text
    assert "recsys_requests_total" in text
    assert "recsys_recommendations_total" in text


# ── M1: cardinality guard ──────────────────────────────────────────────────────


def test_unmatched_route_label_is_constant_not_raw_path() -> None:
    """404 em rota inexistente registra endpoint="unmatched", não o path raw."""
    client = _client()
    resp = client.get("/nao-existe")
    assert resp.status_code == 404
    metrics_text = client.get("/metrics").text
    assert 'endpoint="unmatched"' in metrics_text
    assert "/nao-existe" not in metrics_text


# ── M3: 503 quando serviço não carregado ──────────────────────────────────────


def test_require_service_returns_service_when_present() -> None:
    """_require_service retorna o serviço quando holder tem instância válida."""
    svc = _FakeService()
    result = _require_service({"service": svc})
    assert result is svc


def test_require_service_raises_503_when_none() -> None:
    """_require_service levanta HTTPException 503 quando service é None."""
    with pytest.raises(HTTPException) as exc_info:
        _require_service({"service": None})
    assert exc_info.value.status_code == 503
