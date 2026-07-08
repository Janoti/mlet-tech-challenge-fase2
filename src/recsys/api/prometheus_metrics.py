"""Métricas Prometheus da API de recomendação (prefixo recsys_)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

REQUESTS = Counter("recsys_requests_total", "Total de requests HTTP.", ["endpoint", "status"])
REQUEST_DURATION = Histogram(
    "recsys_request_duration_seconds", "Latência das requests HTTP.", ["endpoint"]
)
RECOMMENDATIONS = Counter(
    "recsys_recommendations_total", "Recomendações servidas por origem.", ["source"]
)
MODEL_LOADED = Gauge("recsys_model_loaded", "1 se o modelo está carregado, senão 0.")
MODEL_INFO = Info("recsys_model", "Metadados do modelo servido.")
