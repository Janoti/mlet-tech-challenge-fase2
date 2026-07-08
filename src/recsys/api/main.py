"""API FastAPI de recomendação: serving + Swagger + health + métricas."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import RequestResponseEndpoint

from recsys.api.model_service import ModelService, load_model_service
from recsys.api.prometheus_metrics import (
    MODEL_INFO,
    MODEL_LOADED,
    RECOMMENDATIONS,
    REQUEST_DURATION,
    REQUESTS,
)
from recsys.api.schemas import HealthResponse, RecommendationResponse


def create_app(service: ModelService | None = None) -> FastAPI:
    """Cria o app FastAPI. ``service`` injetável facilita testes (fake).

    Args:
        service: Instância de ModelService a usar. Se None, carrega do MLflow
            no evento de startup (somente em produção).

    Returns:
        Instância do FastAPI configurada com endpoints e middleware.
    """
    holder: dict[str, ModelService | None] = {"service": service}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
        if holder["service"] is None:
            holder["service"] = load_model_service()
        MODEL_LOADED.set(1)
        MODEL_INFO.info({"version": holder["service"].model_version})  # type: ignore[union-attr]
        yield

    app = FastAPI(
        title="Recsys API",
        version="1.0.0",
        description="Recomendações de e-commerce (embedding + fallback de popularidade).",
        docs_url="/docs",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _record_metrics(request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        REQUEST_DURATION.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(endpoint=endpoint, status=response.status_code).inc()
        return response

    @app.get("/health", response_model=HealthResponse, tags=["Monitoramento"])
    def health() -> HealthResponse:
        """Verifica se a API e o modelo estão operacionais."""
        svc = holder["service"]
        return HealthResponse(
            status="ok",
            model_loaded=svc is not None,
            model_version=svc.model_version if svc else None,
        )

    @app.get(
        "/recommendations/{user_id}",
        response_model=RecommendationResponse,
        tags=["Inferência"],
    )
    def recommend(
        user_id: int,
        k: int = Query(10, ge=1, le=100),  # noqa: B008
    ) -> RecommendationResponse:
        """Retorna top-k recomendações para o usuário."""
        items, source = holder["service"].recommend(user_id, k)  # type: ignore[union-attr]
        RECOMMENDATIONS.labels(source=source).inc()
        return RecommendationResponse(
            user_id=user_id,
            k=k,
            items=items,
            model_version=holder["service"].model_version,  # type: ignore[union-attr]
            source=source,
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Expõe métricas Prometheus em formato texto."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", tags=["Info"])
    def root() -> dict[str, str]:
        """Endpoint raiz com links úteis."""
        return {"service": "recsys-api", "docs": "/docs", "health": "/health"}

    return app


app = create_app()
