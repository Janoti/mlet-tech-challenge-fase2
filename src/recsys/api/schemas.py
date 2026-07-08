"""Schemas Pydantic de request/response da API de recomendação."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationResponse(BaseModel):
    """Resposta do endpoint de recomendações."""

    user_id: int = Field(..., description="ID do usuário consultado.")
    k: int = Field(..., description="Número de itens solicitados.")
    items: list[int] = Field(..., description="Item_ids recomendados, ordem decrescente de score.")
    model_version: str = Field(..., description="Versão do modelo em Production servida.")
    source: str = Field(
        ..., description="Origem: 'embedding' (primário) ou 'fallback' (popularidade)."
    )


class HealthResponse(BaseModel):
    """Resposta do health check."""

    status: str = Field(..., description="'ok' se a API está operacional.")
    model_loaded: bool = Field(..., description="True se o modelo foi carregado.")
    model_version: str | None = Field(None, description="Versão servida, ou None se não carregado.")
