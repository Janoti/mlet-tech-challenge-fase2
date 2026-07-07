"""Governança do MLflow Model Registry: registro em Staging e gate de promoção.

Fluxo Staging -> Production com gate de qualidade: o stage de treino registra
a nova versão em ``Staging``; o stage de avaliação só a promove a ``Production``
se ela superar o baseline na métrica primária. Evita promover automaticamente
um modelo pior — anti-padrão comum de MLOps.

Obs.: a API de stages está deprecada no MLflow 3.x (migra para aliases), mas é
a suportada no MLflow 2.x usado aqui e a esperada pelo enunciado.
"""

from __future__ import annotations

import mlflow

MODEL_NAME = "EmbeddingRecommender"
PRIMARY_METRIC = "ndcg_at_k"

_STAGING = "Staging"
_PRODUCTION = "Production"


def should_promote(candidate: dict[str, float], baseline: dict[str, float], metric: str) -> bool:
    """Decide se o candidato deve ser promovido a Production.

    Args:
        candidate: Métricas do modelo recém-avaliado.
        baseline: Métricas do modelo de referência (incumbente).
        metric: Métrica primária usada na decisão (maior é melhor).

    Returns:
        ``True`` se o candidato supera estritamente o baseline (ou se não há
        incumbente com essa métrica).
    """
    incumbent = baseline.get(metric)
    if incumbent is None:
        return True
    return candidate.get(metric, float("-inf")) > incumbent


def register_staging(model_uri: str, name: str) -> str:
    """Registra o modelo e o coloca em Staging. Retorna a versão criada."""
    version = mlflow.register_model(model_uri, name).version
    mlflow.MlflowClient().transition_model_version_stage(name=name, version=version, stage=_STAGING)
    return version


def latest_staging_version(name: str) -> str | None:
    """Versão mais recente do modelo em Staging, ou None se não houver."""
    versions = mlflow.MlflowClient().get_latest_versions(name, stages=[_STAGING])
    return versions[0].version if versions else None


def promote_to_production(name: str, version: str) -> None:
    """Promove a versão a Production, arquivando as versões anteriores."""
    mlflow.MlflowClient().transition_model_version_stage(
        name=name, version=version, stage=_PRODUCTION, archive_existing_versions=True
    )
