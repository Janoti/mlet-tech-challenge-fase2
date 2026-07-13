"""Serviço de modelo para inferência online.

Compõe o modelo primário (embedding em Production) com um fallback de
popularidade e reporta a origem de cada recomendação (observabilidade).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mlflow

from recsys.models.base import Recommender
from recsys.utils.logging_utils import get_logger, log_kv

_logger = get_logger(__name__)
_MODEL_NAME = "EmbeddingRecommender"
_EMBEDDING_PATH = Path("models/embedding.pkl")
_BASELINE_PATH = Path("models/baseline.pkl")


class ModelService:
    """Serve recomendações; cai no fallback quando o primário não responde.

    Attributes:
        model_version: Versão do modelo primário em Production.
    """

    def __init__(self, primary: Recommender, fallback: Recommender, model_version: str) -> None:
        """Recebe os modelos já carregados (injeção de dependência)."""
        self._primary = primary
        self._fallback = fallback
        self.model_version = model_version

    def recommend(self, user_id: int, k: int) -> tuple[list[int], str]:
        """Top-k do primário; se vazio (cold-start), usa o fallback.

        Returns:
            Tupla (items, source) com source em {"embedding", "fallback"}.
        """
        items = self._primary.recommend(user_id, k)
        if items:
            return items, "embedding"
        return self._fallback.recommend(user_id, k), "fallback"


def _load_production(name: str) -> tuple[Recommender, str]:
    """Baixa e desserializa a versão em Production do Registry.

    Args:
        name: Nome do modelo no MLflow Registry.

    Returns:
        Tupla (modelo, version) carregada do Registry Production.

    Raises:
        RuntimeError: Se nenhuma versão em Production está registrada.
    """
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(name, stages=["Production"])
    if not versions:
        raise RuntimeError(f"Nenhuma versão de '{name}' em Production no Registry.")
    mv = versions[0]
    local_path = mlflow.artifacts.download_artifacts(mv.source)
    with open(local_path, "rb") as fh:
        return pickle.load(fh), str(mv.version)


def load_model_service(
    name: str = _MODEL_NAME, baseline_path: Path = _BASELINE_PATH
) -> ModelService:
    """Carrega o primário do Registry e o fallback do disco, compondo o serviço.

    Args:
        name: Nome do modelo no Registry (padrão: EmbeddingRecommender).
        baseline_path: Caminho para o arquivo .pkl de fallback.

    Returns:
        ModelService pronto para inferência com ambos os modelos.
    """
    primary, version = _load_production(name)
    with baseline_path.open("rb") as fh:
        fallback = pickle.load(fh)
    log_kv(_logger, "model_service_loaded", name=name, version=version, source="registry")
    return ModelService(primary=primary, fallback=fallback, model_version=version)


def load_model_service_from_disk(
    embedding_path: Path = _EMBEDDING_PATH,
    baseline_path: Path = _BASELINE_PATH,
    version: str = "local",
) -> ModelService:
    """Carrega primário e fallback de arquivos .pkl no disco (deploy imutável).

    Alternativa ao Registry para imagens que bakeiam o modelo: sem dependência
    do MLflow no boot. Selecionada via ``MODEL_SOURCE=local``.

    Args:
        embedding_path: Caminho do .pkl do modelo primário (embedding).
        baseline_path: Caminho do .pkl do fallback (popularidade).
        version: Rótulo de versão reportado pelo serviço.

    Returns:
        ModelService pronto para inferência com ambos os modelos.
    """
    with embedding_path.open("rb") as fh:
        primary = pickle.load(fh)
    with baseline_path.open("rb") as fh:
        fallback = pickle.load(fh)
    log_kv(_logger, "model_service_loaded", version=version, source="disk")
    return ModelService(primary=primary, fallback=fallback, model_version=version)
