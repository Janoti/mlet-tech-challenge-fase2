"""Camada fina sobre o MLflow — isola o tracking do núcleo de ML.

Mantém a Regra de Dependência: os módulos de domínio (preprocessing, models,
evaluation) não importam MLflow; só os stages e este helper conhecem o MLflow.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator

import mlflow

_DEFAULT_URI = "./mlruns"
_logger = logging.getLogger(__name__)


def get_tracking_uri() -> str:
    """URI do MLflow: ``MLFLOW_TRACKING_URI`` ou ``./mlruns`` local."""
    return os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_URI)


def data_version(path: str) -> str | None:
    """Versão DVC do arquivo de dados (para cross-link), ou None se indisponível."""
    try:
        import dvc.api

        return dvc.api.get_url(path)
    except Exception as exc:  # noqa: BLE001 — cross-link é best-effort, não pode quebrar o treino
        _logger.warning("cross-link DVC indisponível para %s: %s", path, exc)
        return None


@contextlib.contextmanager
def run(experiment: str, run_name: str) -> Iterator[None]:
    """Abre um run do MLflow configurando URI e experimento."""
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        yield
