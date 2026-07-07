"""Stage ``evaluate_embedding``: métricas de ranking do modelo MLP (DVC metrics + MLflow)."""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd

from recsys.data.schema import COLUMN_ITEM_ID, COLUMN_USER_ID
from recsys.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recsys.models.base import Recommender
from recsys.pipeline.params import load_params
from recsys.registry import (
    MODEL_NAME,
    PRIMARY_METRIC,
    latest_staging_version,
    promote_to_production,
    should_promote,
)
from recsys.tracking import run
from recsys.utils.logging_utils import get_logger, log_kv, setup_logging

_MODEL = Path("models/embedding.pkl")
_TEST = Path("data/interim/test.parquet")
_METRICS = Path("metrics/metrics_embedding.json")
_BASELINE_METRICS = Path("metrics/metrics.json")


def _relevant_by_user(test: pd.DataFrame) -> dict[int, set[int]]:
    """Itens relevantes (vistos no teste) por usuário."""
    return {int(u): set(map(int, g[COLUMN_ITEM_ID])) for u, g in test.groupby(COLUMN_USER_ID)}


def _mean_metrics(model: Recommender, relevant: dict[int, set[int]], k: int) -> dict[str, float]:
    """Média das 4 métricas de ranking sobre os usuários do teste."""
    acc = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "map": 0.0}
    for user, rel in relevant.items():
        recs = model.recommend(user, k)
        acc["precision"] += precision_at_k(recs, rel, k)
        acc["recall"] += recall_at_k(recs, rel, k)
        acc["ndcg"] += ndcg_at_k(recs, rel, k)
        acc["map"] += average_precision_at_k(recs, rel, k)
    n = len(relevant) or 1
    return {f"{m}_at_k": v / n for m, v in acc.items()}


def _load_baseline() -> dict[str, float]:
    """Métricas do baseline (metrics.json), ou vazio se ainda não existir."""
    if not _BASELINE_METRICS.exists():
        return {}
    return json.loads(_BASELINE_METRICS.read_text(encoding="utf-8"))


def _maybe_promote(metrics: dict[str, float], logger: logging.Logger) -> None:
    """Promove a versão em Staging a Production se superar o baseline."""
    baseline = _load_baseline()
    if not should_promote(metrics, baseline, PRIMARY_METRIC):
        log_kv(logger, "promotion_skipped", metric=PRIMARY_METRIC, stage="Staging")
        return
    version = latest_staging_version(MODEL_NAME)
    if version is None:
        log_kv(logger, "promotion_skipped", reason="sem_versao_em_staging")
        return
    promote_to_production(MODEL_NAME, version)
    log_kv(logger, "model_promoted", name=MODEL_NAME, version=version, stage="Production")


def main() -> int:
    """Avalia o embedding model no teste, grava metrics_embedding.json e loga no MLflow."""
    setup_logging()
    logger = get_logger("recsys.pipeline.evaluate_embedding")
    k = load_params()["evaluate"]["top_k"]
    with _MODEL.open("rb") as fh:
        model = pickle.load(fh)
    test = pd.read_parquet(_TEST)
    metrics = _mean_metrics(model, _relevant_by_user(test), k)

    with run(experiment="recsys-ecommerce", run_name="embedding-eval"):
        mlflow.log_param("top_k", k)
        mlflow.log_metrics(metrics)
        _maybe_promote(metrics, logger)

    _METRICS.parent.mkdir(parents=True, exist_ok=True)
    _METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log_kv(logger, "evaluate_embedding_done", **{key: round(v, 4) for key, v in metrics.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
