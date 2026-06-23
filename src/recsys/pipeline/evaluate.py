"""Stage ``evaluate``: métricas de ranking do baseline (DVC metrics + MLflow)."""

from __future__ import annotations

import json
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
from recsys.tracking import run
from recsys.utils.logging_utils import get_logger, log_kv, setup_logging

_MODEL = Path("models/baseline.pkl")
_TEST = Path("data/interim/test.parquet")
_METRICS = Path("metrics/metrics.json")


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


def main() -> int:
    """Avalia o baseline no teste, grava metrics.json e loga no MLflow."""
    setup_logging()
    logger = get_logger("recsys.pipeline.evaluate")
    k = load_params()["evaluate"]["top_k"]
    with _MODEL.open("rb") as fh:
        model = pickle.load(fh)
    test = pd.read_parquet(_TEST)
    metrics = _mean_metrics(model, _relevant_by_user(test), k)

    with run(experiment="recsys-ecommerce", run_name="baseline-eval"):
        mlflow.log_param("top_k", k)
        mlflow.log_metrics(metrics)

    _METRICS.parent.mkdir(parents=True, exist_ok=True)
    _METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log_kv(logger, "evaluate_done", **{key: round(v, 4) for key, v in metrics.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
