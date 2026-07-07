"""Stage ``evaluate_svd``: métricas de ranking do baseline SVD (DVC + MLflow)."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd

from recsys.pipeline.evaluate import _mean_metrics, _relevant_by_user
from recsys.pipeline.params import load_params
from recsys.tracking import run
from recsys.utils.logging_utils import get_logger, log_kv, setup_logging

_MODEL = Path("models/svd.pkl")
_TEST = Path("data/interim/test.parquet")
_METRICS = Path("metrics/metrics_svd.json")


def main() -> int:
    """Avalia o baseline SVD no teste, grava metrics_svd.json e loga no MLflow."""
    setup_logging()
    logger = get_logger("recsys.pipeline.evaluate_svd")
    k = load_params()["evaluate"]["top_k"]
    with _MODEL.open("rb") as fh:
        model = pickle.load(fh)
    test = pd.read_parquet(_TEST)
    metrics = _mean_metrics(model, _relevant_by_user(test), k)

    with run(experiment="recsys-ecommerce", run_name="svd-eval"):
        mlflow.log_param("top_k", k)
        mlflow.log_metrics(metrics)

    _METRICS.parent.mkdir(parents=True, exist_ok=True)
    _METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log_kv(logger, "evaluate_svd_done", **{key: round(v, 4) for key, v in metrics.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
