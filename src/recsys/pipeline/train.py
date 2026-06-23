"""Stage ``train``: treina o baseline e registra run no MLflow."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd

from recsys.models.baseline import PopularityRecommender
from recsys.pipeline.params import load_params
from recsys.tracking import data_version, run
from recsys.utils.logging_utils import get_logger, log_kv, setup_logging

_TRAIN = Path("data/interim/train.parquet")
_POPULARITY = Path("data/processed/popularity.parquet")
_MODEL = Path("models/baseline.pkl")
_RAW = "data/raw/interactions.parquet"


def main() -> int:
    """Treina PopularityRecommender, serializa em models/ e loga no MLflow."""
    setup_logging()
    logger = get_logger("recsys.pipeline.train")
    params = load_params()
    train_df = pd.read_parquet(_TRAIN)
    popular_items = pd.read_parquet(_POPULARITY)["item_id"].tolist()

    with run(experiment="recsys-ecommerce", run_name="baseline-popularity"):
        mlflow.log_params(params["generate"])
        mlflow.log_param("model", "popularity")
        version = data_version(_RAW)
        if version:
            mlflow.set_tag("train_data_version", version)
        model = PopularityRecommender().fit(train_df, popular_items=popular_items)
        mlflow.log_metric("train_n_items", model.n_items)
        _MODEL.parent.mkdir(parents=True, exist_ok=True)
        with _MODEL.open("wb") as fh:
            pickle.dump(model, fh)
    log_kv(logger, "train_done", model_path=str(_MODEL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
