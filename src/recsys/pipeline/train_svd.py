"""Stage ``train_svd``: treina o baseline SVD (scikit-learn) e loga no MLflow."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd

from recsys.models.svd import SvdRecommender
from recsys.pipeline.params import load_params
from recsys.tracking import data_version, run
from recsys.utils.logging_utils import get_logger, log_kv, setup_logging

_TRAIN = Path("data/interim/train.parquet")
_MODEL = Path("models/svd.pkl")
_RAW = "data/raw/interactions.parquet"


def main() -> int:
    """Treina SvdRecommender, serializa em models/ e loga no MLflow."""
    setup_logging()
    logger = get_logger("recsys.pipeline.train_svd")
    params = load_params()
    svd_params = params["svd"]
    train_df = pd.read_parquet(_TRAIN)

    with run(experiment="recsys-ecommerce", run_name="baseline-svd"):
        mlflow.log_params({**params["generate"], **svd_params})
        mlflow.log_param("model", "svd")
        version = data_version(_RAW)
        if version:
            mlflow.set_tag("train_data_version", version)
        model = SvdRecommender(**svd_params).fit(train_df)
        mlflow.log_metric("train_n_items", model.n_items)
        _MODEL.parent.mkdir(parents=True, exist_ok=True)
        with _MODEL.open("wb") as fh:
            pickle.dump(model, fh)
    log_kv(logger, "train_svd_done", model_path=str(_MODEL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
