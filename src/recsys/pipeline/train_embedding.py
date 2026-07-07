"""Stage ``train_embedding``: treina MLP com embeddings e loga no MLflow."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd

from recsys.models.embedding import EmbeddingRecommender
from recsys.pipeline.params import load_params
from recsys.registry import MODEL_NAME, register_staging
from recsys.tracking import data_version, run
from recsys.utils.logging_utils import get_logger, log_kv, setup_logging

_TRAIN = Path("data/interim/train.parquet")
_MODEL = Path("models/embedding.pkl")
_RAW = "data/raw/interactions.parquet"


def main() -> int:
    """Treina EmbeddingRecommender, serializa em models/ e loga no MLflow."""
    setup_logging()
    logger = get_logger("recsys.pipeline.train_embedding")
    params = load_params()
    emb_params = params["embedding"]
    train_df = pd.read_parquet(_TRAIN)

    with run(experiment="recsys-ecommerce", run_name="embedding-mlp"):
        mlflow.log_params({**params["generate"], **emb_params})
        mlflow.log_param("model", "embedding-mlp")
        version = data_version(_RAW)
        if version:
            mlflow.set_tag("train_data_version", version)

        model = EmbeddingRecommender(**emb_params).fit(train_df)

        for epoch, loss in enumerate(model.epoch_losses):
            mlflow.log_metric("train_loss", loss, step=epoch)
        for epoch, loss in enumerate(model.val_losses):
            mlflow.log_metric("val_loss", loss, step=epoch)
        mlflow.log_metric("epochs_trained", len(model.epoch_losses))
        mlflow.log_metric("train_n_users", model.n_users)
        mlflow.log_metric("train_n_items", model.n_items)

        _MODEL.parent.mkdir(parents=True, exist_ok=True)
        with _MODEL.open("wb") as fh:
            pickle.dump(model, fh)

        mlflow.log_artifact(str(_MODEL), artifact_path="model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model/embedding.pkl"

        version = register_staging(model_uri, MODEL_NAME)
        log_kv(
            logger,
            "model_registered",
            name=MODEL_NAME,
            version=version,
            stage="Staging",
        )

    log_kv(logger, "train_embedding_done", model_path=str(_MODEL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
