"""Stage ``feature_eng``: features de popularidade (data/processed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from recsys.features.build_features import build_popularity_features
from recsys.utils.logging_utils import get_logger, log_kv, setup_logging

_TRAIN = Path("data/interim/train.parquet")
_PROCESSED = Path("data/processed")


def main() -> int:
    """Constrói features de popularidade a partir do train e grava em processed."""
    setup_logging()
    logger = get_logger("recsys.pipeline.feature_eng")
    train = pd.read_parquet(_TRAIN)
    feats = build_popularity_features(train)
    _PROCESSED.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(_PROCESSED / "popularity.parquet", index=False)
    log_kv(logger, "features_done", items=len(feats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
