"""Stage ``preprocess``: split temporal train/test (data/interim)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from recsys.pipeline.params import load_params
from recsys.preprocessing.splitter import temporal_split
from recsys.utils.logging_utils import get_logger, log_kv, setup_logging

_RAW = Path("data/raw/interactions.parquet")
_INTERIM = Path("data/interim")


def main() -> int:
    """Lê o raw, faz split temporal e grava train/test em data/interim."""
    setup_logging()
    logger = get_logger("recsys.pipeline.preprocess")
    test_size = load_params()["preprocess"]["test_size"]
    df = pd.read_parquet(_RAW)
    train, test = temporal_split(df, test_size=test_size)
    _INTERIM.mkdir(parents=True, exist_ok=True)
    train.to_parquet(_INTERIM / "train.parquet", index=False)
    test.to_parquet(_INTERIM / "test.parquet", index=False)
    log_kv(logger, "preprocess_done", train=len(train), test=len(test))
    return 0


if __name__ == "__main__":
    sys.exit(main())
