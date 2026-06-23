"""Engenharia de atributos: contagens de popularidade por item (DataPipeline)."""

from __future__ import annotations

import pandas as pd

from recsys.data.schema import COLUMN_ITEM_ID

POPULARITY_COUNT: str = "popularity_count"


def build_popularity_features(train: pd.DataFrame) -> pd.DataFrame:
    """Conta interações por item, ordenando do mais para o menos popular.

    Args:
        train: Interações de treino.

    Returns:
        DataFrame com ``item_id`` e ``popularity_count`` (ordem decrescente;
        empates desempatados por ``item_id`` crescente para determinismo).
    """
    counts = train.groupby(COLUMN_ITEM_ID).size().rename(POPULARITY_COUNT).reset_index()
    return counts.sort_values(
        [POPULARITY_COUNT, COLUMN_ITEM_ID], ascending=[False, True]
    ).reset_index(drop=True)
