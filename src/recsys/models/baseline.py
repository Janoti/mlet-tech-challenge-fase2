"""Baseline de popularidade (ModelTrainer). Implementa Recommender."""

from __future__ import annotations

import pandas as pd

from recsys.data.schema import COLUMN_ITEM_ID, COLUMN_USER_ID
from recsys.models.base import Recommender


class PopularityRecommender(Recommender):
    """Recomenda os itens globalmente mais populares ainda não vistos."""

    def __init__(self) -> None:
        """Inicializa o recomendador vazio (use ``fit`` antes de recomendar)."""
        self._popular: list[int] = []
        self._seen: dict[int, set[int]] = {}

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        """Ordena itens por frequência e memoriza itens vistos por usuário."""
        counts = train[COLUMN_ITEM_ID].value_counts()
        self._popular = [int(i) for i in counts.index]
        self._seen = {
            int(u): set(map(int, g[COLUMN_ITEM_ID]))
            for u, g in train.groupby(COLUMN_USER_ID)
        }
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Top-k itens populares que ``user_id`` ainda não viu."""
        seen = self._seen.get(int(user_id), set())
        recs = [item for item in self._popular if item not in seen]
        return recs[:k]

    @property
    def n_items(self) -> int:
        """Número de itens conhecidos pelo modelo (telemetria)."""
        return len(self._popular)
