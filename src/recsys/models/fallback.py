"""Wrapper de fallback para cold-start em modelos de recomendação.

Quando o modelo primário não conhece o usuário (cold-start) e retorna lista
vazia, delega para o modelo de fallback — tipicamente o baseline de popularidade.
"""

from __future__ import annotations

import pandas as pd

from recsys.models.base import Recommender


class FallbackRecommender(Recommender):
    """Delega ao primário; usa fallback quando o primário retorna vazio.

    Attributes:
        primary: Modelo principal (ex.: EmbeddingRecommender).
        fallback: Modelo de reserva para cold-start (ex.: PopularityRecommender).
    """

    def __init__(self, primary: Recommender, fallback: Recommender) -> None:
        self._primary = primary
        self._fallback = fallback

    def fit(self, train: pd.DataFrame) -> FallbackRecommender:
        """Treina ambos os modelos com os mesmos dados."""
        self._primary.fit(train)
        self._fallback.fit(train)
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Top-k do primário; se vazio (cold-start), top-k do fallback.

        Args:
            user_id: ID do usuário.
            k: Número máximo de recomendações.

        Returns:
            Lista de até ``k`` item_ids.
        """
        try:
            recs = self._primary.recommend(user_id, k)
        except Exception:
            recs = []
        if not recs:
            return self._fallback.recommend(user_id, k)
        return recs
