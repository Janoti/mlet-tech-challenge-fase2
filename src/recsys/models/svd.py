"""Baseline de matrix factorization com scikit-learn (TruncatedSVD).

Fatoriza a matriz de interações user-item em fatores latentes. É o baseline
competitivo (personalizado, porém linear) contra o qual a rede neural MLP é
comparada — isola o ganho atribuível à não-linearidade do embedding.

Implementa a interface ``Recommender`` (OCP): entra no pipeline DVC/MLflow
sem alterar os stages de avaliação.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from recsys.data.schema import COLUMN_ITEM_ID, COLUMN_USER_ID
from recsys.models.base import Recommender
from recsys.preprocessing.encoder import IdEncoder


class SvdRecommender(Recommender):
    """Recomenda por produto interno de fatores latentes (SVD truncado).

    Attributes:
        n_components: Número de fatores latentes (dimensão do espaço).
        seed: Seed do TruncatedSVD para reprodutibilidade.
    """

    def __init__(self, n_components: int = 32, seed: int = 42) -> None:
        """Configura hiperparâmetros; fatores são aprendidos em ``fit``."""
        self._n_components = n_components
        self._seed = seed
        self._user_enc = IdEncoder()
        self._item_enc = IdEncoder()
        self._seen: dict[int, set[int]] = {}
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> SvdRecommender:
        """Aprende fatores latentes de usuários e itens via TruncatedSVD.

        Args:
            train: Interações de treino com ``user_id`` e ``item_id``.

        Returns:
            Self (fluent interface).
        """
        self._user_enc.fit(train[COLUMN_USER_ID])
        self._item_enc.fit(train[COLUMN_ITEM_ID])
        self._seen = {
            int(u): set(map(int, g[COLUMN_ITEM_ID])) for u, g in train.groupby(COLUMN_USER_ID)
        }
        matrix = self._interaction_matrix(train)
        n_comp = min(self._n_components, min(matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=self._seed)
        self._user_factors = svd.fit_transform(matrix)
        self._item_factors = svd.components_.T
        return self

    def _interaction_matrix(self, train: pd.DataFrame) -> np.ndarray:
        """Matriz densa user×item com a contagem de interações."""
        matrix = np.zeros((self._user_enc.n_ids, self._item_enc.n_ids), dtype=np.float32)
        users = self._user_enc.transform(train[COLUMN_USER_ID])
        items = self._item_enc.transform(train[COLUMN_ITEM_ID])
        np.add.at(matrix, (users, items), 1.0)
        return matrix

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Top-k itens não vistos, ordenados por score latente decrescente.

        Args:
            user_id: ID original do usuário (antes do encoding).
            k: Número máximo de recomendações.

        Returns:
            Lista de até ``k`` item_ids. Vazia para cold-start.
        """
        if self._user_factors is None:
            raise RuntimeError("Chame fit() antes de recommend().")
        try:
            user_idx = int(self._user_enc.transform([user_id])[0])
        except KeyError:
            return []
        scores = self._item_factors @ self._user_factors[user_idx]
        seen = self._seen.get(int(user_id), set())
        ranked = np.argsort(-scores)
        item_ids = self._item_enc.inverse_transform(ranked)
        return [int(iid) for iid in item_ids if int(iid) not in seen][:k]

    @property
    def n_items(self) -> int:
        """Número de itens conhecidos pelo modelo."""
        return self._item_enc.n_ids
