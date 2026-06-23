"""Abstração de modelo de recomendação (ModelTrainer) — ponto de extensão.

A MLP PyTorch da Etapa 4 implementará esta mesma interface, entrando na
pipeline sem alterar os stages train/evaluate (OCP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Recommender(ABC):
    """Contrato mínimo de um recomendador top-k."""

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "Recommender":
        """Treina o modelo a partir das interações de treino."""

    @abstractmethod
    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna até ``k`` item_ids recomendados para ``user_id``."""
