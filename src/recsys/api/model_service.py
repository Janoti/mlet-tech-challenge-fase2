"""Serviço de modelo para inferência online.

Compõe o modelo primário (embedding em Production) com um fallback de
popularidade e reporta a origem de cada recomendação (observabilidade).
"""

from __future__ import annotations

from recsys.models.base import Recommender


class ModelService:
    """Serve recomendações; cai no fallback quando o primário não responde.

    Attributes:
        model_version: Versão do modelo primário em Production.
    """

    def __init__(self, primary: Recommender, fallback: Recommender, model_version: str) -> None:
        """Recebe os modelos já carregados (injeção de dependência)."""
        self._primary = primary
        self._fallback = fallback
        self.model_version = model_version

    def recommend(self, user_id: int, k: int) -> tuple[list[int], str]:
        """Top-k do primário; se vazio (cold-start), usa o fallback.

        Returns:
            Tupla (items, source) com source em {"embedding", "fallback"}.
        """
        items = self._primary.recommend(user_id, k)
        if items:
            return items, "embedding"
        return self._fallback.recommend(user_id, k), "fallback"
