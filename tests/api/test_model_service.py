"""Testes do ModelService (composição embedding + fallback)."""

from __future__ import annotations

import pandas as pd

from recsys.api.model_service import ModelService
from recsys.models.base import Recommender


class _Primary(Recommender):
    def fit(self, train: pd.DataFrame) -> _Primary:
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        return [1, 2, 3] if user_id == 1 else []  # user 1 conhecido; outros cold-start


class _Fallback(Recommender):
    def fit(self, train: pd.DataFrame) -> _Fallback:
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        return [9, 8, 7]


def _service() -> ModelService:
    return ModelService(primary=_Primary(), fallback=_Fallback(), model_version="2")


def test_known_user_served_by_embedding() -> None:
    items, source = _service().recommend(user_id=1, k=3)
    assert items == [1, 2, 3]
    assert source == "embedding"


def test_cold_start_falls_back_to_popularity() -> None:
    items, source = _service().recommend(user_id=999, k=3)
    assert items == [9, 8, 7]
    assert source == "fallback"
