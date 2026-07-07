"""Testes do FallbackRecommender (cold-start fallback)."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.models.fallback import FallbackRecommender
from recsys.models.base import Recommender


class _StubRecommender(Recommender):
    """Recomendador stub para testes: retorna lista fixa ou vazia."""

    def __init__(self, recs: list[int]) -> None:
        self._recs = recs
        self.fit_called = False

    def fit(self, train: pd.DataFrame) -> "_StubRecommender":
        self.fit_called = True
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        return self._recs[:k]


def _df() -> pd.DataFrame:
    return pd.DataFrame({"user_id": [1], "item_id": [10]})


def test_returns_primary_recs_when_available() -> None:
    model = FallbackRecommender(
        primary=_StubRecommender([1, 2, 3]),
        fallback=_StubRecommender([9, 8, 7]),
    )
    assert model.recommend(user_id=1, k=3) == [1, 2, 3]


def test_falls_back_when_primary_returns_empty() -> None:
    model = FallbackRecommender(
        primary=_StubRecommender([]),
        fallback=_StubRecommender([9, 8, 7]),
    )
    assert model.recommend(user_id=999, k=3) == [9, 8, 7]


def test_cold_start_never_returns_empty_when_fallback_has_items() -> None:
    model = FallbackRecommender(
        primary=_StubRecommender([]),
        fallback=_StubRecommender([1, 2, 3, 4, 5]),
    )
    recs = model.recommend(user_id=999, k=5)
    assert len(recs) == 5


def test_fit_delegates_to_both() -> None:
    primary = _StubRecommender([1])
    fallback = _StubRecommender([2])
    FallbackRecommender(primary, fallback).fit(_df())
    assert primary.fit_called
    assert fallback.fit_called


def test_respects_k_from_fallback() -> None:
    model = FallbackRecommender(
        primary=_StubRecommender([]),
        fallback=_StubRecommender([1, 2, 3, 4, 5]),
    )
    assert len(model.recommend(user_id=999, k=2)) == 2


def test_falls_back_when_primary_raises_exception() -> None:
    class _BrokenRecommender(Recommender):
        def fit(self, train: pd.DataFrame) -> "_BrokenRecommender":
            return self

        def recommend(self, user_id: int, k: int) -> list[int]:
            raise RuntimeError("modelo não inicializado")

    model = FallbackRecommender(
        primary=_BrokenRecommender(),
        fallback=_StubRecommender([9, 8, 7]),
    )
    assert model.recommend(user_id=1, k=3) == [9, 8, 7]
