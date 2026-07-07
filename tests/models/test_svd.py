"""Testes do SvdRecommender (baseline scikit-learn — matrix factorization)."""

from __future__ import annotations

import pandas as pd

from recsys.models.svd import SvdRecommender


def _train() -> pd.DataFrame:
    users = [u for u in range(1, 13) for _ in range(4)]
    items = [(u + i) % 8 for u in range(1, 13) for i in range(4)]
    return pd.DataFrame(
        {
            "user_id": users,
            "item_id": items,
            "interaction_type": ["view"] * len(users),
            "timestamp": pd.date_range("2026-01-01", periods=len(users), freq="h"),
        }
    )


def test_recommend_returns_unseen_items() -> None:
    model = SvdRecommender(n_components=4).fit(_train())
    recs = model.recommend(user_id=1, k=3)
    seen = {(1 + i) % 8 for i in range(4)}
    assert len(recs) <= 3
    assert all(item not in seen for item in recs)


def test_cold_start_user_returns_empty() -> None:
    model = SvdRecommender(n_components=4).fit(_train())
    assert model.recommend(user_id=999, k=3) == []


def test_same_seed_produces_same_recommendations() -> None:
    train = _train()
    first = SvdRecommender(n_components=4, seed=7).fit(train).recommend(1, 5)
    second = SvdRecommender(n_components=4, seed=7).fit(train).recommend(1, 5)
    assert first == second


def test_n_items_reflects_catalogue() -> None:
    assert SvdRecommender(n_components=4).fit(_train()).n_items == 8
