import pandas as pd

from recsys.models.baseline import PopularityRecommender


def _train():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 3],
            "item_id": [10, 20, 10, 30, 10, 20, 30],
            "interaction_type": ["view"] * 7,
            "timestamp": pd.date_range("2026-01-01", periods=7, freq="h"),
        }
    )


def test_recommends_popular_unseen():
    model = PopularityRecommender().fit(_train())
    # usuário 1 já viu 10 e 20 -> recebe 30 (próximo mais popular não visto)
    assert model.recommend(user_id=1, k=1) == [30]


def test_respects_k_and_cold_start_values():
    model = PopularityRecommender().fit(_train())
    # usuário novo (cold start): top-2 mais populares na ordem [10, 20]
    assert model.recommend(user_id=99, k=2) == [10, 20]


def test_n_items():
    assert PopularityRecommender().fit(_train()).n_items == 3


def test_fit_with_precomputed_popularity_uses_given_order():
    # popular_items vindo do stage feature_eng define a ordem do ranking.
    model = PopularityRecommender().fit(_train(), popular_items=[30, 20, 10])
    assert model.recommend(user_id=99, k=3) == [30, 20, 10]
