import pandas as pd

from recsys.features.build_features import build_popularity_features


def test_popularity_counts_desc():
    train = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "item_id": [10, 10, 10, 20, 30],
            "interaction_type": ["view"] * 5,
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h"),
        }
    )
    feats = build_popularity_features(train)
    assert list(feats["item_id"]) == [10, 20, 30]
    assert list(feats["popularity_count"]) == [3, 1, 1]
