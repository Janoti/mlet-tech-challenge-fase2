import pandas as pd

from recsys.preprocessing.splitter import temporal_split


def _df():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "item_id": [1] * 10,
            "interaction_type": ["view"] * 10,
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="D"),
        }
    )


def test_temporal_split_sizes():
    train, test = temporal_split(_df(), test_size=0.2)
    assert len(train) == 8
    assert len(test) == 2


def test_temporal_split_no_leakage():
    train, test = temporal_split(_df(), test_size=0.3)
    assert test["timestamp"].min() >= train["timestamp"].max()
