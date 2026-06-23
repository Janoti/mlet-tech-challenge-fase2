import pandas as pd
import pytest
import yaml

from recsys.pipeline import feature_eng, generate, preprocess


@pytest.fixture
def mini_project(tmp_path, monkeypatch):
    (tmp_path / "params.yaml").write_text(
        yaml.safe_dump(
            {
                "generate": {
                    "seed": 1,
                    "num_users": 50,
                    "num_items": 20,
                    "num_interactions": 10000,
                },
                "preprocess": {"test_size": 0.2},
                "evaluate": {"top_k": 5},
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_generate_preprocess_features(mini_project):
    assert generate.main() == 0
    assert (mini_project / "data/raw/interactions.parquet").exists()

    assert preprocess.main() == 0
    train = pd.read_parquet(mini_project / "data/interim/train.parquet")
    test = pd.read_parquet(mini_project / "data/interim/test.parquet")
    assert len(train) + len(test) == 10000

    assert feature_eng.main() == 0
    feats = pd.read_parquet(mini_project / "data/processed/popularity.parquet")
    assert list(feats.columns) == ["item_id", "popularity_count"]
