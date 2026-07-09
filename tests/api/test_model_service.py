"""Testes do ModelService (composição embedding + fallback)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

import recsys.api.model_service as ms
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


def test_load_model_service_composes_registry_and_baseline(tmp_path, monkeypatch) -> None:
    # primário vem do Registry (mockado); baseline é lido do disco.
    monkeypatch.setattr(ms, "_load_production", lambda name: (_Primary(), "7"))
    baseline_file = tmp_path / "baseline.pkl"
    baseline_file.write_bytes(pickle.dumps(_Fallback()))

    service = ms.load_model_service(baseline_path=Path(baseline_file))

    assert service.model_version == "7"
    assert service.recommend(999, 3) == ([9, 8, 7], "fallback")


def test_load_model_service_from_disk_composes(tmp_path) -> None:
    # deploy imutável: embedding e baseline lidos de arquivos .pkl no disco.
    embedding_file = tmp_path / "embedding.pkl"
    embedding_file.write_bytes(pickle.dumps(_Primary()))
    baseline_file = tmp_path / "baseline.pkl"
    baseline_file.write_bytes(pickle.dumps(_Fallback()))

    service = ms.load_model_service_from_disk(
        embedding_path=Path(embedding_file), baseline_path=Path(baseline_file), version="local"
    )

    assert service.model_version == "local"
    assert service.recommend(1, 3) == ([1, 2, 3], "embedding")
    assert service.recommend(999, 3) == ([9, 8, 7], "fallback")
