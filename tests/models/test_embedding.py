"""Testes do EmbeddingRecommender (rede neural MLP da Etapa 4)."""

from __future__ import annotations

import pandas as pd

from recsys.models.embedding import EmbeddingRecommender


def _train() -> pd.DataFrame:
    # 12 usuários com padrões de interação distintos entre 8 itens.
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


def _model(**overrides: object) -> EmbeddingRecommender:
    params: dict = dict(emb_dim=4, hidden_dim=8, epochs=3, batch_size=16, neg_samples=2)
    params.update(overrides)
    return EmbeddingRecommender(**params)  # type: ignore[arg-type]


def test_recommend_returns_unseen_items() -> None:
    model = _model().fit(_train())
    recs = model.recommend(user_id=1, k=3)
    seen = {(1 + i) % 8 for i in range(4)}
    assert len(recs) <= 3
    assert all(item not in seen for item in recs)


def test_cold_start_user_returns_empty() -> None:
    model = _model().fit(_train())
    assert model.recommend(user_id=999, k=3) == []


def test_same_seed_produces_same_recommendations() -> None:
    # Reprodutibilidade ponta-a-ponta: mesma seed -> mesmo treino -> mesmas recs.
    train = _train()
    first = _model(seed=7).fit(train).recommend(1, 5)
    second = _model(seed=7).fit(train).recommend(1, 5)
    assert first == second


def test_records_validation_ndcg_aligned_with_train_loss() -> None:
    # Early stopping monitora NDCG de validação (ranking), não a loss BCE.
    model = _model(epochs=3, val_frac=0.3).fit(_train())
    assert len(model.val_ndcgs) == len(model.epoch_losses)
    assert len(model.epoch_losses) <= 3


def test_stops_early_when_validation_ndcg_does_not_improve() -> None:
    # patience baixa + muitas épocas: com seed fixa o treino para antes do teto.
    model = _model(epochs=100, patience=1, val_frac=0.3, seed=7).fit(_train())
    assert len(model.epoch_losses) < 100
