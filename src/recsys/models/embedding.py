"""MLP com embeddings para recomendação colaborativa (Etapa 4).

Arquitetura:
    user_emb(user_idx) ─┐
                         ├─ concat ─► Linear → ReLU → Linear → score
    item_emb(item_idx) ─┘

Treinamento com negative sampling implícito: para cada interação positiva,
amostram-se ``neg_samples`` itens aleatórios do catálogo como negativos.
Loss: BCEWithLogitsLoss (inclui sigmoid internamente — mais estável).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from recsys.data.schema import COLUMN_ITEM_ID, COLUMN_USER_ID
from recsys.models.base import Recommender
from recsys.preprocessing.encoder import IdEncoder


class MLPScorer(nn.Module):
    """User+item embeddings concatenados e passados por MLP para score de relevância."""

    def __init__(self, n_users: int, n_items: int, emb_dim: int, hidden_dim: int) -> None:
        """Inicializa camadas de embedding e MLP.

        Args:
            n_users: Número de usuários (tamanho da tabela de embedding).
            n_items: Número de itens (tamanho da tabela de embedding).
            emb_dim: Dimensão dos vetores de embedding.
            hidden_dim: Número de neurônios na camada oculta.
        """
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Calcula logits de relevância para pares (user, item).

        Args:
            user_idx: Índices de usuários, shape (B,).
            item_idx: Índices de itens, shape (B,).

        Returns:
            Logits de relevância, shape (B,).
        """
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        return self.mlp(torch.cat([u, i], dim=-1)).squeeze(-1)


class EmbeddingRecommender(Recommender):
    """Treina MLPScorer com negative sampling; top-k via score sobre todos os itens.

    Implementa a interface ``Recommender`` (OCP): entra no pipeline DVC/MLflow
    sem alterar os stages de evaluate.

    Attributes:
        emb_dim: Dimensão dos embeddings de usuário e item.
        hidden_dim: Tamanho da camada oculta do MLP.
        lr: Taxa de aprendizado do Adam.
        epochs: Número de épocas de treino.
        batch_size: Tamanho do mini-batch.
        neg_samples: Amostras negativas por interação positiva.
        device: Dispositivo PyTorch (``"cpu"`` ou ``"cuda"``).
    """

    def __init__(
        self,
        emb_dim: int = 32,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 1024,
        neg_samples: int = 4,
        device: str = "cpu",
    ) -> None:
        """Configura hiperparâmetros; modelo é criado em ``fit``."""
        self._emb_dim = emb_dim
        self._hidden_dim = hidden_dim
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._neg_samples = neg_samples
        self._device = torch.device(device)
        self._model: MLPScorer | None = None
        self._user_enc = IdEncoder()
        self._item_enc = IdEncoder()
        self._seen: dict[int, set[int]] = {}
        self.epoch_losses: list[float] = []

    def fit(self, train: pd.DataFrame) -> "EmbeddingRecommender":
        """Treina o MLPScorer com pares positivos e amostras negativas aleatórias.

        Args:
            train: DataFrame de interações de treino com ``user_id`` e ``item_id``.

        Returns:
            Self (fluent interface).
        """
        self._user_enc.fit(train[COLUMN_USER_ID])
        self._item_enc.fit(train[COLUMN_ITEM_ID])

        self._seen = {
            int(u): set(map(int, g[COLUMN_ITEM_ID]))
            for u, g in train.groupby(COLUMN_USER_ID)
        }

        pos_users = self._user_enc.transform(train[COLUMN_USER_ID])
        pos_items = self._item_enc.transform(train[COLUMN_ITEM_ID])
        n_pos = len(pos_users)
        n_items = self._item_enc.n_ids

        rng = np.random.default_rng(42)
        neg_items = rng.integers(0, n_items, size=n_pos * self._neg_samples)
        neg_users = np.repeat(pos_users, self._neg_samples)

        all_users = np.concatenate([pos_users, neg_users])
        all_items = np.concatenate([pos_items, neg_items])
        all_labels = np.concatenate([
            np.ones(n_pos, dtype=np.float32),
            np.zeros(n_pos * self._neg_samples, dtype=np.float32),
        ])

        dataset = TensorDataset(
            torch.tensor(all_users, dtype=torch.long),
            torch.tensor(all_items, dtype=torch.long),
            torch.tensor(all_labels, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        n_users = self._user_enc.n_ids
        self._model = MLPScorer(n_users, n_items, self._emb_dim, self._hidden_dim).to(
            self._device
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.BCEWithLogitsLoss()

        self._model.train()
        self.epoch_losses = []
        for _epoch in range(self._epochs):
            total_loss = 0.0
            for u_batch, i_batch, y_batch in loader:
                u_batch = u_batch.to(self._device)
                i_batch = i_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._model(u_batch, i_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_batch)
            self.epoch_losses.append(total_loss / len(all_labels))

        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Top-k itens não vistos, ordenados por score decrescente.

        Args:
            user_id: ID original do usuário (antes do encoding).
            k: Número máximo de recomendações.

        Returns:
            Lista de até ``k`` item_ids. Retorna lista vazia para cold-start.
        """
        if self._model is None:
            raise RuntimeError("Chame fit() antes de recommend().")

        try:
            user_idx = int(self._user_enc.transform([user_id])[0])
        except KeyError:
            return []

        n_items = self._item_enc.n_ids
        seen = self._seen.get(int(user_id), set())

        self._model.eval()
        with torch.no_grad():
            u_tensor = torch.full((n_items,), user_idx, dtype=torch.long, device=self._device)
            i_tensor = torch.arange(n_items, dtype=torch.long, device=self._device)
            scores = torch.sigmoid(self._model(u_tensor, i_tensor)).cpu().numpy()

        ranked_indices = np.argsort(-scores)
        item_ids = self._item_enc.inverse_transform(ranked_indices)
        return [int(iid) for iid in item_ids if int(iid) not in seen][:k]

    @property
    def n_users(self) -> int:
        """Número de usuários conhecidos pelo modelo."""
        return self._user_enc.n_ids

    @property
    def n_items(self) -> int:
        """Número de itens conhecidos pelo modelo."""
        return self._item_enc.n_ids
