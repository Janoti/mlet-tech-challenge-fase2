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
from recsys.evaluation.metrics import ndcg_at_k
from recsys.models.base import Recommender
from recsys.models.early_stopping import EarlyStopping
from recsys.preprocessing.encoder import IdEncoder
from recsys.utils.seed import set_global_seed


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
        epochs: Número máximo de épocas de treino.
        batch_size: Tamanho do mini-batch.
        neg_samples: Amostras negativas por interação positiva.
        device: Dispositivo PyTorch (``"cpu"`` ou ``"cuda"``).
        seed: Seed global para reprodutibilidade do treino.
        patience: Épocas sem melhora no NDCG de validação antes de parar.
        val_frac: Fração das interações reservada para validação de ranking.
        val_k: Corte top-k usado no NDCG de validação.
        epoch_losses: Loss de treino por época.
        val_ndcgs: NDCG de validação por época (critério do early stopping).
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
        seed: int = 42,
        patience: int = 3,
        val_frac: float = 0.1,
        val_k: int = 10,
    ) -> None:
        """Configura hiperparâmetros; modelo é criado em ``fit``."""
        self._emb_dim = emb_dim
        self._hidden_dim = hidden_dim
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._neg_samples = neg_samples
        self._device = torch.device(device)
        self._seed = seed
        self._patience = patience
        self._val_frac = val_frac
        self._val_k = val_k
        self._model: MLPScorer | None = None
        self._user_enc = IdEncoder()
        self._item_enc = IdEncoder()
        self._seen: dict[int, set[int]] = {}
        self._val_seen: dict[int, set[int]] = {}
        self._val_relevant: dict[int, set[int]] = {}
        self.epoch_losses: list[float] = []
        self.val_ndcgs: list[float] = []

    def fit(self, train: pd.DataFrame) -> EmbeddingRecommender:
        """Treina o MLPScorer com negative sampling e early stopping por NDCG.

        Reserva ``val_frac`` das interações para validação de *ranking*: a
        parada monitora o NDCG@k de validação (o objetivo real), não a loss
        BCE — que é um proxy ruim para ranking top-k.

        Args:
            train: DataFrame de interações de treino com ``user_id`` e ``item_id``.

        Returns:
            Self (fluent interface).
        """
        set_global_seed(self._seed)
        train_fit, val = self._split_interactions(train)
        self._fit_encoders(train, train_fit, val)
        users, items, labels = self._build_pairs(train_fit)
        loader = self._make_loader(users, items, labels)
        self._model = MLPScorer(
            self._user_enc.n_ids, self._item_enc.n_ids, self._emb_dim, self._hidden_dim
        ).to(self._device)
        self._train_model(loader)
        return self

    def _split_interactions(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Separa interações em treino-de-ajuste e validação (split por linha)."""
        rng = np.random.default_rng(self._seed)
        is_val = rng.random(len(train)) < self._val_frac
        return train[~is_val], train[is_val]

    def _fit_encoders(self, full: pd.DataFrame, train_fit: pd.DataFrame, val: pd.DataFrame) -> None:
        """Ajusta encoders (catálogo completo) e prepara os conjuntos de validação."""
        self._user_enc.fit(full[COLUMN_USER_ID])
        self._item_enc.fit(full[COLUMN_ITEM_ID])
        self._seen = self._items_by_user(full)
        self._val_seen = self._items_by_user(train_fit)
        self._val_relevant = self._items_by_user(val)

    @staticmethod
    def _items_by_user(df: pd.DataFrame) -> dict[int, set[int]]:
        """Mapa usuário -> conjunto de itens com que interagiu."""
        return {int(u): set(map(int, g[COLUMN_ITEM_ID])) for u, g in df.groupby(COLUMN_USER_ID)}

    def _build_pairs(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Monta pares (user, item, label) com negative sampling determinístico."""
        pos_users = self._user_enc.transform(train[COLUMN_USER_ID])
        pos_items = self._item_enc.transform(train[COLUMN_ITEM_ID])
        n_pos = len(pos_users)
        rng = np.random.default_rng(self._seed)
        neg_items = rng.integers(0, self._item_enc.n_ids, size=n_pos * self._neg_samples)
        neg_users = np.repeat(pos_users, self._neg_samples)
        labels = np.concatenate(
            [
                np.ones(n_pos, dtype=np.float32),
                np.zeros(n_pos * self._neg_samples, dtype=np.float32),
            ]
        )
        return (
            np.concatenate([pos_users, neg_users]),
            np.concatenate([pos_items, neg_items]),
            labels,
        )

    def _make_loader(self, users: np.ndarray, items: np.ndarray, labels: np.ndarray) -> DataLoader:
        """Empacota os pares num DataLoader de treino (embaralhado)."""
        dataset = TensorDataset(
            torch.tensor(users, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

    def _train_model(self, loader: DataLoader) -> None:
        """Loop de treino com early stopping por NDCG; restaura os melhores pesos."""
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.BCEWithLogitsLoss()
        stopper = EarlyStopping(patience=self._patience)
        best_state, best_ndcg = None, -1.0
        self.epoch_losses, self.val_ndcgs = [], []
        for _epoch in range(self._epochs):
            self.epoch_losses.append(self._run_epoch(loader, optimizer, criterion))
            ndcg = self._val_ndcg()
            self.val_ndcgs.append(ndcg)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            if stopper.update(-ndcg):  # EarlyStopping minimiza -> monitora -NDCG
                break
        if best_state is not None:
            self._model.load_state_dict(best_state)

    def _run_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module
    ) -> float:
        """Executa uma época de treino e devolve a loss média."""
        self._model.train()
        total, n = 0.0, 0
        for u_batch, i_batch, y_batch in loader:
            u_batch, i_batch, y_batch = self._to_device(u_batch, i_batch, y_batch)
            optimizer.zero_grad()
            loss = criterion(self._model(u_batch, i_batch), y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y_batch)
            n += len(y_batch)
        return total / max(n, 1)

    def _val_ndcg(self) -> float:
        """NDCG@val_k médio nos usuários de validação (itens de treino excluídos)."""
        total = 0.0
        for user, relevant in self._val_relevant.items():
            recs = self._recommend(user, self._val_k, exclude=self._val_seen.get(user, set()))
            total += ndcg_at_k(recs, relevant, self._val_k)
        return total / max(len(self._val_relevant), 1)

    def _to_device(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Move os tensores do batch para o dispositivo do modelo."""
        return tuple(t.to(self._device) for t in tensors)

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Top-k itens não vistos no treino, ordenados por score decrescente.

        Args:
            user_id: ID original do usuário (antes do encoding).
            k: Número máximo de recomendações.

        Returns:
            Lista de até ``k`` item_ids. Retorna lista vazia para cold-start.
        """
        return self._recommend(user_id, k, exclude=self._seen.get(int(user_id), set()))

    def _recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        """Top-k itens fora de ``exclude``, ordenados por score do modelo."""
        if self._model is None:
            raise RuntimeError("Chame fit() antes de recommend().")
        try:
            user_idx = int(self._user_enc.transform([user_id])[0])
        except KeyError:
            return []
        scores = self._score_all_items(user_idx)
        ranked = np.argsort(-scores)
        item_ids = self._item_enc.inverse_transform(ranked)
        return [int(iid) for iid in item_ids if int(iid) not in exclude][:k]

    def _score_all_items(self, user_idx: int) -> np.ndarray:
        """Score (sigmoid) do usuário contra todos os itens do catálogo."""
        n_items = self._item_enc.n_ids
        self._model.eval()
        with torch.no_grad():
            u_tensor = torch.full((n_items,), user_idx, dtype=torch.long, device=self._device)
            i_tensor = torch.arange(n_items, dtype=torch.long, device=self._device)
            return torch.sigmoid(self._model(u_tensor, i_tensor)).cpu().numpy()

    @property
    def n_users(self) -> int:
        """Número de usuários conhecidos pelo modelo."""
        return self._user_enc.n_ids

    @property
    def n_items(self) -> int:
        """Número de itens conhecidos pelo modelo."""
        return self._item_enc.n_ids
