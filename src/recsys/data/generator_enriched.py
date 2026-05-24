"""Gerador enriquecido: sazonalidade semanal, categorias de produto e gênero.

Versão alternativa ao ``generator.py`` que adiciona três dimensões de realismo:

- **Sazonalidade semanal**: fins de semana concentram 2.5× mais tráfego por
  dia e têm funil com maior taxa de purchase (6% vs 2% em dias úteis).
- **Categoria de produto**: cada item pertence a uma das 5 categorias;
  permite análise por segmento no EDA e uso como feature no modelo.
- **Gênero do usuário**: atributo fixo por usuário; útil para análise de
  fairness no Model Card (alinhado com o que foi feito na Fase 1).

Mantém as mesmas convenções do ``generator.py``:
    - Funções ≤ 20 linhas.
    - Constantes nomeadas em vez de magic numbers.
    - Reprodutibilidade bit-a-bit: mesma seed → mesmo DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import pandas as pd

from recsys.utils.logging_utils import get_logger

_logger = get_logger(__name__)

# ── Catálogo de categorias ────────────────────────────────────────────────────
CATEGORIES: Final[tuple[str, ...]] = ("eletronicos", "moda", "casa", "esportes", "beleza")

# ── Funil de interações por período ──────────────────────────────────────────
# (view_prob, add_to_cart_prob, purchase_prob) — soma = 1.0 em cada tupla.
_WEEKDAY_FUNNEL: Final[tuple[float, float, float]] = (0.87, 0.11, 0.02)
_WEEKEND_FUNNEL: Final[tuple[float, float, float]] = (0.80, 0.14, 0.06)

# Fins de semana recebem 2.5× mais tráfego por dia que dias úteis.
_WEEKEND_TRAFFIC_WEIGHT: Final[float] = 2.5

# ── Gênero do usuário ─────────────────────────────────────────────────────────
GENDERS: Final[tuple[str, ...]] = ("M", "F", "NB")
_GENDER_PROBS: Final[tuple[float, float, float]] = (0.48, 0.47, 0.05)

# Ordem fixa para rng.choice (deve coincidir com os funis acima).
_INTERACTION_VALUES: Final[list[str]] = ["view", "add_to_cart", "purchase"]

_SECONDS_PER_DAY: Final[int] = 24 * 3600


@dataclass(frozen=True)
class EnrichedGenerationConfig:
    """Parâmetros para o gerador enriquecido.

    Attributes:
        num_users: Quantidade de usuários distintos.
        num_items: Quantidade de itens distintos no catálogo.
        num_interactions: Total de interações (>= 10_000 por requisito).
        seed: Semente para reprodutibilidade bit-a-bit.
        time_window_days: Janela temporal em dias.
        item_skew: Expoente Zipf para popularidade de itens (> 1.0).
        user_skew: Expoente Zipf para atividade de usuários (> 1.0).
    """

    num_users: int
    num_items: int
    num_interactions: int
    seed: int = 42
    time_window_days: int = 90
    item_skew: float = 1.2
    user_skew: float = 1.1

    def __post_init__(self) -> None:
        """Valida parâmetros no construtor — falha cedo."""
        if self.num_users <= 0 or self.num_items <= 0:
            raise ValueError("num_users e num_items devem ser positivos")
        if self.num_interactions < 10_000:
            raise ValueError("num_interactions deve ser >= 10.000 (requisito)")
        if self.item_skew <= 1.0 or self.user_skew <= 1.0:
            raise ValueError("item_skew e user_skew devem ser > 1.0")


class EnrichedDatasetGenerator:
    """Gera dataset de interações com sazonalidade, categorias e gênero.

    Não usa o padrão Strategy (diferente do ``DatasetGenerator``) porque
    sazonalidade e categorias são parte intrínseca do mecanismo — não faz
    sentido trocá-las independentemente. Manter simples > manter consistente.

    Exemplo:
        >>> config = EnrichedGenerationConfig(
        ...     num_users=2000, num_items=500, num_interactions=50_000
        ... )
        >>> df = EnrichedDatasetGenerator().generate(config)
        >>> df.columns.tolist()
        ['user_id', 'item_id', 'category', 'user_gender', 'interaction_type', 'timestamp']
    """

    def generate(self, config: EnrichedGenerationConfig) -> pd.DataFrame:
        """Gera o DataFrame enriquecido a partir de ``config``.

        Args:
            config: Parâmetros de geração.

        Returns:
            DataFrame ordenado por timestamp com colunas:
            user_id, item_id, category, user_gender, interaction_type, timestamp.
        """
        rng = np.random.default_rng(config.seed)
        _logger.info(
            "generate_enriched_started | num_interactions=%d seed=%d",
            config.num_interactions,
            config.seed,
        )
        user_ids, item_ids = self._sample_pairs(config, rng)
        item_cat_map = self._build_item_category_map(config.num_items, rng)
        user_gender_map = self._build_user_gender_map(config.num_users, rng)
        timestamps, is_weekend = self._sample_seasonal_timestamps(config, rng)
        interaction_types = self._sample_interaction_types(
            config.num_interactions, is_weekend, rng
        )
        df = self._build_dataframe(
            user_ids, item_ids, item_cat_map, user_gender_map,
            timestamps, interaction_types,
        )
        return df.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _sample_pairs(
        config: EnrichedGenerationConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Amostra pares (user_id, item_id) com distribuição de Zipf truncada."""
        def _zipf(n_distinct: int, n_samples: int, skew: float) -> np.ndarray:
            return (rng.zipf(skew, size=n_samples) - 1) % n_distinct

        user_ids = _zipf(config.num_users, config.num_interactions, config.user_skew)
        item_ids = _zipf(config.num_items, config.num_interactions, config.item_skew)
        return user_ids, item_ids

    @staticmethod
    def _build_item_category_map(num_items: int, rng: np.random.Generator) -> np.ndarray:
        """Retorna array de índice de categoria por item_id (fixo por seed)."""
        return rng.integers(0, len(CATEGORIES), size=num_items)

    @staticmethod
    def _build_user_gender_map(num_users: int, rng: np.random.Generator) -> np.ndarray:
        """Retorna array de índice de gênero por user_id (fixo por seed)."""
        return rng.choice(len(GENDERS), size=num_users, p=list(_GENDER_PROBS))

    def _sample_seasonal_timestamps(
        self,
        config: EnrichedGenerationConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Timestamps com concentração maior nos fins de semana.

        Returns:
            Tupla ``(timestamps, is_weekend)`` onde ``is_weekend`` é bool array.
        """
        end_utc = datetime.now(tz=UTC)
        start = (end_utc - timedelta(days=config.time_window_days)).replace(tzinfo=None)
        day_weights = self._compute_day_weights(start, config.time_window_days)
        day_offsets = rng.choice(
            config.time_window_days, size=config.num_interactions, p=day_weights
        )
        second_offsets = rng.integers(0, _SECONDS_PER_DAY, size=config.num_interactions)
        start_np = np.datetime64(start, "s")
        timestamps = start_np + (day_offsets * _SECONDS_PER_DAY + second_offsets).astype(
            "timedelta64[s]"
        )
        is_weekend = ((start.weekday() + day_offsets) % 7) >= 5
        return timestamps, is_weekend

    @staticmethod
    def _compute_day_weights(start: datetime, window_days: int) -> np.ndarray:
        """Pesos normalizados por dia — fins de semana têm peso maior."""
        weights = np.array([
            _WEEKEND_TRAFFIC_WEIGHT if (start + timedelta(days=i)).weekday() >= 5 else 1.0
            for i in range(window_days)
        ])
        return weights / weights.sum()

    @staticmethod
    def _sample_interaction_types(
        n: int, is_weekend: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Funil diferenciado: mais purchase/add_to_cart nos fins de semana."""
        result = np.empty(n, dtype=object)
        n_weekend = int(is_weekend.sum())
        if n_weekend > 0:
            result[is_weekend] = rng.choice(
                _INTERACTION_VALUES, size=n_weekend, p=list(_WEEKEND_FUNNEL)
            )
        if n_weekend < n:
            result[~is_weekend] = rng.choice(
                _INTERACTION_VALUES, size=n - n_weekend, p=list(_WEEKDAY_FUNNEL)
            )
        return result

    @staticmethod
    def _build_dataframe(
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        item_cat_map: np.ndarray,
        user_gender_map: np.ndarray,
        timestamps: np.ndarray,
        interaction_types: np.ndarray,
    ) -> pd.DataFrame:
        """Monta o DataFrame final a partir dos arrays gerados."""
        return pd.DataFrame({
            "user_id": user_ids.astype(np.int32),
            "item_id": item_ids.astype(np.int32),
            "category": np.array(CATEGORIES)[item_cat_map[item_ids]],
            "user_gender": np.array(GENDERS)[user_gender_map[user_ids]],
            "interaction_type": interaction_types,
            "timestamp": timestamps,
        })
