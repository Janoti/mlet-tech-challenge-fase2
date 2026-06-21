"""Gerador de dataset sintético de interações user-item.

Este módulo é o **principal entregável da Etapa 1** sob o ponto de vista de
design: ele aplica o padrão **Strategy** (Gang of Four) para que possamos
trocar a forma como as interações são amostradas sem mexer no orquestrador.

Hierarquia:

    ┌────────────────────────────┐
    │  InteractionStrategy (ABC) │   <- contrato / interface
    └─────────────┬──────────────┘
                  │
        ┌─────────┴────────────┐
        │                      │
┌───────▼──────────┐  ┌────────▼──────────────┐
│ UniformStrategy  │  │ PopularityBiased...   │   <- estratégias concretas
└──────────────────┘  └───────────────────────┘
                  ▲
                  │ usado por composição
       ┌──────────┴─────────┐
       │  DatasetGenerator  │   <- contexto (orquestrador)
       └────────────────────┘

Por que Strategy aqui?
    - **OCP (Open/Closed Principle):** podemos adicionar uma nova estratégia
      (ex.: `TemporalDriftStrategy`, `ColdStartStrategy`) sem alterar o
      `DatasetGenerator`. Basta escrever uma nova subclasse.
    - **DIP (Dependency Inversion):** o gerador depende da abstração
      (`InteractionStrategy`), não de uma implementação concreta.
    - **Testabilidade:** podemos injetar uma estratégia "fake" determinística
      em testes para isolar a lógica de orquestração.

Referências:
    - Clean Code (Robert C. Martin), cap. 10 — Classes.
    - Design Patterns (GoF), Strategy pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from recsys.data.schema import (
    COLUMN_INTERACTION_TYPE,
    COLUMN_ITEM_ID,
    COLUMN_TIMESTAMP,
    COLUMN_USER_ID,
    INTERACTION_COLUMNS,
    InteractionType,
)
from recsys.utils.logging_utils import get_logger

_logger = get_logger(__name__)

# Funil de e-commerce: view (85%) → add_to_cart (12%) → purchase (3%).
_INTERACTION_TYPE_PROBS: tuple[float, float, float] = (0.85, 0.12, 0.03)


@dataclass(frozen=True)
class GenerationConfig:
    """Parâmetros de geração do dataset sintético.

    Usar um dataclass imutável (`frozen=True`) deixa explícito que esses
    parâmetros não devem ser alterados durante a execução — qualquer mudança
    cria uma nova instância. Isso ajuda a evitar bugs de estado compartilhado.

    Attributes:
        num_users: Quantidade de usuários distintos.
        num_items: Quantidade de itens distintos no catálogo.
        num_interactions: Total de interações a gerar (>= 10_000 por requisito).
        seed: Semente para reprodutibilidade.
        time_window_days: Janela temporal em que as interações são distribuídas.
    """

    num_users: int
    num_items: int
    num_interactions: int
    seed: int = 42
    time_window_days: int = 90

    def __post_init__(self) -> None:
        """Validação no construtor — falha cedo, antes de gerar nada."""
        if self.num_users <= 0 or self.num_items <= 0:
            raise ValueError("num_users e num_items devem ser positivos")
        if self.num_interactions < 10_000:
            raise ValueError("num_interactions deve ser >= 10.000 (requisito)")


# ============================================================================
# Strategy: interface
# ============================================================================
class InteractionStrategy(ABC):
    """Contrato para estratégias de amostragem de pares (user, item).

    Toda subclasse decide *como* sortear quem interage com o quê. Outras
    dimensões (tipo de interação, timestamp) são responsabilidade do
    `DatasetGenerator` para evitar acoplar a estratégia a detalhes do schema.
    """

    @abstractmethod
    def sample_pairs(
        self,
        config: GenerationConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Amostra pares (user_id, item_id).

        Args:
            config: Parâmetros da geração.
            rng: Gerador de números aleatórios já com seed fixada
                (injetado pelo orquestrador para garantir reprodutibilidade).

        Returns:
            Tupla `(user_ids, item_ids)`, cada um com shape
            `(config.num_interactions,)` e dtype inteiro.
        """


# ============================================================================
# Strategy concreta 1: uniforme (baseline)
# ============================================================================
class UniformInteractionStrategy(InteractionStrategy):
    """Cada usuário e cada item têm a mesma probabilidade de aparecer.

    Use como **baseline ingênua** — não reflete o comportamento real de um
    e-commerce, onde poucos itens concentram a maior parte do tráfego.
    Útil para comparar com a estratégia enviesada e quantificar o efeito.
    """

    def sample_pairs(
        self,
        config: GenerationConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sorteia uniformemente em [0, num_users) e [0, num_items)."""
        user_ids = rng.integers(0, config.num_users, size=config.num_interactions)
        item_ids = rng.integers(0, config.num_items, size=config.num_interactions)
        return user_ids, item_ids


# ============================================================================
# Strategy concreta 2: viés de popularidade (cauda longa)
# ============================================================================
class PopularityBiasedStrategy(InteractionStrategy):
    """Distribui interações segundo uma **lei de potência** (Zipf-like).

    Esta é a estratégia *default* — modela a realidade de e-commerces, onde:
        - Alguns poucos itens "bestsellers" concentram a maior parte das views.
        - A cauda longa contém milhares de itens com pouquíssimas interações.
        - Há um efeito similar em usuários: heavy users vs. visitantes
          esporádicos.

    Attributes:
        item_skew: Expoente da lei de potência para itens (maior => mais
            concentrado em poucos itens).
        user_skew: Expoente para usuários.
    """

    def __init__(self, item_skew: float = 1.2, user_skew: float = 1.1) -> None:
        # Validação inline — `_skew` precisa ser > 1 para que a distribuição
        # de Zipf seja bem definida pelo numpy.
        if item_skew <= 1.0 or user_skew <= 1.0:
            raise ValueError("item_skew e user_skew devem ser > 1.0")
        self._item_skew = item_skew
        self._user_skew = user_skew

    def sample_pairs(
        self,
        config: GenerationConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Amostra pares com viés de popularidade."""
        n = config.num_interactions
        user_ids = self._sample_with_zipf(config.num_users, n, self._user_skew, rng)
        item_ids = self._sample_with_zipf(config.num_items, n, self._item_skew, rng)
        return user_ids, item_ids

    @staticmethod
    def _sample_with_zipf(
        n_distinct: int,
        n_samples: int,
        skew: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Amostra IDs em [0, n_distinct) com distribuição de Zipf truncada."""
        # `rng.zipf` pode retornar valores muito grandes (cauda infinita).
        # Truncamos com módulo para mantê-los dentro do range válido — pequeno
        # viés extra, aceitável para um dataset sintético didático.
        raw = rng.zipf(skew, size=n_samples)
        return (raw - 1) % n_distinct


# ============================================================================
# Contexto / Orquestrador
# ============================================================================
class DatasetGenerator:
    """Orquestra a geração do dataset usando uma `InteractionStrategy`.

    Aplica composição em vez de herança (Clean Code, cap. 10): o gerador
    *tem-uma* estratégia, ele não *é-uma* estratégia. Isso permite trocar a
    estratégia em runtime e reutilizar o orquestrador.

    Exemplo:
        >>> config = GenerationConfig(num_users=100, num_items=50,
        ...                           num_interactions=10_000)
        >>> generator = DatasetGenerator(strategy=PopularityBiasedStrategy())
        >>> df = generator.generate(config)
        >>> df.columns.tolist()
        ['user_id', 'item_id', 'interaction_type', 'timestamp']
    """

    def __init__(self, strategy: InteractionStrategy) -> None:
        self._strategy = strategy

    def generate(self, config: GenerationConfig) -> pd.DataFrame:
        """Gera o DataFrame de interações conforme `config` e a estratégia.

        Args:
            config: Parâmetros de geração.

        Returns:
            DataFrame com as colunas de `INTERACTION_COLUMNS`, ordenado por
            timestamp crescente (ordem natural para séries temporais).
        """
        # Um único RNG é instanciado aqui e propagado — garante que duas
        # execuções com a mesma seed produzem exatamente o mesmo dataset.
        rng = np.random.default_rng(config.seed)

        _logger.info(
            "generate_started | strategy=%s num_interactions=%d seed=%d",
            type(self._strategy).__name__,
            config.num_interactions,
            config.seed,
        )

        user_ids, item_ids = self._strategy.sample_pairs(config, rng)
        interaction_types = self._sample_interaction_types(config.num_interactions, rng)
        timestamps = self._sample_timestamps(config.num_interactions, config.time_window_days, rng)

        df = pd.DataFrame(
            {
                COLUMN_USER_ID: user_ids.astype(np.int32),
                COLUMN_ITEM_ID: item_ids.astype(np.int32),
                COLUMN_INTERACTION_TYPE: interaction_types,
                COLUMN_TIMESTAMP: timestamps,
            },
            columns=INTERACTION_COLUMNS,
        )
        return df.sort_values(COLUMN_TIMESTAMP).reset_index(drop=True)

    @staticmethod
    def _sample_interaction_types(n: int, rng: np.random.Generator) -> np.ndarray:
        """Sorteia tipos de interação respeitando o funil view→cart→purchase."""
        # Lista pura: `rng.choice` aceita sequência de strings sem precisar
        # do `dtype=object` (que gera FutureWarning no pandas >= 2.2).
        choices = [t.value for t in InteractionType]
        return rng.choice(choices, size=n, p=list(_INTERACTION_TYPE_PROBS))

    @staticmethod
    def _sample_timestamps(n: int, window_days: int, rng: np.random.Generator) -> np.ndarray:
        """Sorteia timestamps uniformes na janela [now - window_days, now]."""
        # UTC explícito para evitar bugs com TZ local. Convertemos para naive
        # porque np.datetime64[s] não suporta tz-aware — todos os timestamps
        # no parquet são UTC por convenção.
        end_utc = datetime.now(tz=timezone.utc)
        start_naive = (end_utc - timedelta(days=window_days)).replace(tzinfo=None)
        total_seconds = window_days * 24 * 60 * 60
        offsets = rng.integers(0, total_seconds, size=n)
        # Vetorizado — ~50× mais rápido que list-comprehension para n grande.
        start = np.datetime64(start_naive, "s")
        return start + offsets.astype("timedelta64[s]")
