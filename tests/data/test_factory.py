"""Testes do DatasetGeneratorFactory.

Cobre:
    - Criação do gerador básico (modo "basic").
    - Criação do gerador enriquecido (modo "enriched").
    - Injeção de estratégia customizada no modo "basic".
    - Estratégia default do modo "basic".
    - Erro para modo desconhecido.
"""

from __future__ import annotations

import pytest

from recsys.data.factory import DatasetGeneratorFactory
from recsys.data.generator import (
    DatasetGenerator,
    PopularityBiasedStrategy,
    UniformInteractionStrategy,
)
from recsys.data.generator_enriched import EnrichedDatasetGenerator


class TestDatasetGeneratorFactory:
    """Contrato público do Factory — o que ele deve criar em cada modo."""

    def test_create_basic_returns_dataset_generator(self) -> None:
        gen = DatasetGeneratorFactory.create("basic")
        assert isinstance(gen, DatasetGenerator)

    def test_create_enriched_returns_enriched_generator(self) -> None:
        gen = DatasetGeneratorFactory.create("enriched")
        assert isinstance(gen, EnrichedDatasetGenerator)

    def test_create_basic_default_strategy_is_popularity_biased(self) -> None:
        gen = DatasetGeneratorFactory.create("basic")
        assert isinstance(gen, DatasetGenerator)
        assert isinstance(gen._strategy, PopularityBiasedStrategy)

    def test_create_basic_accepts_custom_strategy(self) -> None:
        strategy = UniformInteractionStrategy()
        gen = DatasetGeneratorFactory.create("basic", strategy=strategy)
        assert isinstance(gen, DatasetGenerator)
        assert isinstance(gen._strategy, UniformInteractionStrategy)

    def test_create_unknown_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Modo desconhecido"):
            DatasetGeneratorFactory.create("streaming")
