"""Testes do gerador de dataset sintético.

Cobre:
    - Contrato do schema (colunas, tipos, ordem).
    - Reprodutibilidade (mesma seed => mesmo DataFrame).
    - Strategy pattern (estratégias diferentes => resultados diferentes).
    - Validação de configuração (falha cedo com inputs inválidos).
"""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.data.generator import (
    DatasetGenerator,
    GenerationConfig,
    PopularityBiasedStrategy,
    UniformInteractionStrategy,
)
from recsys.data.schema import INTERACTION_COLUMNS_BASIC, InteractionType


# ----------------------------------------------------------------------------
# Fixtures — config pequena para que os testes rodem em ~ms.
# ----------------------------------------------------------------------------
@pytest.fixture()
def small_config() -> GenerationConfig:
    """Config mínima válida (10k é o mínimo do enunciado)."""
    return GenerationConfig(
        num_users=50, num_items=20, num_interactions=10_000, seed=123
    )


# ============================================================================
# Schema
# ============================================================================
class TestSchema:
    """O DataFrame retornado deve respeitar o contrato de schema."""

    def test_columns_match_schema(self, small_config: GenerationConfig) -> None:
        generator = DatasetGenerator(strategy=UniformInteractionStrategy())
        df = generator.generate(small_config)
        assert tuple(df.columns) == INTERACTION_COLUMNS_BASIC

    def test_row_count_matches_config(self, small_config: GenerationConfig) -> None:
        generator = DatasetGenerator(strategy=UniformInteractionStrategy())
        df = generator.generate(small_config)
        assert len(df) == small_config.num_interactions

    def test_user_and_item_ids_in_range(self, small_config: GenerationConfig) -> None:
        generator = DatasetGenerator(strategy=UniformInteractionStrategy())
        df = generator.generate(small_config)
        assert df["user_id"].between(0, small_config.num_users - 1).all()
        assert df["item_id"].between(0, small_config.num_items - 1).all()

    def test_interaction_types_are_valid(self, small_config: GenerationConfig) -> None:
        generator = DatasetGenerator(strategy=UniformInteractionStrategy())
        df = generator.generate(small_config)
        valid = {t.value for t in InteractionType}
        assert set(df["interaction_type"].unique()).issubset(valid)


# ============================================================================
# Reprodutibilidade — o requisito "seeds fixados" do enunciado.
# ============================================================================
class TestReproducibility:
    """Mesma seed deve produzir exatamente o mesmo DataFrame."""

    def test_same_seed_yields_identical_df(self, small_config: GenerationConfig) -> None:
        generator = DatasetGenerator(strategy=PopularityBiasedStrategy())
        df_first = generator.generate(small_config)
        df_second = generator.generate(small_config)
        pd.testing.assert_frame_equal(df_first, df_second)

    def test_different_seed_yields_different_df(self) -> None:
        cfg_a = GenerationConfig(num_users=50, num_items=20, num_interactions=10_000, seed=1)
        cfg_b = GenerationConfig(num_users=50, num_items=20, num_interactions=10_000, seed=2)
        generator = DatasetGenerator(strategy=PopularityBiasedStrategy())
        df_a = generator.generate(cfg_a)
        df_b = generator.generate(cfg_b)
        # Improvável (essencialmente p=0) que dois RNGs distintos produzam o
        # mesmo DataFrame — afirmar "diferentes" é seguro.
        assert not df_a.equals(df_b)


# ============================================================================
# Strategy pattern — duas estratégias produzem distribuições diferentes.
# ============================================================================
class TestStrategyPattern:
    """Estratégias distintas devem gerar distribuições distintas."""

    def test_popularity_bias_concentrates_more_than_uniform(
        self, small_config: GenerationConfig
    ) -> None:
        """No mundo enviesado, o item top deve aparecer mais que no uniforme."""
        uniform_top = self._top_item_frequency(UniformInteractionStrategy(), small_config)
        biased_top = self._top_item_frequency(PopularityBiasedStrategy(), small_config)
        # A estratégia de Zipf deve concentrar interações no item mais popular
        # com frequência superior à uniforme.
        assert biased_top > uniform_top

    @staticmethod
    def _top_item_frequency(strategy, config: GenerationConfig) -> int:
        """Auxilia: retorna a contagem do item mais frequente."""
        df = DatasetGenerator(strategy=strategy).generate(config)
        return int(df["item_id"].value_counts().iloc[0])


# ============================================================================
# Validação de inputs — falhar cedo.
# ============================================================================
class TestConfigValidation:
    """`GenerationConfig` deve rejeitar valores inválidos no construtor."""

    def test_negative_users_raises(self) -> None:
        with pytest.raises(ValueError, match="num_users e num_items"):
            GenerationConfig(num_users=-1, num_items=10, num_interactions=10_000)

    def test_too_few_interactions_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 10.000"):
            GenerationConfig(num_users=10, num_items=10, num_interactions=100)
