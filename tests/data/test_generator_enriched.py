"""Testes do gerador de dataset enriquecido.

Cobre:
    - Contrato do schema (colunas, tipos, valores válidos).
    - Reprodutibilidade (mesma seed => mesmo DataFrame).
    - Ordenação por timestamp.
    - Validação de configuração (falha cedo com inputs inválidos).
"""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.data.generator_enriched import (
    CATEGORIES,
    GENDERS,
    EnrichedDatasetGenerator,
    EnrichedGenerationConfig,
)


# ----------------------------------------------------------------------------
# Fixtures — config pequena para que os testes rodem em ~ms.
# ----------------------------------------------------------------------------
@pytest.fixture()
def small_config() -> EnrichedGenerationConfig:
    """Config mínima válida para testes rápidos."""
    return EnrichedGenerationConfig(
        num_users=50, num_items=20, num_interactions=10_000, seed=123
    )


@pytest.fixture()
def small_df(small_config: EnrichedGenerationConfig) -> pd.DataFrame:
    """DataFrame gerado a partir da config mínima."""
    return EnrichedDatasetGenerator().generate(small_config)


# ============================================================================
# Schema — contrato de colunas e valores.
# ============================================================================
class TestSchema:
    """O DataFrame retornado deve respeitar o contrato de schema."""

    EXPECTED_COLUMNS = ("user_id", "item_id", "category", "user_gender", "interaction_type", "timestamp")

    def test_columns_match_schema(self, small_df: pd.DataFrame) -> None:
        assert tuple(small_df.columns) == self.EXPECTED_COLUMNS

    def test_row_count_matches_config(
        self, small_config: EnrichedGenerationConfig, small_df: pd.DataFrame
    ) -> None:
        assert len(small_df) == small_config.num_interactions

    def test_user_ids_in_range(
        self, small_config: EnrichedGenerationConfig, small_df: pd.DataFrame
    ) -> None:
        assert small_df["user_id"].between(0, small_config.num_users - 1).all()

    def test_item_ids_in_range(
        self, small_config: EnrichedGenerationConfig, small_df: pd.DataFrame
    ) -> None:
        assert small_df["item_id"].between(0, small_config.num_items - 1).all()

    def test_categories_are_valid(self, small_df: pd.DataFrame) -> None:
        assert set(small_df["category"].unique()).issubset(set(CATEGORIES))

    def test_genders_are_valid(self, small_df: pd.DataFrame) -> None:
        assert set(small_df["user_gender"].unique()).issubset(set(GENDERS))

    def test_interaction_types_are_valid(self, small_df: pd.DataFrame) -> None:
        valid = {"view", "add_to_cart", "purchase"}
        assert set(small_df["interaction_type"].unique()).issubset(valid)

    def test_sorted_by_timestamp(self, small_df: pd.DataFrame) -> None:
        assert small_df["timestamp"].is_monotonic_increasing


# ============================================================================
# Reprodutibilidade — o requisito "seeds fixados" do enunciado.
# ============================================================================
class TestReproducibility:
    """Mesma seed deve produzir exatamente o mesmo DataFrame."""

    def test_same_seed_yields_identical_df(
        self, small_config: EnrichedGenerationConfig
    ) -> None:
        generator = EnrichedDatasetGenerator()
        df_first = generator.generate(small_config)
        df_second = generator.generate(small_config)
        pd.testing.assert_frame_equal(df_first, df_second)

    def test_different_seed_yields_different_df(self) -> None:
        cfg_a = EnrichedGenerationConfig(num_users=50, num_items=20, num_interactions=10_000, seed=1)
        cfg_b = EnrichedGenerationConfig(num_users=50, num_items=20, num_interactions=10_000, seed=2)
        generator = EnrichedDatasetGenerator()
        df_a = generator.generate(cfg_a)
        df_b = generator.generate(cfg_b)
        assert not df_a.equals(df_b)


# ============================================================================
# Validação de inputs — falhar cedo.
# ============================================================================
class TestConfigValidation:
    """`EnrichedGenerationConfig` deve rejeitar valores inválidos no construtor."""

    def test_negative_users_raises(self) -> None:
        with pytest.raises(ValueError, match="positivos"):
            EnrichedGenerationConfig(num_users=-1, num_items=10, num_interactions=10_000)

    def test_zero_items_raises(self) -> None:
        with pytest.raises(ValueError, match="positivos"):
            EnrichedGenerationConfig(num_users=10, num_items=0, num_interactions=10_000)

    def test_too_few_interactions_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 10.000"):
            EnrichedGenerationConfig(num_users=10, num_items=10, num_interactions=9_999)

    def test_item_skew_at_boundary_raises(self) -> None:
        with pytest.raises(ValueError, match="item_skew"):
            EnrichedGenerationConfig(
                num_users=10, num_items=10, num_interactions=10_000, item_skew=1.0
            )

    def test_user_skew_at_boundary_raises(self) -> None:
        with pytest.raises(ValueError, match="user_skew"):
            EnrichedGenerationConfig(
                num_users=10, num_items=10, num_interactions=10_000, user_skew=1.0
            )
