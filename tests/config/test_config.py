"""Testes do módulo de configuração (Pydantic Settings).

Cobre:
    - Valores default carregados corretamente.
    - Normalização do log_level para maiúsculas.
    - Rejeição de valores inválidos (falha cedo).
    - Leitura de variáveis de ambiente sobrepondo os defaults.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from recsys.config import Settings


# ============================================================================
# Defaults
# ============================================================================
class TestDefaults:
    """Sem .env, os defaults do Settings devem ser carregados."""

    def test_random_seed_default(self) -> None:
        s = Settings()
        assert s.random_seed == 42

    def test_num_users_default(self) -> None:
        s = Settings()
        assert s.num_users == 2000

    def test_num_items_default(self) -> None:
        s = Settings()
        assert s.num_items == 500

    def test_num_interactions_default(self) -> None:
        s = Settings()
        assert s.num_interactions == 50_000

    def test_mlflow_tracking_uri_default(self) -> None:
        s = Settings()
        assert s.mlflow_tracking_uri == "./mlruns"

    def test_log_level_default(self) -> None:
        s = Settings()
        assert s.log_level == "INFO"


# ============================================================================
# Normalização de log_level
# ============================================================================
class TestLogLevelNormalization:
    """log_level deve ser normalizado para maiúsculas."""

    def test_lowercase_is_normalized(self) -> None:
        s = Settings(log_level="info")
        assert s.log_level == "INFO"

    def test_mixed_case_is_normalized(self) -> None:
        s = Settings(log_level="Warning")
        assert s.log_level == "WARNING"

    def test_all_valid_levels_accepted(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            s = Settings(log_level=level)
            assert s.log_level == level


# ============================================================================
# Validação — falhar cedo com inputs inválidos.
# ============================================================================
class TestValidation:
    """Settings deve rejeitar valores inválidos no construtor."""

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(ValidationError, match="log_level"):
            Settings(log_level="VERBOSE")

    def test_negative_random_seed_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(random_seed=-1)

    def test_zero_num_users_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(num_users=0)

    def test_zero_num_items_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(num_items=0)

    def test_num_interactions_below_minimum_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(num_interactions=9_999)


# ============================================================================
# Sobrescrita via variáveis de ambiente
# ============================================================================
class TestEnvOverride:
    """Variáveis de ambiente devem sobrescrever os defaults."""

    def test_env_var_overrides_random_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANDOM_SEED", "99")
        s = Settings()
        assert s.random_seed == 99

    def test_env_var_overrides_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        s = Settings()
        assert s.log_level == "DEBUG"

    def test_env_var_overrides_mlflow_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
        s = Settings()
        assert s.mlflow_tracking_uri == "http://mlflow-server:5000"
