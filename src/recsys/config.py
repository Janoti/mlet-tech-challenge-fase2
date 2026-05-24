"""Configuração centralizada via Pydantic Settings.

Lê variáveis do arquivo ``.env`` (ou do ambiente do sistema) e valida tipos
e valores na inicialização — princípio "falhar cedo" aplicado à configuração.

Uso:
    >>> from recsys.config import settings
    >>> settings.random_seed
    42
    >>> settings.mlflow_tracking_uri
    './mlruns'
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações do projeto lidas do ``.env``.

    Attributes:
        random_seed: Semente global para reprodutibilidade.
        num_users: Número de usuários no dataset sintético.
        num_items: Número de itens no catálogo sintético.
        num_interactions: Total de interações a gerar.
        data_raw_dir: Diretório de dados brutos.
        data_interim_dir: Diretório de dados intermediários.
        data_processed_dir: Diretório de dados processados.
        mlflow_tracking_uri: URI do servidor MLflow.
        mlflow_experiment_name: Nome do experimento no MLflow.
        log_level: Nível de log da aplicação.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Reprodutibilidade
    random_seed: int = Field(default=42, ge=0)

    # Dataset sintético
    num_users: int = Field(default=2000, gt=0)
    num_items: int = Field(default=500, gt=0)
    num_interactions: int = Field(default=50_000, ge=10_000)

    # Diretórios
    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_interim_dir: Path = Field(default=Path("data/interim"))
    data_processed_dir: Path = Field(default=Path("data/processed"))

    # MLflow
    mlflow_tracking_uri: str = Field(default="./mlruns")
    mlflow_experiment_name: str = Field(default="recsys-ecommerce")

    # Logging
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Garante que o nível de log é um dos valores válidos do Python."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level deve ser um de {valid}, recebido: {v!r}")
        return upper


# Instância singleton — importar diretamente nos módulos que precisarem.
settings = Settings()
