"""Script CLI para gerar o dataset sintético da Fase 2.

Uso:
    python scripts/generate_dataset.py
    NUM_INTERACTIONS=100000 python scripts/generate_dataset.py

Lê configuração das variáveis de ambiente (ver `.env.example`) — princípio
"externalize configuration" do 12-Factor App. Default seguro: gera 50k
interações em `data/raw/interactions.parquet` com seed=42.

Decisões de design:
    - Strategy default: `PopularityBiasedStrategy` (mais realista que uniforme).
    - Saída em Parquet (não CSV): tipagem preservada, ~10x menor, leitura
      muito mais rápida em pandas/pyarrow.
    - Funções pequenas (Clean Code): a função `main` orquestra e cada
      sub-função tem uma única responsabilidade.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Garante que o pacote `recsys` é importável quando rodamos como script
# (antes do `poetry install` da Etapa 2). Após instalado, esta linha vira
# inofensiva — Clean Code: "minimize precondições para execução".
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from recsys.data.generator import (  # noqa: E402  (import após sys.path)
    DatasetGenerator,
    GenerationConfig,
    PopularityBiasedStrategy,
)
from recsys.utils.seed import set_global_seed  # noqa: E402


# ----------------------------------------------------------------------------
# Defaults — coincidem com `.env.example`. Mantê-los aqui também garante que o
# script funciona mesmo sem `.env` configurado (boa experiência de onboarding).
# ----------------------------------------------------------------------------
_DEFAULTS = {
    "RANDOM_SEED": "42",
    "NUM_USERS": "2000",
    "NUM_ITEMS": "500",
    "NUM_INTERACTIONS": "50000",
    "DATA_RAW_DIR": "data/raw",
    "LOG_LEVEL": "INFO",
}


def _get_int_env(key: str) -> int:
    """Lê variável de ambiente como int, caindo em default se ausente."""
    return int(os.getenv(key, _DEFAULTS[key]))


def _build_logger() -> logging.Logger:
    """Configura logger simples para o script."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", _DEFAULTS["LOG_LEVEL"]),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    return logging.getLogger("generate_dataset")


def _build_config() -> GenerationConfig:
    """Constrói `GenerationConfig` a partir das variáveis de ambiente."""
    return GenerationConfig(
        num_users=_get_int_env("NUM_USERS"),
        num_items=_get_int_env("NUM_ITEMS"),
        num_interactions=_get_int_env("NUM_INTERACTIONS"),
        seed=_get_int_env("RANDOM_SEED"),
    )


def _resolve_output_path() -> Path:
    """Resolve o caminho absoluto do parquet de saída."""
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / os.getenv("DATA_RAW_DIR", _DEFAULTS["DATA_RAW_DIR"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / "interactions.parquet"


def main() -> int:
    """Entrypoint do script. Retorna código de saída (0 em sucesso)."""
    logger = _build_logger()
    config = _build_config()

    # Seed global garante reprodutibilidade total (numpy + random + hashes).
    set_global_seed(config.seed)
    logger.info("Configuração: %s", config)

    # Strategy default: viés de popularidade (cauda longa, realista).
    generator = DatasetGenerator(strategy=PopularityBiasedStrategy())
    df = generator.generate(config)

    output_path = _resolve_output_path()
    df.to_parquet(output_path, index=False)

    logger.info("Dataset gerado: %s linhas em %s", len(df), output_path)
    logger.info("Distribuição de tipos:\n%s", df["interaction_type"].value_counts())
    return 0


if __name__ == "__main__":
    sys.exit(main())
