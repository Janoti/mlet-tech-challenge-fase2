"""Script CLI para gerar o dataset enriquecido (sazonalidade + categorias + gênero).

Uso:
    poetry run python scripts/generate_dataset_enriched.py
    NUM_INTERACTIONS=100000 poetry run python scripts/generate_dataset_enriched.py

Saída: data/raw/interactions_enriched.parquet

Mantido separado do ``generate_dataset.py`` intencionalmente — o grupo pode
comparar as duas bases e decidir qual usar antes de commitar a escolha.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from recsys.data.factory import DatasetGeneratorFactory  # noqa: E402
from recsys.data.generator_enriched import EnrichedGenerationConfig  # noqa: E402
from recsys.utils.logging_utils import get_logger, log_kv, setup_logging  # noqa: E402
from recsys.utils.seed import set_global_seed  # noqa: E402

_DEFAULTS: Final[dict[str, str]] = {
    "RANDOM_SEED": "42",
    "NUM_USERS": "2000",
    "NUM_ITEMS": "500",
    "NUM_INTERACTIONS": "50000",
    "DATA_RAW_DIR": "data/raw",
}

_OUTPUT_FILENAME: Final[str] = "interactions_enriched.parquet"


def _get_int_env(key: str) -> int:
    """Lê variável de ambiente como int, caindo em default se ausente."""
    return int(os.getenv(key, _DEFAULTS[key]))


def _build_config() -> EnrichedGenerationConfig:
    """Constrói ``EnrichedGenerationConfig`` a partir das variáveis de ambiente."""
    return EnrichedGenerationConfig(
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
    return raw_dir / _OUTPUT_FILENAME


def main() -> int:
    """Entrypoint do script. Retorna código de saída (0 em sucesso)."""
    setup_logging()
    logger = get_logger("recsys.scripts.generate_dataset_enriched")

    config = _build_config()
    set_global_seed(config.seed)

    log_kv(
        logger,
        "config_resolved",
        num_users=config.num_users,
        num_items=config.num_items,
        num_interactions=config.num_interactions,
        seed=config.seed,
    )

    generator = DatasetGeneratorFactory.create("enriched")
    df = generator.generate(config)

    output_path = _resolve_output_path()
    df.to_parquet(output_path, index=False)

    log_kv(logger, "dataset_written", rows=len(df), path=str(output_path))
    log_kv(
        logger,
        "interaction_distribution",
        **df["interaction_type"].value_counts().to_dict(),
    )
    log_kv(
        logger,
        "category_distribution",
        **df["category"].value_counts().to_dict(),
    )
    log_kv(
        logger,
        "gender_distribution",
        **df["user_gender"].value_counts().to_dict(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
