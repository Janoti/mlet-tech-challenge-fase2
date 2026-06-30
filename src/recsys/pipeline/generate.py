"""Stage ``generate``: cria o dataset bruto de interações (data/raw).

Console script ``recsys-generate``. Lê parâmetros de ``params.yaml`` (seção
``generate``), fixa a seed e usa o gerador enriquecido (com afinidade
usuário→categoria) para produzir dados com sinal de personalização.
"""

from __future__ import annotations

import sys
from pathlib import Path

from recsys.data.factory import DatasetGeneratorFactory
from recsys.data.generator_enriched import EnrichedGenerationConfig
from recsys.pipeline.params import load_params
from recsys.utils.logging_utils import get_logger, log_kv, setup_logging
from recsys.utils.seed import set_global_seed

_RAW_PATH = Path("data/raw/interactions.parquet")


def main() -> int:
    """Gera o dataset enriquecido e o grava em ``data/raw/interactions.parquet``."""
    setup_logging()
    logger = get_logger("recsys.pipeline.generate")
    cfg = load_params()["generate"]
    set_global_seed(cfg["seed"])
    config = EnrichedGenerationConfig(
        num_users=cfg["num_users"],
        num_items=cfg["num_items"],
        num_interactions=cfg["num_interactions"],
        seed=cfg["seed"],
        affinity_strength=cfg.get("affinity_strength", 3.0),
        n_pref_categories=cfg.get("n_pref_categories", 2),
    )
    df = DatasetGeneratorFactory.create("enriched").generate(config)
    _RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_RAW_PATH, index=False)
    log_kv(logger, "raw_generated", rows=len(df), path=str(_RAW_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
