"""Leitura centralizada de ``params.yaml`` (fonte única de hiperparâmetros)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_params(path: str = "params.yaml") -> dict[str, Any]:
    """Lê ``params.yaml`` e retorna o dicionário de parâmetros por seção.

    Args:
        path: Caminho para o arquivo de parâmetros.

    Returns:
        Dicionário com as seções da pipeline (``generate``, ``preprocess``...).

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    content = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(content)
