"""Controle centralizado de reprodutibilidade.

Manter um único ponto de configuração de seeds é uma aplicação direta do
princípio DRY (Don't Repeat Yourself) e do SRP (Single Responsibility):
qualquer pipeline ou experimento que precisar de reprodutibilidade chama
`set_global_seed(n)` e pronto — não fica espalhando `random.seed(...)` /
`np.random.seed(...)` pelo código.

O enunciado do Tech Challenge exige explicitamente "seeds fixados".
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Fixa a seed de todas as fontes de aleatoriedade conhecidas.

    Cobre:
        - `random` (biblioteca padrão)
        - `numpy.random`
        - `PYTHONHASHSEED` (para reprodutibilidade de estruturas baseadas em
          hash, como `set` e `dict` em alguns cenários)

    A seed do PyTorch será adicionada na Etapa 4, quando o framework for
    introduzido — manter este módulo enxuto agora evita uma dependência
    pesada (torch) na Etapa 1.

    Args:
        seed: Valor inteiro não-negativo a ser usado como seed global.

    Raises:
        ValueError: Se `seed` for negativo.
    """
    if seed < 0:
        raise ValueError(f"seed deve ser >= 0, recebido: {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
