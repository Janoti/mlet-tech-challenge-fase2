"""Tipos e schemas do domínio "interações user-item".

Centralizar tipos do domínio em um módulo dedicado é uma aplicação do
princípio de **Bounded Context** (DDD) e do Clean Code (cap. 6 — "Objects and
Data Structures"): o resto do código depende de tipos com nomes do domínio
(`InteractionType.PURCHASE`) em vez de strings mágicas (`"purchase"`),
o que elimina toda uma classe de bugs por typo.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class InteractionType(StrEnum):
    """Tipos de interação possíveis entre um usuário e um item.

    Modela o funil clássico de e-commerce, do menos engajado ao mais engajado.
    Usar `StrEnum` (Python 3.11+) dá o melhor dos dois mundos: comparações com
    string funcionam (`x == "view"`), mas IDEs e linters reconhecem o tipo.
    """

    VIEW = "view"
    ADD_TO_CART = "add_to_cart"
    PURCHASE = "purchase"


# ----------------------------------------------------------------------------
# Constantes do schema (Final para deixar explícito que NÃO devem ser
# reatribuídas — substitui "magic numbers" e "magic strings" por nomes).
# ----------------------------------------------------------------------------

COLUMN_USER_ID: Final[str] = "user_id"
COLUMN_ITEM_ID: Final[str] = "item_id"
COLUMN_INTERACTION_TYPE: Final[str] = "interaction_type"
COLUMN_TIMESTAMP: Final[str] = "timestamp"

# Ordem canônica das colunas no DataFrame final.
# Manter como constante evita inconsistência entre quem escreve o parquet e
# quem o lê em etapas posteriores.
INTERACTION_COLUMNS: Final[tuple[str, ...]] = (
    COLUMN_USER_ID,
    COLUMN_ITEM_ID,
    COLUMN_INTERACTION_TYPE,
    COLUMN_TIMESTAMP,
)
