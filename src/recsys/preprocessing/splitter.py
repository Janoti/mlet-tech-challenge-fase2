"""Split temporal train/test sem vazamento (DataPipeline)."""

from __future__ import annotations

import pandas as pd

from recsys.data.schema import COLUMN_TIMESTAMP


def temporal_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide ``df`` por tempo: as interações mais recentes viram teste.

    Args:
        df: Interações com a coluna de timestamp.
        test_size: Fração (0-1) das interações mais recentes para teste.

    Returns:
        Tupla ``(train, test)`` reindexada; todo timestamp de teste é >= ao
        maior timestamp de treino (sem vazamento temporal).

    Raises:
        ValueError: Se ``test_size`` não estiver em (0, 1).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size deve estar em (0, 1)")
    ordered = df.sort_values(COLUMN_TIMESTAMP).reset_index(drop=True)
    cut = int(len(ordered) * (1.0 - test_size))
    train = ordered.iloc[:cut].reset_index(drop=True)
    test = ordered.iloc[cut:].reset_index(drop=True)
    return train, test
