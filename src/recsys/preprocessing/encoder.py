"""Mapeia IDs arbitrários para índices contíguos 0..n-1 (DataPipeline)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class IdEncoder:
    """Codifica IDs (user/item) em índices contíguos e inverte o mapeamento."""

    def __init__(self) -> None:
        """Inicializa o encoder vazio (use ``fit`` antes de transformar)."""
        self._to_idx: dict[int, int] = {}
        self._to_id: np.ndarray = np.array([], dtype=np.int64)

    def fit(self, ids: Iterable[int]) -> IdEncoder:
        """Aprende o mapeamento a partir dos IDs únicos (ordenados)."""
        uniques = np.unique(np.asarray(list(ids), dtype=np.int64))
        self._to_id = uniques
        self._to_idx = {int(v): i for i, v in enumerate(uniques)}
        return self

    def transform(self, ids: Iterable[int]) -> np.ndarray:
        """Converte IDs em índices contíguos."""
        return np.array([self._to_idx[int(v)] for v in ids], dtype=np.int64)

    def inverse_transform(self, indices: Iterable[int]) -> np.ndarray:
        """Converte índices de volta para os IDs originais."""
        return self._to_id[np.asarray(list(indices), dtype=np.int64)]

    @property
    def n_ids(self) -> int:
        """Número de IDs distintos aprendidos."""
        return len(self._to_id)
