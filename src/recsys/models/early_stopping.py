"""Política de early stopping para o treino da rede neural (Etapa 4).

Interrompe o treino quando a loss de validação para de melhorar, evitando
overfitting e épocas desnecessárias. Requisito de avaliação do Tech Challenge
("MLP funcional, early stopping").
"""

from __future__ import annotations

import math


class EarlyStopping:
    """Sinaliza quando parar o treino após ``patience`` épocas sem melhora.

    Attributes:
        best: Menor loss de validação observada até o momento.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        """Configura a paciência e o ganho mínimo considerado uma melhora.

        Args:
            patience: Épocas consecutivas sem melhora toleradas antes de parar.
            min_delta: Redução mínima na loss para contar como melhora.
        """
        self._patience = patience
        self._min_delta = min_delta
        self.best = math.inf
        self._epochs_without_improvement = 0

    def update(self, loss: float) -> bool:
        """Registra a loss da época e indica se o treino deve parar.

        Args:
            loss: Loss de validação da época atual.

        Returns:
            ``True`` se atingiu ``patience`` épocas sem melhora significativa.
        """
        if loss < self.best - self._min_delta:
            self.best = loss
            self._epochs_without_improvement = 0
            return False
        self._epochs_without_improvement += 1
        return self._epochs_without_improvement >= self._patience
