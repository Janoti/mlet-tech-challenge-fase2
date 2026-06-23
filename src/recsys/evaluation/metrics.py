"""Métricas de ranking para recomendação (ModelEvaluator). Funções puras."""

from __future__ import annotations

import math


def _hits(recommended: list[int], relevant: set[int], k: int) -> list[int]:
    """Vetor binário de acertos nas top-k posições (1=relevante)."""
    return [1 if item in relevant else 0 for item in recommended[:k]]


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fração das top-k recomendações que são relevantes."""
    if k <= 0:
        return 0.0
    return sum(_hits(recommended, relevant, k)) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fração dos itens relevantes recuperados nas top-k posições."""
    if not relevant:
        return 0.0
    return sum(_hits(recommended, relevant, k)) / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """NDCG@k com ganho binário e desconto logarítmico."""
    if not relevant:
        return 0.0
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(_hits(recommended, relevant, k)))
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Average Precision@k (base do MAP)."""
    if not relevant:
        return 0.0
    score, hits = 0.0, 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)
