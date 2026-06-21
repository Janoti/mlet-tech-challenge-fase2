"""Factory para criação de geradores de dataset.

Centraliza a decisão de qual gerador instanciar com base em um modo string.
Scripts e módulos chamadores não precisam conhecer as classes concretas nem
suas dependências de construção.

Padrão aplicado: Factory Method (GoF — Criacional).
Por que aqui: o projeto tem dois geradores concretos hoje (basic, enriched)
e terá mais na Etapa 3+ (streaming, replay). Sem Factory, cada ponto de
criação acumula um ``if/elif`` independente que precisa ser atualizado em
vários lugares ao mesmo tempo.
"""

from __future__ import annotations

from recsys.data.generator import (
    DatasetGenerator,
    InteractionStrategy,
    PopularityBiasedStrategy,
)
from recsys.data.generator_enriched import EnrichedDatasetGenerator

_VALID_MODES: frozenset[str] = frozenset({"basic", "enriched"})


class DatasetGeneratorFactory:
    """Instancia geradores de dataset pelo nome do modo.

    Exemplo:
        >>> gen = DatasetGeneratorFactory.create("basic")
        >>> isinstance(gen, DatasetGenerator)
        True
        >>> gen = DatasetGeneratorFactory.create("enriched")
        >>> isinstance(gen, EnrichedDatasetGenerator)
        True
    """

    @staticmethod
    def create(
        mode: str,
        strategy: InteractionStrategy | None = None,
    ) -> DatasetGenerator | EnrichedDatasetGenerator:
        """Instancia o gerador correspondente ao modo solicitado.

        Args:
            mode: Tipo de gerador. Valores válidos: ``"basic"``, ``"enriched"``.
            strategy: Estratégia de amostragem para o modo ``"basic"``.
                Se ``None``, usa :class:`~recsys.data.generator.PopularityBiasedStrategy`.
                Ignorado para o modo ``"enriched"``.

        Returns:
            Instância pronta para uso do gerador correspondente.

        Raises:
            ValueError: Se ``mode`` não pertencer ao conjunto de modos válidos.
        """
        if mode == "basic":
            return DatasetGenerator(strategy=strategy or PopularityBiasedStrategy())
        if mode == "enriched":
            return EnrichedDatasetGenerator()
        raise ValueError(
            f"Modo desconhecido: {mode!r}. Modos válidos: {sorted(_VALID_MODES)}"
        )
