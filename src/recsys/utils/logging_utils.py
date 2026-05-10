"""Logging padronizado para todo o projeto.

Mesmo padrão adotado na Fase 1 do grupo (mlet-grupo4-tech-challenge): formato
único `timestamp | level | logger | mensagem`, configurável via env var
`LOG_LEVEL`. Centralizar essa configuração é uma aplicação direta do princípio
DRY do Clean Code — qualquer script ou módulo que precise logar chama
`setup_logging()` uma vez e depois `get_logger(__name__)` para obter o logger.

Por que não usar `print()`?
    - Logs vão para `stderr` por padrão, não para `stdout`. Útil para
      separar saída do programa (parquet path, métrica) de mensagens
      operacionais.
    - Logs têm **nível** (DEBUG/INFO/WARNING/ERROR) — controlamos verbosidade
      sem comentar/descomentar código.
    - Logs incluem **timestamp e nome do logger** automaticamente — essencial
      para depurar pipelines longos.
    - É a única forma compatível com observabilidade (CloudWatch, Loki, ELK)
      em produção (Etapa 3+).
"""

from __future__ import annotations

import logging
import os
from typing import Any

# Formato canônico do projeto. Mantê-lo igual ao da Fase 1 facilita reuso
# de ferramentas externas (greps, parsers de log).
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_DEFAULT_LEVEL = "INFO"


def setup_logging(level: str | None = None) -> None:
    """Configura o root logger com formato e nível padronizados.

    Idempotente: chamadas repetidas não duplicam handlers — apenas ajustam
    o nível. Isso é importante porque `pytest` e notebooks podem reimportar
    módulos várias vezes; sem essa proteção, cada log apareceria N vezes.

    Args:
        level: Nível de log (DEBUG/INFO/WARNING/ERROR/CRITICAL). Se `None`,
            usa a env var `LOG_LEVEL` ou cai em `INFO`.
    """
    resolved_level = (level or os.getenv("LOG_LEVEL", _DEFAULT_LEVEL)).upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        # Já configurado — só ajusta o nível.
        root.setLevel(numeric_level)
        return

    logging.basicConfig(level=numeric_level, format=_LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger nomeado.

    Convenção: usar `__name__` do módulo que está chamando, para que o nome
    do logger reflita a hierarquia de pacotes (`recsys.data.generator`).

    Args:
        name: Nome do logger (geralmente `__name__`).

    Returns:
        Instância de `logging.Logger` configurada com o formato canônico
        (desde que `setup_logging()` tenha sido chamado antes — caso
        contrário, herda o default do Python).
    """
    return logging.getLogger(name)


def log_kv(logger: logging.Logger, event: str, **kwargs: Any) -> None:  # noqa: ANN401
    """Loga um evento estruturado no formato `event | key1=v1 key2=v2`.

    Estilo "logs em key=value" facilita parsing por ferramentas como
    `awk`, Loki/Grafana e CloudWatch Insights.

    Args:
        logger: Logger a ser usado.
        event: Nome descritivo do evento (ex.: "dataset_generated").
        **kwargs: Pares chave=valor a anexar (tipos arbitrários — o `Any`
            aqui é deliberado: log de telemetria precisa aceitar qualquer
            valor serializável; ver `noqa: ANN401` na assinatura).
    """
    if not kwargs:
        logger.info(event)
        return
    fields = " ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    logger.info("%s | %s", event, fields)
