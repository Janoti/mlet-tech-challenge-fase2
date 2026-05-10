"""Testes do módulo `recsys.utils.logging_utils`."""

from __future__ import annotations

import logging

import pytest

from recsys.utils.logging_utils import get_logger, log_kv, setup_logging


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Limpa handlers do root logger entre testes — evita vazamento de estado.

    O `logging.basicConfig` é stateful e seus efeitos persistem entre testes.
    Sem este fixture, o segundo teste herda handlers/níveis do primeiro.
    """
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    root.handlers.clear()
    yield
    root.handlers.clear()
    root.handlers.extend(original_handlers)
    root.setLevel(original_level)


class TestSetupLogging:
    """`setup_logging` deve configurar o root logger com formato canônico."""

    def test_default_level_is_info(self) -> None:
        setup_logging()
        assert logging.getLogger().level == logging.INFO

    def test_custom_level_lowercase_is_normalized(self) -> None:
        setup_logging(level="debug")
        assert logging.getLogger().level == logging.DEBUG

    def test_invalid_level_falls_back_to_info(self) -> None:
        # Estratégia "falha silenciosa" deliberada: log nunca deve quebrar
        # a aplicação. Um typo em LOG_LEVEL não deve derrubar o script.
        setup_logging(level="NOSUCHLEVEL")
        assert logging.getLogger().level == logging.INFO

    def test_idempotent_does_not_duplicate_handlers(self) -> None:
        setup_logging()
        handler_count_after_first = len(logging.getLogger().handlers)
        setup_logging()
        setup_logging()
        # Chamadas repetidas não devem multiplicar handlers (cada um
        # produziria uma linha duplicada no terminal).
        assert len(logging.getLogger().handlers) == handler_count_after_first

    def test_env_var_is_used_when_no_arg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        setup_logging()
        assert logging.getLogger().level == logging.WARNING


class TestGetLogger:
    """`get_logger` é apenas um wrapper de `logging.getLogger`."""

    def test_returns_logger_with_correct_name(self) -> None:
        logger = get_logger("recsys.test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "recsys.test"

    def test_same_name_returns_same_instance(self) -> None:
        # Garantia da stdlib: `getLogger` retorna o mesmo objeto para o
        # mesmo nome — útil para que diferentes módulos compartilhem config.
        assert get_logger("a") is get_logger("a")


class TestLogKv:
    """`log_kv` formata key=value de forma determinística e ordenada."""

    def test_logs_event_alone_when_no_kwargs(self, caplog: pytest.LogCaptureFixture) -> None:
        setup_logging()
        logger = get_logger("recsys.test")
        with caplog.at_level(logging.INFO, logger="recsys.test"):
            log_kv(logger, "simple_event")
        assert "simple_event" in caplog.text

    def test_keys_are_sorted_alphabetically(self, caplog: pytest.LogCaptureFixture) -> None:
        # Ordem fixa torna o log parseável por ferramentas como grep/awk.
        setup_logging()
        logger = get_logger("recsys.test")
        with caplog.at_level(logging.INFO, logger="recsys.test"):
            log_kv(logger, "evt", zebra=1, alpha=2, mango=3)
        message = caplog.records[-1].getMessage()
        assert message.index("alpha=") < message.index("mango=") < message.index("zebra=")
