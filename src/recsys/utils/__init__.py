"""Utilitários transversais ao projeto (seed, logging, etc.)."""

from recsys.utils.logging_utils import get_logger, log_kv, setup_logging
from recsys.utils.seed import set_global_seed

__all__ = ["get_logger", "log_kv", "set_global_seed", "setup_logging"]
