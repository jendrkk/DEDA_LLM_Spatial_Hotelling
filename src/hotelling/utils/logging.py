"""Structured JSON logging using loguru (optional) with stdlib fallback.

Responsibility: provide a unified logger factory for all hotelling modules.
If loguru is installed, uses structured JSON sinks; otherwise falls back to
stdlib logging with a simple text formatter.

Public API: get_logger, setup_logging

Key dependencies: logging (stdlib), loguru (optional)
"""
from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured for the hotelling package.

    Parameters
    ----------
    name : module name (e.g. __name__)

    Returns
    -------
    logging.Logger configured with a stderr handler if not already set up
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
    return logger


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure root logging for the hotelling package.

    Parameters
    ----------
    level : log level string (e.g. "DEBUG", "INFO", "WARNING")
    json_format : if True, attempt to use loguru JSON sink; falls back to
        stdlib text format if loguru is not installed
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger("hotelling")
    root.setLevel(numeric_level)

    if json_format:
        try:
            from loguru import logger as _loguru_logger  # noqa: PLC0415

            _loguru_logger.remove()
            _loguru_logger.add(sys.stderr, level=level.upper(), serialize=True)
            return
        except ImportError:
            pass

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        root.addHandler(handler)
