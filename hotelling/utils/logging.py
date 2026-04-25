"""
Logging configuration for the hotelling package.

Provides a single :func:`setup_logging` entry point that configures the root
logger for the package consistently.  Individual modules obtain a module-level
logger via :func:`get_logger`.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = logging.INFO,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DATE_FORMAT,
    filename: Optional[str] = None,
) -> None:
    """Configure root-level logging for the ``hotelling`` package.

    Parameters
    ----------
    level:
        Logging level (e.g. ``logging.DEBUG`` or ``"DEBUG"``).
    fmt:
        Log record format string.
    datefmt:
        Date/time format string.
    filename:
        If given, also write logs to this file.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if filename:
        handlers.append(logging.FileHandler(filename, mode="a"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("hotelling").setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for *name*, scoped under ``hotelling.*``.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    """
    if not name.startswith("hotelling"):
        name = f"hotelling.{name}"
    return logging.getLogger(name)
