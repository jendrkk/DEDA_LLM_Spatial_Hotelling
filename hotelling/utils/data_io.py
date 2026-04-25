"""
Data I/O utilities.

Thin wrappers around standard library I/O for consistent error handling and
logging across the package.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_json(path: str | Path) -> Any:
    """Load and return a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    object
        Parsed JSON content.
    """
    path = Path(path)
    with open(path) as fh:
        data = json.load(fh)
    logger.debug("Loaded JSON from %s", path)
    return data


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Serialise *data* to a JSON file.

    Parameters
    ----------
    data:
        JSON-serialisable object.
    path:
        Output file path.
    indent:
        Indentation level for pretty-printing.

    Returns
    -------
    Path
        The path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=indent)
    logger.debug("Saved JSON to %s", path)
    return path
