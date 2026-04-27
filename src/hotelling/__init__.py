"""hotelling: LLM-Driven 2-D Spatial Hotelling Competition Toolkit.

Public API re-exports for convenience.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hotelling")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
