"""
DEDA_LLM_Spatial_Hotelling
==========================
A Python toolkit for simulating 2-dimensional Hotelling models of spatial
competition where market actors can be LLM or Q-learning agents.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hotelling")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
