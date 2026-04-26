"""Reproducible random number generation.

Responsibility: provide factory functions for numpy.random.Generator instances
so that every model/agent has its own isolated RNG without touching the global
numpy random state.

Public API: make_rng, derive_seed

Key dependencies: numpy, hashlib

Convention: never call np.random.* directly; always use a Generator returned
by make_rng().
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a fresh numpy.random.Generator from an optional integer seed.

    Parameters
    ----------
    seed : integer seed or None (uses random entropy)

    Returns
    -------
    numpy.random.Generator instance
    """
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, *labels: str) -> int:
    """Deterministically derive a child seed from a base seed and string labels.

    Useful for generating reproducible per-agent seeds from a global seed.

    Parameters
    ----------
    base_seed : parent seed integer
    labels : string labels to mix in (e.g. agent_id, run_id)

    Returns
    -------
    int in [0, 2**31) - derived seed
    """
    key = str(base_seed) + "".join(labels)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)
