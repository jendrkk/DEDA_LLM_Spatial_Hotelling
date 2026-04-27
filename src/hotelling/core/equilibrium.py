"""Equilibrium solvers: Bertrand-Nash, joint monopoly, Tabuchi 2-D benchmark.

Responsibility: compute theoretical equilibrium benchmarks for the spatial
Hotelling model.

Public API: bertrand_nash, joint_monopoly, tabuchi_2d_benchmark

Key dependencies: numpy, scipy.optimize, hotelling.core.city

References:
    Calvano et al. (2020 AER);
    Tabuchi (1994) JUE;
    Bertrand (1883).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from hotelling.core.city import City


def bertrand_nash(
    city: City,
    transport_cost: float = 1.0,
    mu: float = 0.25,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Dict[str, float]:
    """Find Bertrand-Nash equilibrium prices by iterating best responses.

    Returns
    -------
    dict mapping firm_id -> equilibrium price
    """
    raise NotImplementedError


def joint_monopoly(
    city: City,
    transport_cost: float = 1.0,
    mu: float = 0.25,
) -> Dict[str, float]:
    """Find joint-monopoly (cartel) prices maximising total profit.

    Returns
    -------
    dict mapping firm_id -> monopoly price
    """
    raise NotImplementedError


def tabuchi_2d_benchmark(
    n: int = 2,
    t: float = 0.5,
    mu: float = 0.25,
) -> Tuple[float, float]:
    """Return (equilibrium_price, equilibrium_profit) for Tabuchi (1994) symmetric 2-D case.

    Returns
    -------
    (price, profit) tuple for the symmetric case
    """
    raise NotImplementedError
