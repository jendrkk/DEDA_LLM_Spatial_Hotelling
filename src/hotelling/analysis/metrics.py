"""Post-simulation metrics: profit gain Δ, price gain, HHI, Gini, welfare proxy.

Responsibility: compute standard industrial-organisation metrics from simulation
output DataFrames.

Public API: profit_gain, price_gain, herfindahl_hirschman, gini, welfare_proxy

Key dependencies: numpy, pandas

References:
    Calvano et al. (2020 AER) §IV - profit gain Δ definition;
    Tirole (1988) Theory of Industrial Organization - HHI, welfare;
    Anderson, de Palma, Thisse (1992) - logit inclusive value / welfare proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def profit_gain(
    avg_profits: np.ndarray,
    nash_profits: np.ndarray,
    monopoly_profits: np.ndarray,
) -> float:
    """Compute Calvano profit gain Delta.

    Delta = (sum(avg_profits) - sum(nash_profits)) / (sum(monopoly_profits) - sum(nash_profits))

    Sum-aggregated over the supplied subset of firms, consistent with Calvano
    et al. (2020 AER) eq. (9): Δ = (π̄ − πᴺ) / (πᴹ − πᴺ) where each π is
    the per-firm average.  Sum and mean formulations are equivalent when
    applied to the same-length arrays; sum is preferred here so subgroup slices
    (per chain type) are handled identically to the global case.

    Clipped to [−0.5, 1.5] to suppress outliers on thin chain-type subsets
    with near-degenerate denominators.

    A value of 0 indicates competitive (Nash) outcome; 1 indicates joint
    monopoly.

    Parameters
    ----------
    avg_profits : shape (N,) average per-firm profits in the trained policy
    nash_profits : shape (N,) per-firm profits at Bertrand-Nash equilibrium
    monopoly_profits : shape (N,) per-firm profits under joint monopoly

    Returns
    -------
    float in [0, 1] (can exceed 1 for super-collusive outcomes); nan when
    denominator is near-zero.
    """
    denom = float(np.nansum(monopoly_profits) - np.nansum(nash_profits))
    if abs(denom) < 1e-9:
        return float("nan")
    numer = float(np.nansum(avg_profits) - np.nansum(nash_profits))
    return float(np.clip(numer / denom, -0.5, 1.5))


def price_gain(
    avg_prices: np.ndarray,
    nash_prices: np.ndarray,
    monopoly_prices: np.ndarray,
) -> float:
    """Compute analogous price gain metric.

    Delta_p = (mean(avg_prices) - mean(nash_prices)) / (mean(monopoly_prices) - mean(nash_prices))

    Mean-aggregated (not sum) for prices, consistent with the inline price-Δ
    computation in runner.py.  Clipped to [−0.5, 1.5].

    Parameters
    ----------
    avg_prices : shape (N,) average per-firm prices in trained policy
    nash_prices : shape (N,) Bertrand-Nash equilibrium prices
    monopoly_prices : shape (N,) joint-monopoly prices

    Returns
    -------
    float — price gain analogous to profit_gain; nan when denominator is
    near-zero.
    """
    denom = float(np.nanmean(monopoly_prices) - np.nanmean(nash_prices))
    if abs(denom) < 1e-9:
        return float("nan")
    numer = float(np.nanmean(avg_prices) - np.nanmean(nash_prices))
    return float(np.clip(numer / denom, -0.5, 1.5))


def herfindahl_hirschman(market_shares: np.ndarray) -> float:
    """Compute the Herfindahl-Hirschman Index.

    HHI = sum(s_i^2) * 10000, where s_i are market shares in [0, 1].

    Parameters
    ----------
    market_shares : shape (N,) array of market shares summing to <= 1

    Returns
    -------
    float in [0, 10000]
    """
    raise NotImplementedError


def gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient for profit or share inequality.

    Parameters
    ----------
    values : shape (N,) non-negative values (profits, shares, etc.)

    Returns
    -------
    float in [0, 1]; 0 = perfect equality, 1 = maximum inequality
    """
    raise NotImplementedError


def welfare_proxy(
    prices: np.ndarray,
    demands: np.ndarray,
    marginal_costs: np.ndarray,
    mu: float = 0.25,
) -> float:
    """Compute consumer surplus proxy via logit inclusive value.

    Based on the logit inclusive value W = mu * log(sum(exp(V_i / mu)))
    per Anderson, de Palma, Thisse (1992) §2.3.

    Parameters
    ----------
    prices : shape (N,) current prices
    demands : shape (N,) current market shares
    marginal_costs : shape (N,) marginal costs
    mu : logit scale parameter

    Returns
    -------
    float - welfare proxy (higher = better for consumers)
    """
    raise NotImplementedError
