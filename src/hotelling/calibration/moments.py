from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from hotelling.core.city import City
from hotelling.core.equilibrium import bertrand_nash
from hotelling.core.market import cell_choice_mass

_CHAIN_TYPES = ("discount", "standard", "bio")


def _firm_chain_types(city: City) -> np.ndarray:
    """Return (N,) array of chain-type labels read directly from
    firm.chain_type. Raises ValueError if any firm.chain_type is None
    (loader must populate it) or not in {'discount','standard','bio'}."""
    labels = np.empty(len(city.firms), dtype=object)
    for j, firm in enumerate(city.firms):
        ct = getattr(firm, "chain_type", None)
        if ct not in _CHAIN_TYPES:
            raise ValueError(
                f"Firm {firm.id} has invalid chain_type={ct!r}; the loader "
                "must populate chain_type. Re-run with the updated loader."
            )
        labels[j] = ct
    return labels


def _qualities(city: City) -> np.ndarray:
    return np.ascontiguousarray(
        [f.quality for f in city.firms], dtype=np.float64
    )


def _marginal_costs(city: City) -> np.ndarray:
    return np.ascontiguousarray(
        [f.marginal_cost for f in city.firms], dtype=np.float64
    )


def _resolve_nash(
    city: City,
    transport_cost: float,
    prices: np.ndarray | None,
    efforts: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if prices is None or efforts is None:
        prices, efforts = bertrand_nash(
            city, transport_cost, cache_path=None
        )
    return prices, efforts


def _choice_mass(
    city: City,
    transport_cost: float,
    prices: np.ndarray,
    efforts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return cell_choice_mass(
        prices=prices,
        efforts=efforts,
        dist2_km2=city.dist2_km2,
        cell_pop=city.cell_pop,
        lambda_phi=city.lambda_phi,
        pi_H=city.pi_H,
        pi_H_lambda_phi=city.pi_H_lambda_phi,
        alpha=city.alpha,
        quality=_qualities(city),
        beta=city.beta,
        transport_cost=transport_cost,
        mu=city.mu,
        a0=city.a0,
        transport_exponent=city.transport_exponent,
    )


def _mean_gross_margin_from(
    city: City,
    prices: np.ndarray,
    inside: np.ndarray,
) -> float:
    costs = _marginal_costs(city)
    demand = inside.sum(axis=0)
    if np.any(prices <= 0.0):
        raise ValueError("Nash prices must be positive for gross-margin computation")
    gross_margins = (prices - costs) / prices
    total_demand = demand.sum()
    if total_demand <= 0.0:
        raise ValueError("Total inside demand is zero; cannot compute mean gross margin")
    return float(np.dot(gross_margins, demand) / total_demand)


def _outside_share_from(city: City, outside: np.ndarray) -> float:
    consumer_mass = city.cell_pop + city.lambda_phi
    total_mass = consumer_mass.sum()
    if total_mass <= 0.0:
        raise ValueError("Total consumer mass is zero; cannot compute outside share")
    return float(outside.sum() / total_mass)


def _chain_shares_from(
    city: City,
    inside: np.ndarray,
) -> Dict[str, float]:
    demand = inside.sum(axis=0)
    total_inside = demand.sum()
    if total_inside <= 0.0:
        raise ValueError("Total inside demand is zero; cannot compute chain shares")

    chain_types = _firm_chain_types(city)
    shares: Dict[str, float] = {}
    for tau in _CHAIN_TYPES:
        mask = chain_types == tau
        shares[tau] = float(demand[mask].sum() / total_inside)
    return shares


def _bio_income_gradient_from(
    city: City,
    inside: np.ndarray,
    n_quantile_bins: int = 4,
) -> float:
    if n_quantile_bins < 4:
        raise ValueError("n_quantile_bins must be at least 4 for quartile bins")

    chain_types = _firm_chain_types(city)
    bio_cols = chain_types == "bio"
    if not np.any(bio_cols):
        raise ValueError("No bio stores found; cannot compute bio income gradient")

    inside_row_totals = inside.sum(axis=1)
    bio_share = inside[:, bio_cols].sum(axis=1) / np.maximum(inside_row_totals, 1e-12)

    q_low = np.quantile(city.pi_H, 1.0 / n_quantile_bins)
    q_high = np.quantile(city.pi_H, 1.0 - 1.0 / n_quantile_bins)
    bottom_mask = city.pi_H <= q_low
    top_mask = city.pi_H >= q_high

    if not np.any(bottom_mask):
        raise ValueError("No cells in bottom pi_H quartile")
    if not np.any(top_mask):
        raise ValueError("No cells in top pi_H quartile")

    consumer_mass = city.cell_pop + city.lambda_phi
    bottom_mass = consumer_mass[bottom_mask].sum()
    top_mass = consumer_mass[top_mask].sum()
    if bottom_mass <= 0.0:
        raise ValueError("Bottom pi_H quartile has zero consumer mass")
    if top_mass <= 0.0:
        raise ValueError("Top pi_H quartile has zero consumer mass")

    bottom_mean = float(
        np.dot(bio_share[bottom_mask], consumer_mass[bottom_mask]) / bottom_mass
    )
    top_mean = float(
        np.dot(bio_share[top_mask], consumer_mass[top_mask]) / top_mass
    )
    return top_mean / max(bottom_mean, 1e-9)


def mean_gross_margin(city: City, transport_cost: float) -> float:
    """Solve Bertrand-Nash, return the demand-weighted mean gross margin
    (p_j - c_j)/p_j across stores, weighted by Nash demand D_j.
    Demand D_j is obtained from cell_choice_mass(...).sum(axis=0) at the
    Nash prices. Costs from firm.marginal_cost. Guard p_j>0."""
    prices, efforts = _resolve_nash(city, transport_cost, None, None)
    inside, _outside = _choice_mass(city, transport_cost, prices, efforts)
    return _mean_gross_margin_from(city, prices, inside)


def outside_share(
    city: City,
    transport_cost: float,
    prices: np.ndarray | None = None,
    efforts: np.ndarray | None = None,
) -> float:
    """Return aggregate outside-option share:
    sum_i outside_mass[i] / sum_i (cell_pop[i] + lambda_phi[i]).
    If prices/efforts are None, solve Bertrand-Nash first. Uses
    cell_choice_mass for outside_mass."""
    prices, efforts = _resolve_nash(city, transport_cost, prices, efforts)
    _inside, outside = _choice_mass(city, transport_cost, prices, efforts)
    return _outside_share_from(city, outside)


def chain_shares(
    city: City,
    transport_cost: float,
    q_S: float,
    q_B: float,
) -> Dict[str, float]:
    """Return INSIDE market shares by chain type {'discount','standard','bio'}.
    Solve Bertrand-Nash, get D_j = inside_mass.sum(axis=0), group by chain
    type via _firm_chain_types, and divide each group's summed demand by the
    total inside demand sum_j D_j (so the three shares sum to 1)."""
    prices, efforts = _resolve_nash(city, transport_cost, None, None)
    inside, _outside = _choice_mass(city, transport_cost, prices, efforts)
    return _chain_shares_from(city, inside)


def bio_income_gradient(
    city: City,
    transport_cost: float,
    q_S: float,
    q_B: float,
    n_quantile_bins: int = 4,
) -> float:
    """Return the ratio of bio inside-share in HIGH-pi_H cells to bio
    inside-share in LOW-pi_H cells.
    Procedure:
      1. Solve Bertrand-Nash -> prices, efforts.
      2. inside, _ = cell_choice_mass(...). inside is (M, N).
      3. bio_cols = boolean mask of stores whose chain type == 'bio'.
      4. For each cell i: bio_share_i = inside[i, bio_cols].sum() /
         max(inside[i, :].sum(), 1e-12).
      5. Rank cells by city.pi_H. TOP bin = cells with pi_H in the top
         quartile; BOTTOM bin = bottom quartile (use np.quantile on pi_H).
      6. Weight bio_share_i by total cell consumer mass
         (cell_pop + lambda_phi) within each bin (mass-weighted mean).
      7. Return top_bin_bio_share / max(bottom_bin_bio_share, 1e-9).
    Guard against empty bins (raise ValueError with a clear message)."""
    prices, efforts = _resolve_nash(city, transport_cost, None, None)
    inside, _outside = _choice_mass(city, transport_cost, prices, efforts)
    return _bio_income_gradient_from(
        city, inside, n_quantile_bins=n_quantile_bins
    )


def all_model_moments(
    city: City,
    transport_cost: float,
    q_S: float,
    q_B: float,
) -> Dict[str, float]:
    """Convenience: solve Bertrand-Nash ONCE, reuse prices/efforts across all
    moment computations to avoid redundant solves, and return a dict with keys:
      'mean_gross_margin', 'outside_share',
      'chain_share_discount', 'chain_share_standard', 'chain_share_bio',
      'bio_income_gradient'."""
    prices, efforts = bertrand_nash(city, transport_cost, cache_path=None)
    inside, outside = _choice_mass(city, transport_cost, prices, efforts)

    shares = _chain_shares_from(city, inside)
    return {
        "mean_gross_margin": _mean_gross_margin_from(city, prices, inside),
        "outside_share": _outside_share_from(city, outside),
        "chain_share_discount": shares["discount"],
        "chain_share_standard": shares["standard"],
        "chain_share_bio": shares["bio"],
        "bio_income_gradient": _bio_income_gradient_from(city, inside),
    }
