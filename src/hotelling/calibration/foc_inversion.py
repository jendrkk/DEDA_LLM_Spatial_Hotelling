"""FOC-inversion calibration (ADR-032).

Alternative to the structural method-of-moments solver in
`src/hotelling/calibration/structural.py`. Uses the empirical bio-to-discount
inside-market-share ratio (and optionally the standard-to-discount ratio) plus
the existing data-side inputs (transport cost t, chain-specific marginal costs
c_τ, prices p_τ from the price ladder, fixed alpha_ratio) to identify
(mu, q_S, q_B) in closed form via the Nash-FOC inversion identity and the
aggregate logit log-share-ratio identity.

This module is invoked only when `scripts/calibrate_structural.py` is called
with `--method foc_inversion`. It does NOT replace `calibrate_structural`;
both methods can be run from the same CLI and write to compatible env-YAML
outputs.

Mathematical summary
--------------------
Inputs (after data-side calibration):
    s_outside     external target, e.g. 0.04
    s_B / s_D     external ratio (required; the one input that closes the
                  identification — see ADR-032 §Identification).
    s_S / s_D     optional ratio; defaults to N_S / N_D from store counts.
    alpha_ratio   = alpha_H / alpha_L, fixed exogenously (default 2.5).
    t             from compute_transport_cost
    c_τ           from compute_marginal_costs (chain-specific, ADR-025)
    p_τ           basket_price_standard_eur × price_index[τ]

Step A — absolute shares:
    s_D = (1 − s_outside) / (1 + s_S/s_D + s_B/s_D)
    s_S = s_S/s_D × s_D
    s_B = s_B/s_D × s_D

Step B — μ from Nash-FOC inversion:
    μ_τ_implied = (p_τ − c_τ)(1 − s_τ)         for τ ∈ {D, S, B}
    μ_hat = share-weighted average across τ
    diagnostic: max(μ_τ) − min(μ_τ) (the "FOC spread")

Step C — population-weighted spatial accessibility per type:
    A_{τ,i} = Σ_{j: θ_j=τ} exp(−t · d_ij / μ_hat)
    A_τ    = Σ_i ω_i · A_{τ,i} / Σ_i ω_i             (mass-weighted scalar)
    where ω_i = cell_pop_i + lambda_phi_i is total cell consumer mass.

Step D — closed-form q recovery (with the existing α-normalisation that
sets the population-weighted mean of α to 1, i.e. ᾱ = 1):
    q_S = μ_hat · [ln(s_S/s_D) − ln(A_S/A_D)] + (p_S − p_D)
    q_B = μ_hat · [ln(s_B/s_D) − ln(A_B/A_D)] + (p_B − p_D)

Step E (optional, gated by `refine_a0`) — refine a_0 by 1D root-find so the
equilibrium outside share matches s_outside. Uses scipy.optimize.brentq.
If `refine_a0=False`, a_0 is held at the value in env_cfg.outside_option.

References
----------
ADR-024 transport cost from VTT.
ADR-025 marginal cost from gross margin (use chain-specific margins).
ADR-026 the existing 5-moment MoM design (now reduced to 2-param in
    scripts/calibrate_structural.py).
ADR-032 (added in Step 2) FOC-inversion calibration design.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "compute_absolute_shares",
    "compute_mu_from_foc",
    "compute_accessibility_by_type",
    "compute_q_closed_form",
    "calibrate_foc_inversion",
    "load_chain_types_from_parquet",
]

_CHAIN_TYPES = ("discount", "standard", "bio")
_STORE_COUNT_DEFAULTS = {"discount": 196, "standard": 207, "bio": 91}


# ---------------------------------------------------------------------------
# Public step functions
# ---------------------------------------------------------------------------

def compute_absolute_shares(
    s_outside: float,
    s_B_over_s_D: float,
    s_S_over_s_D: Optional[float] = None,
    *,
    store_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Resolve absolute inside-market shares from two ratios and the outside
    share. If s_S_over_s_D is None, defaults to the store-count ratio
    N_S / N_D (which is a defensible proxy for standard-vs-discount because
    their spatial distributions are similar; see ADR-032 §3).

    Parameters
    ----------
    s_outside : float in (0, 1)
        Target outside-option share, e.g. 0.04.
    s_B_over_s_D : float > 0
        Bio-to-discount inside-market-share ratio (the one external input
        that fixes the identification problem in Assumption 1).
    s_S_over_s_D : Optional[float]
        Standard-to-discount inside-market-share ratio. If None, uses
        store-count proxy.
    store_counts : Optional[Dict[str, int]]
        Override for the default {'discount': 196, 'standard': 207, 'bio': 91}
        store counts used to derive s_S_over_s_D when not provided. Pass the
        actual current counts from supermarkets.parquet if they have changed.

    Returns
    -------
    Dict[str, float]
        Keys 'discount', 'standard', 'bio'. Values sum to (1 − s_outside).

    Raises
    ------
    ValueError
        If s_outside ∉ (0, 1), ratios non-positive, or counts non-positive.
    """
    if not (0.0 < s_outside < 1.0):
        raise ValueError(f"s_outside must be in (0, 1); got {s_outside}")
    if s_B_over_s_D <= 0.0:
        raise ValueError(f"s_B_over_s_D must be > 0; got {s_B_over_s_D}")

    if s_S_over_s_D is None:
        counts = dict(_STORE_COUNT_DEFAULTS)
        if store_counts is not None:
            counts.update(store_counts)
        if counts["discount"] <= 0 or counts["standard"] <= 0:
            raise ValueError(f"Invalid store_counts: {counts}")
        s_S_over_s_D = counts["standard"] / counts["discount"]
        logger.info(
            "s_S_over_s_D unspecified; using store-count proxy "
            "(N_S=%d, N_D=%d) -> %.4f",
            counts["standard"], counts["discount"], s_S_over_s_D,
        )
    elif s_S_over_s_D <= 0.0:
        raise ValueError(f"s_S_over_s_D must be > 0; got {s_S_over_s_D}")

    inside = 1.0 - s_outside
    denom = 1.0 + s_S_over_s_D + s_B_over_s_D
    s_D = inside / denom
    s_S = s_S_over_s_D * s_D
    s_B = s_B_over_s_D * s_D
    return {"discount": s_D, "standard": s_S, "bio": s_B}


def compute_mu_from_foc(
    prices: Dict[str, float],
    costs: Dict[str, float],
    shares: Dict[str, float],
) -> Dict[str, float]:
    """Invert the per-chain-type Nash-FOC for μ.

    For each τ ∈ {discount, standard, bio}:
        μ_τ_implied = (p_τ − c_τ) · (1 − s_τ)

    In a degenerate single-product-per-type logit Bertrand-Nash these would
    be exactly equal; with multiple firms per type they differ. The
    share-weighted aggregate is the calibrated estimate; the spread is the
    primary diagnostic for whether the share inputs are internally
    consistent with the FOC.

    Parameters
    ----------
    prices, costs, shares : Dict[str, float]
        Each with keys 'discount', 'standard', 'bio'.

    Returns
    -------
    Dict with keys:
        'mu_by_type'         : Dict[str, float] — three implied μ values
        'mu_share_weighted'  : float            — Σ_τ s_τ · μ_τ / Σ_τ s_τ
        'mu_simple_mean'     : float            — arithmetic mean of three
        'mu_max'             : float            — max μ_τ
        'mu_min'             : float            — min μ_τ
        'spread_absolute'    : float            — max − min
        'spread_relative'    : float            — (max − min) / mean

    Raises
    ------
    ValueError
        If any (p_τ − c_τ) ≤ 0 or any s_τ ≥ 1 (degenerate market) or any
        s_τ ≤ 0.
    """
    mu_by_type: Dict[str, float] = {}
    for tau in _CHAIN_TYPES:
        margin = prices[tau] - costs[tau]
        if margin <= 0.0:
            raise ValueError(
                f"Non-positive markup for {tau}: p={prices[tau]:.4f}, "
                f"c={costs[tau]:.4f}. Check ADR-025 cost calibration."
            )
        if shares[tau] >= 1.0 or shares[tau] <= 0.0:
            raise ValueError(
                f"Share for {tau} out of (0, 1): s={shares[tau]:.6f}"
            )
        mu_by_type[tau] = margin * (1.0 - shares[tau])

    mu_values = np.array([mu_by_type[t] for t in _CHAIN_TYPES])
    share_values = np.array([shares[t] for t in _CHAIN_TYPES])
    total_share = float(share_values.sum())
    if total_share <= 0.0:
        raise ValueError("Total inside share is non-positive")
    mu_sw = float(np.dot(share_values, mu_values) / total_share)
    mu_mean = float(mu_values.mean())
    mu_max = float(mu_values.max())
    mu_min = float(mu_values.min())
    spread_abs = mu_max - mu_min
    spread_rel = spread_abs / mu_mean if mu_mean > 0.0 else float("nan")

    return {
        "mu_by_type": mu_by_type,
        "mu_share_weighted": mu_sw,
        "mu_simple_mean": mu_mean,
        "mu_max": mu_max,
        "mu_min": mu_min,
        "spread_absolute": float(spread_abs),
        "spread_relative": float(spread_rel),
    }


def load_chain_types_from_parquet(stores_path: str) -> np.ndarray:
    """Read chain_type column from supermarkets.parquet and return as
    (N,) array of lowercased strings in canonical store order (the file's
    own row order, which matches city.firms[j]).

    This is more robust than `_firm_chain_types(city, q_S, q_B)` for the
    FOC-inversion path because that helper requires q_S, q_B as inputs,
    which are precisely the unknowns we are solving for.

    Parameters
    ----------
    stores_path : str
        Absolute path to data/processed/supermarkets.parquet.

    Returns
    -------
    np.ndarray of shape (N,), dtype object, entries lowercased and
    stripped.

    Raises
    ------
    KeyError
        If 'chain_type' column is absent.
    ValueError
        If any chain_type is outside {'discount', 'standard', 'bio'}.
    """
    df = pd.read_parquet(stores_path, columns=["chain_type"])
    if "chain_type" not in df.columns:
        raise KeyError(
            f"'chain_type' column not found in {stores_path}. "
            "Re-run the GEO pipeline (process_supermarkets) to populate it."
        )
    ct = df["chain_type"].astype(str).str.strip().str.lower().values
    unknown = set(ct) - set(_CHAIN_TYPES)
    if unknown:
        raise ValueError(
            f"Unknown chain_type values in {stores_path}: {unknown}. "
            f"Expected subset of {_CHAIN_TYPES}."
        )
    return np.asarray(ct, dtype=object)


def compute_accessibility_by_type(
    dist_minutes: np.ndarray,
    chain_types: np.ndarray,
    cell_mass: np.ndarray,
    transport_cost: float,
    mu: float,
) -> Dict[str, float]:
    """Population-weighted spatial accessibility per chain type.

    For each cell i and chain type τ:
        A_{τ,i} = Σ_{j: chain_type_j = τ} exp(−transport_cost · d_ij / μ)

    Aggregated:
        A_τ = Σ_i ω_i · A_{τ,i} / Σ_i ω_i        (mass-weighted scalar)

    where d_ij is travel-time minutes (city.dist2_km2 holds minutes per
    ADR-020) and ω_i = cell_mass[i].

    Parameters
    ----------
    dist_minutes : np.ndarray, shape (M, N), float
        Travel time from cell i to store j, in minutes.
    chain_types : np.ndarray, shape (N,), dtype object
        Per-store chain-type labels in {'discount','standard','bio'}.
    cell_mass : np.ndarray, shape (M,), float
        Total consumer mass per cell, ω_i = cell_pop_i + lambda_phi_i.
    transport_cost : float
        EUR per one-way minute (after round-trip factor; ADR-024).
    mu : float
        Logit scale (the calibrated value from compute_mu_from_foc).

    Returns
    -------
    Dict[str, float] with keys 'discount', 'standard', 'bio'. Each value
    is the mass-weighted aggregate accessibility A_τ.

    Notes
    -----
    Computed with a per-cell log-sum-exp for numerical stability:
        log(A_{τ,i}) = logsumexp_{j: θ_j=τ}( −t · d_ij / μ )
    Then A_τ = mean_i(exp(logA_{τ,i}); weights ω_i).
    """
    if dist_minutes.ndim != 2:
        raise ValueError(f"dist_minutes must be 2-D; got {dist_minutes.shape}")
    M, N = dist_minutes.shape
    if chain_types.shape != (N,):
        raise ValueError(
            f"chain_types shape {chain_types.shape} != (N,)=({N},)"
        )
    if cell_mass.shape != (M,):
        raise ValueError(
            f"cell_mass shape {cell_mass.shape} != (M,)=({M},)"
        )
    if mu <= 0.0:
        raise ValueError(f"mu must be > 0; got {mu}")

    total_mass = float(cell_mass.sum())
    if total_mass <= 0.0:
        raise ValueError("Total cell mass is non-positive")

    # Utility component −t · d / μ
    util = -(transport_cost / mu) * dist_minutes  # (M, N)

    out: Dict[str, float] = {}
    for tau in _CHAIN_TYPES:
        mask = (chain_types == tau)
        n_tau = int(mask.sum())
        if n_tau == 0:
            raise ValueError(f"No stores of chain type {tau!r}")
        # logsumexp per cell over the subset of stores of this type
        sub = util[:, mask]                     # (M, n_tau)
        m_row = sub.max(axis=1, keepdims=True)  # (M, 1)
        # safe in case all -inf (won't happen with finite dists, but guard)
        log_a = m_row[:, 0] + np.log(
            np.exp(sub - m_row).sum(axis=1)
        )                                       # (M,)
        a_cell = np.exp(log_a)                  # (M,)
        out[tau] = float(np.dot(a_cell, cell_mass) / total_mass)
    return out


def compute_q_closed_form(
    mu: float,
    shares: Dict[str, float],
    accessibility: Dict[str, float],
    prices: Dict[str, float],
) -> Dict[str, float]:
    """Closed-form recovery of (q_S, q_B) from aggregate log-share ratios.

    Under the existing alpha-normalisation `pi_L_bar·alpha_L +
    pi_H_bar·alpha_H = 1`, the population-weighted mean ᾱ = 1, so:

        q_τ = μ · [ln(s_τ / s_D) − ln(A_τ / A_D)] + (p_τ − p_D)

    with the normalisation q_D = 0.

    Parameters
    ----------
    mu : float
        Calibrated logit scale.
    shares : Dict[str, float]
        Absolute inside shares.
    accessibility : Dict[str, float]
        Mass-weighted aggregate accessibility per chain type.
    prices : Dict[str, float]
        Basket prices per chain type.

    Returns
    -------
    Dict with keys 'q_D' (=0.0), 'q_S', 'q_B'.

    Raises
    ------
    ValueError
        If any share or accessibility is non-positive or if q_B ≤ q_S
        (violates vertical ordering; indicates inconsistent inputs).
    """
    for d, name in ((shares, "shares"), (accessibility, "accessibility")):
        for tau in _CHAIN_TYPES:
            if d[tau] <= 0.0:
                raise ValueError(
                    f"{name}[{tau!r}] = {d[tau]:.6e} is non-positive; "
                    "cannot take logarithm."
                )

    q_D = 0.0
    q_S = (
        mu
        * (np.log(shares["standard"] / shares["discount"])
           - np.log(accessibility["standard"] / accessibility["discount"]))
        + (prices["standard"] - prices["discount"])
    )
    q_B = (
        mu
        * (np.log(shares["bio"] / shares["discount"])
           - np.log(accessibility["bio"] / accessibility["discount"]))
        + (prices["bio"] - prices["discount"])
    )

    if q_B <= q_S:
        raise ValueError(
            f"Vertical ordering violated: q_B={q_B:.4f} ≤ q_S={q_S:.4f}. "
            "Inputs are inconsistent with q_D < q_S < q_B."
        )
    if q_S <= 0.0:
        raise ValueError(
            f"q_S={q_S:.4f} ≤ 0; standard chain less attractive than "
            "discount even after pricing/accessibility adjustment."
        )
    return {"q_D": q_D, "q_S": float(q_S), "q_B": float(q_B)}


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def calibrate_foc_inversion(
    targets: dict,
    env_cfg: dict,
    grid_path: str,
    stores_path: str,
    travel_times_path: str,
    lambda_val: float,
    refine_a0: bool = True,
    force_chain_specific_costs: bool = True,
) -> dict:
    """Top-level FOC-inversion calibration.

    Returns a dict with the same key set as `calibrate_structural` plus
    additional diagnostic keys, so the env-YAML writer in
    `scripts/calibrate_structural.py` works unchanged.

    Pipeline:
      1. Data-side: t from compute_transport_cost; c_τ from
         compute_marginal_costs (chain-specific if
         force_chain_specific_costs=True OR targets['use_common_margin']
         is False).
      2. Read foc_inversion block from targets.yaml.
      3. compute_absolute_shares(...) -> s_τ.
      4. compute_mu_from_foc(...)     -> μ_hat plus diagnostics.
      5. Load chain_types from supermarkets.parquet; build a thin city
         (use load_berlin_city with the resolved μ and tentative q for
         accessibility — accessibility only needs t, dist, μ, so the q
         passed to load_berlin_city is irrelevant for this step but must
         be syntactically valid → use placeholders q_S=1.0, q_B=2.0).
         Actually: read dist matrix and cell_mass without using
         load_berlin_city to avoid the circularity. See implementation.
      6. compute_accessibility_by_type(...) -> A_τ.
      7. compute_q_closed_form(...)         -> q_S, q_B.
      8. alpha_L, alpha_H from alpha_ratio (existing
         _alphas_from_ratio).
      9. If refine_a0=True: build the calibration city with the new
         (μ, q_S, q_B, α_L, α_H), find a_0 by brentq so that the
         simulated outside share matches targets['outside_share_target'].
         Else: keep env_cfg['outside_option'].
     10. Rebuild city with the final parameters and compute all_model_moments
         for the report.

    Return dict matches the existing structural.calibrate_structural
    return schema (same keys), plus:
        'method'              : 'foc_inversion'
        'foc_diagnostics'     : Dict from compute_mu_from_foc
        'shares_resolved'     : Dict from compute_absolute_shares
        'accessibility'       : Dict from compute_accessibility_by_type
        's_S_over_s_D_used'   : float
        's_B_over_s_D_used'   : float
        'a0_refined'          : bool
    """
    # Lazy imports to avoid circularity at module load time.
    from hotelling.calibration.structural import (
        compute_transport_cost,
        compute_marginal_costs,
        _alphas_from_ratio,
        _pi_H_bar,
        _build_calibration_city,
    )
    from hotelling.calibration.moments import all_model_moments

    # 1) Data-side calibration -----------------------------------------------
    t = compute_transport_cost(
        wage_monthly_gross_eur=float(targets["wage_monthly_gross_eur"]),
        work_hours_per_month=float(targets["work_hours_per_month"]),
        vtt_wage_ratio=float(targets["vtt_wage_ratio"]),
        round_trip_factor=float(targets["round_trip_factor"]),
    )

    foc_block = targets.get("foc_inversion", {}) or {}
    use_common_margin_flag = bool(targets.get("use_common_margin", True))
    if force_chain_specific_costs:
        use_common_margin_flag = False

    costs = compute_marginal_costs(
        basket_price_standard_eur=float(targets["basket_price_standard_eur"]),
        price_index=dict(targets["price_index"]),
        gross_margin_common=float(targets["gross_margin_common"]),
        gross_margin_by_chain=dict(targets["gross_margin_by_chain"]),
        use_common_margin=use_common_margin_flag,
    )

    # Prices from the price ladder (ADR-023).
    basket = float(targets["basket_price_standard_eur"])
    price_index = dict(targets["price_index"])
    prices = {tau: basket * price_index[tau] for tau in _CHAIN_TYPES}

    # 2) Resolve absolute shares ---------------------------------------------
    s_outside = float(targets["outside_share_target"])
    s_B_over_s_D = float(foc_block["s_B_over_s_D"])
    s_S_over_s_D_raw = foc_block.get("s_S_over_s_D", None)
    s_S_over_s_D = (
        float(s_S_over_s_D_raw) if s_S_over_s_D_raw is not None else None
    )
    shares = compute_absolute_shares(
        s_outside=s_outside,
        s_B_over_s_D=s_B_over_s_D,
        s_S_over_s_D=s_S_over_s_D,
    )
    # Effective ratio used (after default resolution)
    s_S_over_s_D_used = shares["standard"] / shares["discount"]

    # 3) μ from FOC inversion ------------------------------------------------
    foc_diag = compute_mu_from_foc(prices=prices, costs=costs, shares=shares)
    if foc_block.get("use_share_weighted_mu", True):
        mu_hat = foc_diag["mu_share_weighted"]
    else:
        mu_hat = foc_diag["mu_by_type"]["standard"]

    # 4) Read travel-time matrix and cell_mass directly (avoid circular city) -
    # We need: dist_minutes (M, N), chain_types (N,), cell_mass (M,).
    # Use load_berlin_city with placeholder q values just to obtain dist + masses;
    # the q values do not affect dist or pop. Use a tentative alpha derived from
    # alpha_ratio with the current env_cfg outside_option as a placeholder a0.
    from hotelling.spatial.loader import load_berlin_city

    alpha_ratio = float(
        foc_block.get("alpha_ratio",
                       env_cfg.get("alpha_ratio",
                                   targets.get(
                                       "alpha_ratio_default", 2.5)))
    )
    # First-pass placeholder city to read pi_H_bar and the dist matrix.
    placeholder_city, _ = load_berlin_city(
        grid_path=grid_path,
        stores_path=stores_path,
        travel_times_path=travel_times_path,
        lambda_val=lambda_val,
        q_S=1.0, q_B=2.0,
        alpha_L=1.0, alpha_H=1.0,
        beta_effort=float(env_cfg.get("beta_effort", 0.001)),
        kappa0=float(env_cfg.get("kappa0", 1.0)),
        store_size=float(env_cfg.get("store_size", 600.0)),
        transport_cost=t,
        a0=float(env_cfg.get("outside_option", -5.0)),
        mu=mu_hat,
        nan_fill_minutes=float(env_cfg.get("nan_fill_minutes", 120.0)),
        marginal_cost_D=costs["discount"],
        marginal_cost_S=costs["standard"],
        marginal_cost_B=costs["bio"],
        rent_scale=float(env_cfg.get("rent_scale", 0.0)),
        rent_normalization=str(env_cfg.get("rent_normalization", "mean_ratio")),
        dense_distances=True,
    )

    pi_H_bar = _pi_H_bar(placeholder_city)
    alpha_L, alpha_H = _alphas_from_ratio(alpha_ratio, pi_H_bar)

    chain_types = load_chain_types_from_parquet(stores_path)
    cell_mass = (placeholder_city.cell_pop + placeholder_city.lambda_phi).astype(
        np.float64
    )
    dist_minutes = np.asarray(placeholder_city.dist2_km2, dtype=np.float64)

    # 5) Accessibility -------------------------------------------------------
    accessibility = compute_accessibility_by_type(
        dist_minutes=dist_minutes,
        chain_types=chain_types,
        cell_mass=cell_mass,
        transport_cost=t,
        mu=mu_hat,
    )

    # 6) Closed-form q -------------------------------------------------------
    q_dict = compute_q_closed_form(
        mu=mu_hat,
        shares=shares,
        accessibility=accessibility,
        prices=prices,
    )
    q_S, q_B = q_dict["q_S"], q_dict["q_B"]

    # 7) Optional a_0 refinement --------------------------------------------
    a0_initial = float(env_cfg.get("outside_option", -5.0))
    a0_refined_flag = bool(refine_a0 and foc_block.get("refine_a0", True))
    if a0_refined_flag:
        a0 = _refine_a0_root_find(
            grid_path=grid_path,
            stores_path=stores_path,
            travel_times_path=travel_times_path,
            lambda_val=lambda_val,
            env_cfg=env_cfg,
            transport_cost=t,
            costs=costs,
            mu=mu_hat,
            q_S=q_S, q_B=q_B,
            alpha_L=alpha_L, alpha_H=alpha_H,
            target_outside_share=s_outside,
            a0_initial=a0_initial,
        )
    else:
        a0 = a0_initial

    # 8) Final city for moment report ---------------------------------------
    final_city = _build_calibration_city(
        grid_path=grid_path,
        stores_path=stores_path,
        travel_times_path=travel_times_path,
        lambda_val=lambda_val,
        env_cfg=env_cfg,
        transport_cost=t,
        costs=costs,
        mu=mu_hat,
        a0=a0,
        q_S=q_S, q_B=q_B,
        alpha_L=alpha_L, alpha_H=alpha_H,
    )
    moments_model = all_model_moments(final_city, t, q_S, q_B)

    moments_target = {
        "mean_gross_margin": (
            float(targets["gross_margin_common"])
            if use_common_margin_flag is False
            else float(targets["gross_margin_common"])
        ),
        "outside_share": s_outside,
        "chain_share_discount": shares["discount"] / (1.0 - s_outside),
        "chain_share_bio": shares["bio"] / (1.0 - s_outside),
        "bio_income_gradient": float(
            targets["bio_share_income_gradient_target"]
        ),
    }

    return {
        "method": "foc_inversion",
        "t": float(t),
        "c": costs,
        "mu": float(mu_hat),
        "a0": float(a0),
        "q_S": float(q_S),
        "q_B": float(q_B),
        "alpha_L": float(alpha_L),
        "alpha_H": float(alpha_H),
        "alpha_ratio": float(alpha_ratio),
        "pi_H_bar": float(pi_H_bar),
        "moments_model": moments_model,
        "moments_target": moments_target,
        "residual_norm": float("nan"),  # not applicable for closed-form
        "success": True,
        "nfev": 0,
        # FOC-inversion-specific diagnostics:
        "foc_diagnostics": foc_diag,
        "shares_resolved": shares,
        "accessibility": accessibility,
        "s_S_over_s_D_used": float(s_S_over_s_D_used),
        "s_B_over_s_D_used": float(s_B_over_s_D),
        "a0_refined": a0_refined_flag,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _refine_a0_root_find(
    *,
    grid_path: str,
    stores_path: str,
    travel_times_path: str,
    lambda_val: float,
    env_cfg: dict,
    transport_cost: float,
    costs: Dict[str, float],
    mu: float,
    q_S: float, q_B: float,
    alpha_L: float, alpha_H: float,
    target_outside_share: float,
    a0_initial: float,
    bracket_width: float = 15.0,
    xtol: float = 1e-3,
    max_iter: int = 60,
) -> float:
    """1D root-find for a_0 such that model outside share matches target.

    Uses scipy.optimize.brentq on f(a0) = outside_share_model(a0) − target.
    The function is monotone in a_0 (more negative a_0 → lower outside
    share). Brackets [a0_initial − bracket_width, a0_initial + bracket_width]
    and expands once if needed.
    """
    from scipy.optimize import brentq
    from hotelling.calibration.structural import _build_calibration_city
    from hotelling.calibration.moments import all_model_moments

    def _outside_share_at(a0_trial: float) -> float:
        city = _build_calibration_city(
            grid_path=grid_path,
            stores_path=stores_path,
            travel_times_path=travel_times_path,
            lambda_val=lambda_val,
            env_cfg=env_cfg,
            transport_cost=transport_cost,
            costs=costs,
            mu=mu,
            a0=a0_trial,
            q_S=q_S, q_B=q_B,
            alpha_L=alpha_L, alpha_H=alpha_H,
        )
        m = all_model_moments(city, transport_cost, q_S, q_B)
        return m["outside_share"] - target_outside_share

    # Initial bracket
    a0_lo = a0_initial - bracket_width
    a0_hi = a0_initial + bracket_width
    f_lo = _outside_share_at(a0_lo)
    f_hi = _outside_share_at(a0_hi)

    # Expand bracket up to 3 times if sign-bracketing fails
    expansion = 0
    while f_lo * f_hi > 0.0 and expansion < 3:
        bracket_width *= 2.0
        a0_lo = a0_initial - bracket_width
        a0_hi = a0_initial + bracket_width
        f_lo = _outside_share_at(a0_lo)
        f_hi = _outside_share_at(a0_hi)
        expansion += 1

    if f_lo * f_hi > 0.0:
        logger.warning(
            "a0 root-find could not bracket the target outside share "
            "(target=%.4f, model_lo=%.4f, model_hi=%.4f). "
            "Returning a0_initial=%.4f unchanged.",
            target_outside_share,
            f_lo + target_outside_share,
            f_hi + target_outside_share,
            a0_initial,
        )
        return a0_initial

    a0_root = brentq(
        _outside_share_at, a0_lo, a0_hi, xtol=xtol, maxiter=max_iter
    )
    logger.info("a0 refined: %.6f → %.6f", a0_initial, a0_root)
    return float(a0_root)
