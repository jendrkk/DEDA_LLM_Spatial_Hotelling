from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from scipy.optimize import least_squares

from hotelling.calibration.moments import all_model_moments
from hotelling.spatial.loader import load_berlin_city

logger = logging.getLogger(__name__)

_CHAIN_SHARE_COUNT_PROXY = {
    "discount": 0.397,
    "standard": 0.419,
    "bio": 0.184,
}


def compute_transport_cost(
    wage_monthly_gross_eur: float,
    work_hours_per_month: float,
    vtt_wage_ratio: float,
    round_trip_factor: float,
) -> float:
    """Return the transport-cost coefficient t (euros per ONE-WAY minute)
    to multiply one-way transit minutes in the utility function.

    t = round_trip_factor * vtt_wage_ratio * (wage_monthly_gross_eur
                                              / work_hours_per_month) / 60.0

    The division by 60 converts euro/hour to euro/minute. round_trip_factor
    accounts for travel_times being one-way while a shopping occasion is a
    round trip. See ADR-024.
    """
    wage_per_hour = wage_monthly_gross_eur / work_hours_per_month
    wage_per_minute = wage_per_hour / 60.0
    return round_trip_factor * vtt_wage_ratio * wage_per_minute


def compute_marginal_costs(
    basket_price_standard_eur: float,
    price_index: Dict[str, float],
    gross_margin_common: float,
    gross_margin_by_chain: Dict[str, float],
    use_common_margin: bool,
) -> Dict[str, float]:
    """Return marginal cost per chain type {'discount','standard','bio'} in
    euros per basket.

    p_tau = basket_price_standard_eur * price_index[tau]
    margin_tau = gross_margin_common (if use_common_margin) else
                 gross_margin_by_chain[tau]
    c_tau = p_tau * (1 - margin_tau)

    Guarantees c_discount < c_standard < c_bio whenever the price indices are
    increasing and margins do not increase faster than prices. The function
    asserts this ordering and raises ValueError if violated. See ADR-025.
    """
    chain_types = ("discount", "standard", "bio")
    c: Dict[str, float] = {}
    for tau in chain_types:
        p_tau = basket_price_standard_eur * price_index[tau]
        if use_common_margin:
            margin_tau = gross_margin_common
        else:
            margin_tau = gross_margin_by_chain[tau]
        c[tau] = p_tau * (1.0 - margin_tau)

    if not (c["discount"] < c["standard"] < c["bio"]):
        raise ValueError(
            f"Marginal cost ordering violated: "
            f"discount={c['discount']}, standard={c['standard']}, bio={c['bio']}"
        )
    return c


def compute_qualities(
    basket_price_standard_eur: float,
    price_index: Dict[str, float],
) -> tuple[float, float]:
    """Return (q_S, q_B) in euros from the observed price ladder.

    With the population-weighted normalization alpha_bar = 1, the average
    consumer's willingness-to-pay premium for a chain type equals its market
    price premium over discount:
        q_S = p_S - p_D
        q_B = p_B - p_D
    where p_tau = basket_price_standard_eur * price_index[tau].

    Asserts 0 < q_S < q_B; raises ValueError otherwise. See ADR-028.
    """
    p_D = basket_price_standard_eur * price_index["discount"]
    p_S = basket_price_standard_eur * price_index["standard"]
    p_B = basket_price_standard_eur * price_index["bio"]
    q_S = p_S - p_D
    q_B = p_B - p_D
    if not (0.0 < q_S < q_B):
        raise ValueError(
            f"Quality ordering violated: q_S={q_S}, q_B={q_B} "
            f"(from price ladder {price_index})"
        )
    return q_S, q_B


def _alphas_from_ratio(alpha_ratio: float, pi_H_bar: float) -> tuple[float, float]:
    """Return (alpha_L, alpha_H) given the ratio and the mass-weighted mean
    high-type share, normalized so pi_L_bar*alpha_L + pi_H_bar*alpha_H = 1."""
    pi_L_bar = 1.0 - pi_H_bar
    denom = pi_L_bar + alpha_ratio * pi_H_bar
    alpha_L = 1.0 / denom
    alpha_H = alpha_ratio * alpha_L
    return alpha_L, alpha_H


def _pi_H_bar(city) -> float:
    mass = city.cell_pop + city.lambda_phi
    total = float(mass.sum())
    if total <= 0.0:
        raise ValueError("Total consumer mass is zero; cannot compute pi_H_bar")
    return float(np.dot(city.pi_H, mass) / total)


def _margin_target(targets: dict) -> float:
    if targets.get("use_common_margin", True):
        return float(targets["gross_margin_common"])
    margins = targets["gross_margin_by_chain"]
    return float(
        sum(margins[tau] for tau in ("discount", "standard", "bio")) / 3.0
    )


def _moment_targets(targets: dict) -> Dict[str, float]:
    return {
        "mean_gross_margin": _margin_target(targets),
        "outside_share": float(targets["outside_share_target"]),
    }


def _verify_firm_chain_types(city) -> None:
    for firm in city.firms:
        if firm.chain_type not in ("discount", "standard", "bio"):
            raise ValueError(
                f"Firm {firm.id} has chain_type={firm.chain_type!r}; the loader "
                "must populate chain_type on every store. Re-run with the "
                "updated loader (ADR-028 / Firm.chain_type)."
            )


def _build_calibration_city(
    *,
    grid_path: str,
    stores_path: str,
    travel_times_path: str,
    lambda_val: float,
    env_cfg: dict,
    transport_cost: float,
    costs: Dict[str, float],
    mu: float,
    a0: float,
    q_S: float,
    q_B: float,
    alpha_L: float,
    alpha_H: float,
):
    city, _firms = load_berlin_city(
        grid_path=grid_path,
        stores_path=stores_path,
        travel_times_path=travel_times_path,
        lambda_val=lambda_val,
        q_S=q_S,
        q_B=q_B,
        alpha_L=alpha_L,
        alpha_H=alpha_H,
        beta_effort=float(env_cfg.get("beta_effort", 0.001)),
        kappa0=float(env_cfg.get("kappa0", 1.0)),
        store_size=float(env_cfg.get("store_size", 600.0)),
        transport_cost=transport_cost,
        a0=a0,
        mu=mu,
        nan_fill_minutes=float(env_cfg.get("nan_fill_minutes", 120.0)),
        marginal_cost_D=costs["discount"],
        marginal_cost_S=costs["standard"],
        marginal_cost_B=costs["bio"],
        rent_scale=float(env_cfg.get("rent_scale", 0.0)),
        rent_normalization=str(env_cfg.get("rent_normalization", "mean_ratio")),
        dense_distances=True,
    )
    return city


def compute_effort_params(
    city,
    transport_cost: float,
    basket_price_standard_eur: float,
    e_max: float,
    X: float,
    rho: float,
) -> dict:
    """Calibrate (beta_effort, kappa0) at the euro scale. ADR-031.

    beta = X * basket_price / e_max  (price-scale anchor: WTP for full
    effort = X% of basket). kappa0 = beta * D_bar / (rho * e_max) where
    D_bar = mean store demand at the PRICE-ONLY Nash, so mean(e*)=rho*e_max.
    """
    from hotelling.core.equilibrium import bertrand_nash
    from hotelling.core.market import logit_demand

    N = len(city.firms)
    saved_beta = city.beta
    city.beta = 0.0
    try:
        p_nash, _ = bertrand_nash(
            city, transport_cost=transport_cost, cache_path=None
        )
        quals = np.array([f.quality for f in city.firms], dtype=np.float64)
        D = logit_demand(
            p_nash,
            np.zeros(N),
            city.dist2_km2,
            city.cell_pop,
            city.lambda_phi,
            city.pi_H,
            city.pi_H_lambda_phi,
            city.alpha,
            quals,
            beta=0.0,
            transport_cost=transport_cost,
            mu=city.mu,
            a0=city.a0,
            transport_exponent=getattr(city, "transport_exponent", 1.0),
        )
    finally:
        city.beta = saved_beta

    D_bar = float(D.mean())
    beta = X * basket_price_standard_eur / e_max
    kappa0 = beta * D_bar / (rho * e_max)
    e_star = beta * D / kappa0
    interior = float(np.mean((e_star > 1e-9) & (e_star < e_max)))
    return {
        "beta_effort": float(beta),
        "kappa0": float(kappa0),
        "e_max": float(e_max),
        "X": float(X),
        "rho": float(rho),
        "D_bar": D_bar,
        "e_star_mean": float(e_star.mean()),
        "e_star_min": float(e_star.min()),
        "e_star_max": float(e_star.max()),
        "interior_fraction": interior,
        "wtp_full_pct": float(beta * e_max / basket_price_standard_eur),
        "wtp_equil_pct": float(
            beta * e_star.mean() / basket_price_standard_eur
        ),
    }


def calibrate_structural(
    targets: dict,
    env_cfg: dict,
    grid_path: str,
    stores_path: str,
    travel_times_path: str,
    lambda_val: float,
    x0: dict | None = None,
    max_nfev: int = 40,
) -> dict:
    """Run structural calibration: fix t, c, q_S, q_B, alpha_ratio from data;
    solve only (mu, a0) by method of moments. See ADR-028."""
    t = compute_transport_cost(
        wage_monthly_gross_eur=float(targets["wage_monthly_gross_eur"]),
        work_hours_per_month=float(targets["work_hours_per_month"]),
        vtt_wage_ratio=float(targets["vtt_wage_ratio"]),
        round_trip_factor=float(targets["round_trip_factor"]),
    )
    costs = compute_marginal_costs(
        basket_price_standard_eur=float(targets["basket_price_standard_eur"]),
        price_index=targets["price_index"],
        gross_margin_common=float(targets["gross_margin_common"]),
        gross_margin_by_chain=targets["gross_margin_by_chain"],
        use_common_margin=bool(targets.get("use_common_margin", True)),
    )
    q_S, q_B = compute_qualities(
        float(targets["basket_price_standard_eur"]),
        targets["price_index"],
    )
    alpha_ratio = float(targets["alpha_ratio"])

    mu0 = float((x0 or {}).get("mu", 6.0))
    a00 = float((x0 or {}).get("a0", -5.0))

    city = _build_calibration_city(
        grid_path=grid_path,
        stores_path=stores_path,
        travel_times_path=travel_times_path,
        lambda_val=lambda_val,
        env_cfg=env_cfg,
        transport_cost=t,
        costs=costs,
        mu=mu0,
        a0=a00,
        q_S=q_S,
        q_B=q_B,
        alpha_L=1.0,
        alpha_H=1.0,
    )
    _verify_firm_chain_types(city)

    pi_H_bar = _pi_H_bar(city)
    alpha_L, alpha_H = _alphas_from_ratio(alpha_ratio, pi_H_bar)
    city.alpha = np.array([alpha_L, alpha_H], dtype=np.float64)

    moment_target = _moment_targets(targets)
    tgt_margin = moment_target["mean_gross_margin"]
    tgt_outside = moment_target["outside_share"]

    eval_count = 0

    def residuals(y: np.ndarray) -> np.ndarray:
        nonlocal eval_count
        eval_count += 1
        mu = float(np.exp(y[0]))
        a0 = float(y[1])
        try:
            city.mu = mu
            city.a0 = a0
            moments = all_model_moments(city, t, q_S, q_B)
            r = np.array(
                [
                    (moments["mean_gross_margin"] - tgt_margin) / tgt_margin,
                    (moments["outside_share"] - tgt_outside) / tgt_outside,
                ],
                dtype=np.float64,
            )
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            logger.warning(
                "eval %d failed (mu=%.4f a0=%.4f): %s",
                eval_count,
                mu,
                a0,
                exc,
            )
            r = np.full(2, 1e3, dtype=np.float64)
        logger.info(
            "eval %d: mu=%.4f a0=%.4f "
            "rel[margin]=%.6f rel[outside]=%.6f |res|=%.6f",
            eval_count,
            mu,
            a0,
            float(r[0]),
            float(r[1]),
            float(np.linalg.norm(r)),
        )
        return r

    y0 = np.array([np.log(mu0), a00], dtype=np.float64)
    lower = np.array([np.log(0.5), -50.0], dtype=np.float64)
    upper = np.array([np.log(25.0), 0.0], dtype=np.float64)

    result = least_squares(
        residuals,
        y0,
        method="trf",
        bounds=(lower, upper),
        max_nfev=max_nfev,
        diff_step=0.05,
    )

    mu = float(np.exp(result.x[0]))
    a0 = float(result.x[1])
    city.mu = mu
    city.a0 = a0
    moments_model = all_model_moments(city, t, q_S, q_B)

    effort = compute_effort_params(
        city,
        transport_cost=t,
        basket_price_standard_eur=float(targets["basket_price_standard_eur"]),
        e_max=float(targets.get("effort_e_max", 1.0)),
        X=float(targets.get("effort_importance_X", 0.10)),
        rho=float(targets.get("effort_interior_target_rho", 0.40)),
    )

    return {
        "t": t,
        "c": costs,
        "mu": mu,
        "a0": a0,
        "q_S": q_S,
        "q_B": q_B,
        "alpha_L": alpha_L,
        "alpha_H": alpha_H,
        "alpha_ratio": alpha_ratio,
        "pi_H_bar": pi_H_bar,
        "moments_model": moments_model,
        "moments_target": moment_target,
        "validation_targets": {
            "chain_share_discount": _CHAIN_SHARE_COUNT_PROXY["discount"],
            "chain_share_standard": _CHAIN_SHARE_COUNT_PROXY["standard"],
            "chain_share_bio": _CHAIN_SHARE_COUNT_PROXY["bio"],
            "bio_income_gradient_ref": float(
                targets["bio_share_income_gradient_target"]
            ),
        },
        "residual_norm": float(np.linalg.norm(result.fun)),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "effort": effort,
    }
