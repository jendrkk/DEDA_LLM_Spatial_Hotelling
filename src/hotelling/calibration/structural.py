from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from scipy.optimize import least_squares

from hotelling.calibration.moments import all_model_moments
from hotelling.spatial.loader import load_berlin_city

logger = logging.getLogger(__name__)


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
    shares = targets["chain_share_target"]
    margins = targets["gross_margin_by_chain"]
    return float(
        sum(shares[tau] * margins[tau] for tau in ("discount", "standard", "bio"))
    )


def _moment_targets(targets: dict) -> Dict[str, float]:
    chain = targets["chain_share_target"]
    return {
        "mean_gross_margin": _margin_target(targets),
        "outside_share": float(targets["outside_share_target"]),
        "chain_share_discount": float(chain["discount"]),
        "chain_share_bio": float(chain["bio"]),
        "bio_income_gradient": float(targets["bio_share_income_gradient_target"]),
    }


def _pack_params(
    mu: float,
    a0: float,
    q_S: float,
    q_B: float,
    alpha_ratio: float,
) -> np.ndarray:
    if q_B <= q_S:
        q_B = q_S + 1e-3
    if alpha_ratio <= 1.0:
        alpha_ratio = 1.0 + 1e-3
    if mu <= 0.0 or q_S <= 0.0:
        raise ValueError("mu and q_S must be positive for parameter packing")
    return np.array(
        [
            np.log(mu),
            a0,
            np.log(q_S),
            np.log(q_B - q_S),
            np.log(alpha_ratio - 1.0),
        ],
        dtype=np.float64,
    )


def _unpack_params(x: np.ndarray) -> tuple[float, float, float, float, float]:
    mu = float(np.exp(x[0]))
    a0 = float(x[1])
    q_S = float(np.exp(x[2]))
    q_B = q_S + float(np.exp(x[3]))
    alpha_ratio = 1.0 + float(np.exp(x[4]))
    return mu, a0, q_S, q_B, alpha_ratio


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


def calibrate_structural(
    targets: dict,
    env_cfg: dict,
    grid_path: str,
    stores_path: str,
    travel_times_path: str,
    lambda_val: float,
    x0: dict | None = None,
    max_nfev: int = 60,
) -> dict:
    """Run the full structural calibration and return a dict with the final
    calibrated parameters and the achieved vs target moments.

    See module docstring in ADR-026 for the five-moment just-identified design.
    """
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

    moment_target = _moment_targets(targets)

    init_city = _build_calibration_city(
        grid_path=grid_path,
        stores_path=stores_path,
        travel_times_path=travel_times_path,
        lambda_val=lambda_val,
        env_cfg=env_cfg,
        transport_cost=t,
        costs=costs,
        mu=float((x0 or {}).get("mu", env_cfg.get("logit_scale", 5.0))),
        a0=float((x0 or {}).get("a0", env_cfg.get("outside_option", -5.0))),
        q_S=float((x0 or {}).get("q_S", env_cfg.get("q_S", 3.0))),
        q_B=float((x0 or {}).get("q_B", env_cfg.get("q_B", 8.0))),
        alpha_L=float(env_cfg.get("alpha_L", 0.5)),
        alpha_H=float(env_cfg.get("alpha_H", 1.5)),
    )
    pi_H_bar = _pi_H_bar(init_city)

    if x0 is not None and "alpha_ratio" in x0:
        alpha_ratio0 = float(x0["alpha_ratio"])
    else:
        alpha_L0 = float(env_cfg.get("alpha_L", 0.5))
        alpha_H0 = float(env_cfg.get("alpha_H", 1.5))
        alpha_ratio0 = alpha_H0 / alpha_L0

    x0_vec = _pack_params(
        mu=float((x0 or {}).get("mu", env_cfg.get("logit_scale", 5.0))),
        a0=float((x0 or {}).get("a0", env_cfg.get("outside_option", -5.0))),
        q_S=float((x0 or {}).get("q_S", env_cfg.get("q_S", 3.0))),
        q_B=float((x0 or {}).get("q_B", env_cfg.get("q_B", 8.0))),
        alpha_ratio=alpha_ratio0,
    )

    eval_count = 0

    def residuals(x: np.ndarray) -> np.ndarray:
        nonlocal eval_count
        eval_count += 1
        mu, a0, q_S, q_B, alpha_ratio = _unpack_params(x)
        alpha_L, alpha_H = _alphas_from_ratio(alpha_ratio, pi_H_bar)
        try:
            city = _build_calibration_city(
                grid_path=grid_path,
                stores_path=stores_path,
                travel_times_path=travel_times_path,
                lambda_val=lambda_val,
                env_cfg=env_cfg,
                transport_cost=t,
                costs=costs,
                mu=mu,
                a0=a0,
                q_S=q_S,
                q_B=q_B,
                alpha_L=alpha_L,
                alpha_H=alpha_H,
            )
            moments = all_model_moments(city, t, q_S, q_B)
            r = np.array(
                [
                    (moments["mean_gross_margin"] - moment_target["mean_gross_margin"])
                    / moment_target["mean_gross_margin"],
                    (moments["outside_share"] - moment_target["outside_share"])
                    / moment_target["outside_share"],
                    (
                        moments["chain_share_discount"]
                        - moment_target["chain_share_discount"]
                    )
                    / moment_target["chain_share_discount"],
                    (moments["chain_share_bio"] - moment_target["chain_share_bio"])
                    / moment_target["chain_share_bio"],
                    (
                        moments["bio_income_gradient"]
                        - moment_target["bio_income_gradient"]
                    )
                    / moment_target["bio_income_gradient"],
                ],
                dtype=np.float64,
            )
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            logger.warning(
                "eval %d failed (mu=%.4f a0=%.4f q_S=%.4f q_B=%.4f): %s",
                eval_count,
                mu,
                a0,
                q_S,
                q_B,
                exc,
            )
            r = np.full(5, 1e3, dtype=np.float64)
        logger.info(
            "eval %d: mu=%.4f a0=%.4f q_S=%.4f q_B=%.4f alpha_ratio=%.4f "
            "|res|=%.6f",
            eval_count,
            mu,
            a0,
            q_S,
            q_B,
            alpha_ratio,
            float(np.linalg.norm(r)),
        )
        return r

    result = least_squares(
        residuals,
        x0_vec,
        method="trf",
        max_nfev=max_nfev,
    )

    mu, a0, q_S, q_B, alpha_ratio = _unpack_params(result.x)
    alpha_L, alpha_H = _alphas_from_ratio(alpha_ratio, pi_H_bar)
    final_city = _build_calibration_city(
        grid_path=grid_path,
        stores_path=stores_path,
        travel_times_path=travel_times_path,
        lambda_val=lambda_val,
        env_cfg=env_cfg,
        transport_cost=t,
        costs=costs,
        mu=mu,
        a0=a0,
        q_S=q_S,
        q_B=q_B,
        alpha_L=alpha_L,
        alpha_H=alpha_H,
    )
    moments_model = all_model_moments(final_city, t, q_S, q_B)

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
        "residual_norm": float(np.linalg.norm(result.fun)),
        "success": bool(result.success),
        "nfev": int(result.nfev),
    }
