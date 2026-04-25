"""
Metrics computation for the Hotelling simulation.

Provides helpers that summarise the market state at any given step into a
flat dictionary suitable for logging, plotting, and downstream analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List


def compute_metrics(step: int, market) -> Dict[str, Any]:  # noqa: ANN001
    """Compute per-firm and market-level metrics for a single step.

    Parameters
    ----------
    step:
        Current simulation step index.
    market:
        A :class:`~hotelling.core.market.Market` instance with resolved
        consumer assignments.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``"step"`` – simulation step.
        * ``"firms"`` – list of per-firm metric dicts.
        * ``"hhi"`` – Herfindahl–Hirschman Index (market concentration).
        * ``"avg_price"`` – market-share-weighted average price.
        * ``"price_dispersion"`` – standard deviation of firm prices.
    """
    firms_data: List[Dict[str, Any]] = []
    shares: List[float] = []
    prices: List[float] = []

    for firm in market.firms:
        firms_data.append(
            {
                "firm_id": firm.firm_id,
                "name": firm.name,
                "location_x": firm.location[0],
                "location_y": firm.location[1],
                "price": firm.price,
                "market_share": firm.market_share,
                "profit": firm.profit,
            }
        )
        shares.append(firm.market_share)
        prices.append(firm.price)

    hhi = sum(s**2 for s in shares)
    n = len(prices)
    avg_price = (
        sum(p * s for p, s in zip(prices, shares)) if shares else 0.0
    )
    if n > 1:
        mean_p = sum(prices) / n
        price_dispersion = (sum((p - mean_p) ** 2 for p in prices) / n) ** 0.5
    else:
        price_dispersion = 0.0

    return {
        "step": step,
        "firms": firms_data,
        "hhi": round(hhi, 6),
        "avg_price": round(avg_price, 6),
        "price_dispersion": round(price_dispersion, 6),
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of per-step metrics into time-series arrays.

    Parameters
    ----------
    results:
        List of metric dicts as returned by :func:`compute_metrics`.

    Returns
    -------
    dict
        Dictionary with ``"steps"``, ``"hhi"``, ``"avg_price"``,
        ``"price_dispersion"``, and per-firm series.
    """
    if not results:
        return {}

    steps = [r["step"] for r in results]
    hhi = [r["hhi"] for r in results]
    avg_price = [r["avg_price"] for r in results]
    price_dispersion = [r["price_dispersion"] for r in results]

    # Per-firm time series
    firm_ids = [f["firm_id"] for f in results[0]["firms"]]
    per_firm: Dict[str, Dict[str, List]] = {
        fid: {"market_share": [], "profit": [], "price": [], "x": [], "y": []}
        for fid in firm_ids
    }
    for r in results:
        for fd in r["firms"]:
            fid = fd["firm_id"]
            if fid in per_firm:
                per_firm[fid]["market_share"].append(fd["market_share"])
                per_firm[fid]["profit"].append(fd["profit"])
                per_firm[fid]["price"].append(fd["price"])
                per_firm[fid]["x"].append(fd["location_x"])
                per_firm[fid]["y"].append(fd["location_y"])

    return {
        "steps": steps,
        "hhi": hhi,
        "avg_price": avg_price,
        "price_dispersion": price_dispersion,
        "firms": per_firm,
    }
