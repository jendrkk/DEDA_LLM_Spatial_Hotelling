"""Impulse Response Function analysis.

Responsibility: measure how a one-shot price deviation by one agent propagates
through the market over subsequent periods, following the methodology of
Calvano et al. (2020 AER) Figure 4.

Public API: impulse_response

Key dependencies: pandas, numpy

References:
    Calvano et al. (2020 AER) §IV Fig. 4 - IRF methodology;
    Sims (1980) - original IRF concept in macroeconomics.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def impulse_response(
    policy: Any,
    env: Any,
    deviator: str,
    deviation_price_index: Optional[int] = None,
    horizon: int = 25,
) -> pd.DataFrame:
    """Compute impulse response to a one-shot price deviation.

    Starting from a trained steady state (burn-in), one agent (deviator)
    deviates to ``deviation_price_index`` for a single period, then returns to
    its trained policy.  All other agents follow their trained policies
    throughout.  The response of all agents over the following ``horizon``
    periods is recorded.

    Parameters
    ----------
    policy : dict mapping agent_id -> trained AgentProtocol instance
    env : initialized HotellingMarketEnv (will be reset internally)
    deviator : agent_id of the deviating firm
    deviation_price_index : price index to deviate to;
        None defaults to the Nash price (lowest price index)
    horizon : number of periods to simulate after the deviation

    Returns
    -------
    pd.DataFrame with columns:
        period (int), agent_id (str), price_index (int), price (float),
        profit (float), is_deviation_period (bool)
    """
    raise NotImplementedError
