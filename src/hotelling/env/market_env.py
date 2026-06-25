"""PettingZoo ParallelEnv wrapper for the spatial Hotelling market.

Responsibility: implement the core simulation loop as a PettingZoo
ParallelEnv; manage per-period price/effort decisions, demand clearing,
and reward computation.

Public API: HotellingMarketEnv

Two stepping interfaces
-----------------------
``step(actions: dict)``
    Dict-based API; compatible with the single-agent :class:`~hotelling.simulation.engine.SimulationEngine`.
    Accepts ``{agent_id: joint_action_int}`` and returns the standard
    PettingZoo 5-tuple ``(obs, rewards, terms, truncs, infos)`` as dicts.
    Internally delegates to ``step_array``.

``step_array(actions: np.ndarray) -> (neighbor_actions, rewards, demands)``
    Array-only hot path; used by :class:`~hotelling.simulation.engine.BatchSimulationEngine`.
    Accepts a ``(N,)`` int64 array of joint action indices and returns three
    arrays without any dict construction.  Per-firm attribute arrays are
    precomputed in ``__init__`` and reused every call.

Key dependencies: pettingzoo, numpy, hotelling.core

References: ADR-003; docs/agent_simulation_technical_report.md §3.
"""
from __future__ import annotations

import logging
from typing import Any

import numba as nb
import numpy as np

from hotelling.core.market import (
    FirmArrays,
    market_clearing_arrays,
    precompute_firm_arrays,
)

logger = logging.getLogger(__name__)


@nb.njit(cache=True)
def _local_price_summary(prices, indptr, indices):
    N = indptr.shape[0] - 1
    mean = np.empty(N)
    mn = np.empty(N)
    for j in range(N):
        s = 0.0
        m = 1e18
        cnt = 0
        for p in range(indptr[j], indptr[j + 1]):
            v = prices[indices[p]]
            s += v
            cnt += 1
            if v < m:
                m = v
        if cnt > 0:
            mean[j] = s / cnt
            mn[j] = m
        else:
            mean[j] = prices[j]
            mn[j] = prices[j]
    return mean, mn


@nb.njit(cache=True)
def _gather_rival_price_bins(prices, rivals, edges, B):
    """Bin rivals' EUR prices onto a common axis; neutral bin (B//2) for absent rivals (-1).

    prices : (N,) float64  current EUR price of every store
    rivals : (N, k) int64   rival store indices, -1 = no rival in that slot
    edges  : (B+1,) float64 common bin edges (linspace over the union of all action grids)
    Returns (N, k) int64 bin indices in [0, B-1]; padded slots get the neutral middle bin.
    """
    N, k = rivals.shape
    out = np.empty((N, k), dtype=np.int64)
    mid = B // 2
    n_edges = edges.shape[0]  # = B + 1
    for j in range(N):
        for r in range(k):
            rv = rivals[j, r]
            if rv < 0:
                out[j, r] = mid
            else:
                v = prices[rv]
                b = 0
                for e in range(1, n_edges):
                    if v >= edges[e]:
                        b = e
                    else:
                        break
                if b >= B:
                    b = B - 1
                out[j, r] = b
    return out


class HotellingMarketEnv:
    """PettingZoo-compatible Hotelling market environment.

    Parameters
    ----------
    city : object  City object with boundary and firms attributes.
    firms : list  List of Firm objects.
    m : int  Number of discrete price levels.
    min_price : float | None  Lowest price level; defaults to min marginal cost.
    max_price : float | None  Highest price level; defaults to 2 * min_price.
    m_effort : int  Number of discrete effort levels.
    e_max : float  Maximum effort value.
    k_neighbors : int  Nearest-neighbor count for the observation.
    transport_cost : float  Transport disutility coefficient.
    """

    def __init__(
        self,
        city: Any,
        firms: list,
        m: int = 15,
        min_price: float | None = None,
        max_price: float | None = None,
        m_effort: int = 5,
        e_max: float = 10.0,
        k_neighbors: int = 1,
        transport_cost: float = 0.01,
        state_mode: str = "neighbors",
        local_sum_n: int | None = None,
        n_price_bins: int = 15,
        summary_stats: tuple | list = ("mean",),
        local_summary_detailed: bool = False,
        chain_type_grids: dict[str, np.ndarray] | None = None,
        n_comp_bins: int = 15,
        p_nash_arr: np.ndarray | None = None,
        p_mono_arr: np.ndarray | None = None,
        calvano_k: int = 1,
        hybrid_n_profit: int = 5,
        hybrid_n_gap: int = 9,
        hybrid_gap_lo: float = -0.20,
        hybrid_gap_hi: float = 0.20,
        graph_rivals: "np.ndarray | None" = None,
        graph_k: int = 2,
        graph_n_rival_bins: int = 10,
    ) -> None:
        self.city = city
        self.firms = firms
        self.m = m
        self.m_effort = m_effort
        self.e_max = e_max
        self.k_neighbors = k_neighbors
        self.transport_cost = transport_cost
        self.state_mode = state_mode
        self.local_sum_n = local_sum_n
        self.n_price_bins = int(n_price_bins)

        _valid_stats = {"mean", "min"}
        stats_set = set(summary_stats)
        if not stats_set:
            raise ValueError("summary_stats must be non-empty")
        if not stats_set <= _valid_stats:
            raise ValueError(
                f"summary_stats must be a subset of {_valid_stats}, got {summary_stats}"
            )
        self.summary_stats = [s for s in ("mean", "min") if s in stats_set]
        self.local_summary_detailed = bool(local_summary_detailed)
        self._ls_channels: list[tuple[str, str]] = []

        if self.state_mode not in (
            "neighbors", "local_summary",
            "design4_ownprice", "design5_full", "calvano_local", "strategic_hybrid",
            "hybrid_profit_gap", "graph_states",
        ):
            raise ValueError(
                f"Unknown state_mode: {state_mode!r}"
            )

        # Price grid
        mc_min = min(getattr(f, "marginal_cost", 0.0) for f in firms)
        self._min_price = min_price if min_price is not None else mc_min
        self._max_price = max_price if max_price is not None else max(2.0 * self._min_price, 2.0)
        self.price_grid: np.ndarray = np.linspace(self._min_price, self._max_price, self.m)

        # Chain-type-specific grids (optional; None = global grid for all stores)
        self._chain_type_grids = chain_type_grids
        if chain_type_grids is not None:
            N = len(firms)
            self._store_price_grids = np.zeros((N, self.m), dtype=np.float64)
            for j, f in enumerate(firms):
                ct = getattr(f, "chain_type", "standard")
                if ct in chain_type_grids:
                    self._store_price_grids[j] = chain_type_grids[ct]
                else:
                    self._store_price_grids[j] = self.price_grid
            logger.info(
                "Chain-specific grids active: %s",
                {ct: f"[{g[0]:.3f}, {g[-1]:.3f}]" for ct, g in chain_type_grids.items()},
            )
        else:
            self._store_price_grids = None

        # Effort grid: [0, e_max] with m_effort levels
        self.effort_grid: np.ndarray = np.linspace(0.0, e_max, m_effort)

        # Joint action space size
        self._action_size = m * m_effort

        # Agent lists
        self.possible_agents: list[str] = [str(f.id) for f in firms]
        self.agents: list[str] = list(self.possible_agents)

        # Starting joint action: mid price × zero effort
        mid_price_idx = m // 2
        mid_effort_idx = 0
        mid_joint = mid_price_idx * m_effort + mid_effort_idx

        # Dict-based action state (slow path / compatibility)
        self._current_joint_actions: dict[str, int] = {
            a: mid_joint for a in self.possible_agents
        }

        # Array-based action state (fast path) — shape (N,) int64
        N = len(firms)
        self._current_joint_actions_arr: np.ndarray = np.full(
            N, mid_joint, dtype=np.int64
        )

        # Precompute k-nearest-neighbor lists (dict, used by slow API)
        self._neighbor_indices: dict[str, list[int]] = self._build_neighbor_indices()

        # Precompute (N, k) int32 neighbor index array for the fast path.
        # Padding positions contain sentinel value N (= len(firms)).
        # In step_array these are resolved to mid_joint via a padded lookup.
        self._neighbor_idx: np.ndarray = self._build_neighbor_idx()

        # Precompute per-firm attribute arrays for the market-clearing hot path.
        self._firm_arrays: FirmArrays = precompute_firm_arrays(firms)

        self.n_comp_bins = int(n_comp_bins)
        self.calvano_k = int(calvano_k)
        self.hybrid_n_profit = int(hybrid_n_profit)
        self.hybrid_n_gap = int(hybrid_n_gap)
        self.hybrid_gap_lo = float(hybrid_gap_lo)
        self.hybrid_gap_hi = float(hybrid_gap_hi)
        # ── Hybrid profit-gap state backing storage (None = mode not active) ──────────
        # _hybrid_gap_edges : (n_gap+1,) float64 — bin edges for gap computation
        # _prev_rewards_arr : (N,)       float64 — profits from previous period (t−1)
        # _profit_quantile_thresholds: (3, n_profit-1) float64 — per-chain quantile thresholds
        # _chain_type_ids:              (N,)       int64 — 0=discount/1=standard/2=bio
        self._hybrid_gap_edges: np.ndarray | None = None
        self._prev_rewards_arr: np.ndarray | None = None
        self._profit_quantile_thresholds: np.ndarray | None = None
        self._chain_type_ids: np.ndarray | None = None
        self._comp_sets = None
        self._p_nash_arr = p_nash_arr
        self._p_mono_arr = p_mono_arr

        if self.state_mode == "local_summary":
            if self.local_summary_detailed:
                self._ls_channels = [("all", "mean"), ("same_type", "mean")]
            else:
                self._ls_channels = [("all", s) for s in self.summary_stats]
            self._init_local_summary_competitors()

        if self.state_mode in (
            "design4_ownprice", "design5_full", "calvano_local", "strategic_hybrid",
            "hybrid_profit_gap",
        ):
            from hotelling.env.competitors import build_competitor_sets
            if self._chain_type_grids is not None:
                _price_lo = float(min(g.min() for g in self._chain_type_grids.values()))
                _price_hi = float(max(g.max() for g in self._chain_type_grids.values()))
                _chain_price_ranges = {
                    ct: (float(g.min()), float(g.max()))
                    for ct, g in self._chain_type_grids.items()
                }
            else:
                _price_lo = float(self.price_grid.min())
                _price_hi = float(self.price_grid.max())
                _chain_price_ranges = None
            self._comp_sets = build_competitor_sets(
                firms=firms,
                catch_indptr=getattr(city, "catch_indptr", None),
                catch_indices=getattr(city, "catch_indices", None),
                n_bins=self.n_comp_bins,
                price_lo=_price_lo,
                price_hi=_price_hi,
                chain_type_price_ranges=_chain_price_ranges,
                min_overlap_cells=1,
                k_max_rivals=max(3, self.calvano_k),
                firm_locations=np.array([f.location for f in firms], dtype=np.float64),
            )

        if self.state_mode == "hybrid_profit_gap":
            from hotelling.env.competitors import (
                compute_hybrid_profit_thresholds,
                _CHAIN_TYPE_TO_ID,
            )
            N = len(firms)
            # Gap bin edges: (n_gap+1,) spanning [gap_lo, gap_hi]
            self._hybrid_gap_edges = np.linspace(
                self.hybrid_gap_lo, self.hybrid_gap_hi, self.hybrid_n_gap + 1,
            )
            # Previous-period reward buffer — init to zeros (first step → all stores
            # land in profit bin 0 "very bad"; settles naturally after a few steps)
            self._prev_rewards_arr = np.zeros(N, dtype=np.float64)
            # Chain-type integer IDs: (N,) int64 in {0,1,2}
            self._chain_type_ids = np.array(
                [
                    _CHAIN_TYPE_TO_ID.get(getattr(f, "chain_type", "standard"), 1)
                    for f in firms
                ],
                dtype=np.int64,
            )
            # Marginal costs for proxy-profit computation
            _mc_arr = np.array(
                [float(getattr(f, "marginal_cost", 0.0)) for f in firms],
                dtype=np.float64,
            )
            # Profit quantile thresholds: (3, n_profit-1) float64
            self._profit_quantile_thresholds = compute_hybrid_profit_thresholds(
                firms=firms,
                store_price_grids=self._store_price_grids,
                price_grid=self.price_grid,
                mc_arr=_mc_arr,
                n_profit=self.hybrid_n_profit,
            )
            # Guard: nearest_same_type needs ≥ 2 columns for gap computation
            if (
                self._comp_sets is None
                or self._comp_sets.nearest_same_type is None
                or self._comp_sets.nearest_same_type.shape[1] < 2
            ):
                raise RuntimeError(
                    "hybrid_profit_gap requires CompetitorSets.nearest_same_type "
                    "with ≥ 2 columns. Ensure catch_indptr is available in city "
                    "and k_max_rivals ≥ 2 in build_competitor_sets."
                )
            logger.info(
                "hybrid_profit_gap: m=%d × n_profit=%d × n_gap=%d² = %d states | "
                "gap ∈ [%.2f, %.2f], step≈%.3f | action_size=%d",
                self.m, self.hybrid_n_profit, self.hybrid_n_gap,
                self.m * self.hybrid_n_profit * self.hybrid_n_gap ** 2,
                self.hybrid_gap_lo, self.hybrid_gap_hi,
                (self.hybrid_gap_hi - self.hybrid_gap_lo) / self.hybrid_n_gap,
                self._action_size,
            )

        # ── graph_states: reciprocal rival observation graph ─────────────────────────
        # graph_rivals (N, graph_k) int64, -1 padded — supplied by the runner after it
        # builds the graph at Bertrand-Nash (hotelling.env.rival_graph). Rival prices are
        # re-projected onto a common EUR axis of graph_n_rival_bins bins so cross-type
        # rivals (different action grids) are comparable.  state_size = m * B^k.
        self.graph_k = int(graph_k)
        self.graph_n_rival_bins = int(graph_n_rival_bins)
        self.graph_rivals = (
            np.ascontiguousarray(graph_rivals, dtype=np.int64)
            if graph_rivals is not None
            else None
        )
        self._graph_state_bin_edges = None
        if self.state_mode == "graph_states":
            if self._store_price_grids is not None:
                _glo = float(self._store_price_grids.min())
                _ghi = float(self._store_price_grids.max())
            else:
                _glo = float(self.price_grid.min())
                _ghi = float(self.price_grid.max())
            self._graph_state_bin_edges = np.linspace(
                _glo, _ghi, self.graph_n_rival_bins + 1
            )
            logger.info(
                "graph_states: m=%d × B=%d^(k=%d) = %d states | rival EUR bins [%.2f, %.2f] | "
                "graph_rivals=%s",
                self.m, self.graph_n_rival_bins, self.graph_k,
                self.m * self.graph_n_rival_bins ** self.graph_k, _glo, _ghi,
                "set" if self.graph_rivals is not None else "MISSING(None)",
            )

    @property
    def state_size(self) -> int:
        if self.state_mode == "local_summary":
            return int(self.n_price_bins ** len(self._ls_channels))
        if self.state_mode == "design4_ownprice":
            return int(self.m * self.n_comp_bins)
        if self.state_mode == "design5_full":
            return int(self.m * self.n_comp_bins * self.n_comp_bins)
        if self.state_mode == "graph_states":
            return int(self.m * self.graph_n_rival_bins ** self.graph_k)
        if self.state_mode == "calvano_local":
            return int(self.m ** (self.calvano_k + 1))
        if self.state_mode == "strategic_hybrid":
            return int(self.m * self.n_comp_bins * 3)
        if self.state_mode == "hybrid_profit_gap":
            return int(self.m * self.hybrid_n_profit * self.hybrid_n_gap * self.hybrid_n_gap)
        return int(self._action_size ** self.k_neighbors)

    def decode_prices(self, price_idxs: np.ndarray) -> np.ndarray:
        """Convert (N,) price action indices to (N,) EUR prices.

        Uses chain-specific grids when available, global grid otherwise.
        This is the SINGLE canonical price-decoding path — all reporting,
        logging, and analysis code must call this instead of
        ``self.price_grid[pidx]``.

        Parameters
        ----------
        price_idxs : (N,) int array of price grid indices (one per store).

        Returns
        -------
        (N,) float64 array of EUR prices.
        """
        price_idxs = np.asarray(price_idxs, dtype=np.intp)
        if self._store_price_grids is not None:
            return self._store_price_grids[np.arange(len(self.firms)), price_idxs].astype(np.float64)
        return self.price_grid[price_idxs].astype(np.float64)

    def _init_local_summary_competitors(self) -> None:
        """Precompute competitor CSR and price-bin edges for local_summary mode."""
        N = len(self.firms)
        n_set = self.local_sum_n
        use_overlap = n_set is None or n_set == 0

        if use_overlap:
            catch_indptr = getattr(self.city, "catch_indptr", None)
            if catch_indptr is not None:
                mode_label = "demand_overlap"
                adj = np.zeros((N, N), dtype=bool)
                M = len(catch_indptr) - 1
                catch_indices = self.city.catch_indices
                for i in range(M):
                    start = int(catch_indptr[i])
                    end = int(catch_indptr[i + 1])
                    stores = catch_indices[start:end]
                    for a in stores:
                        for b in stores:
                            if a != b:
                                adj[int(a), int(b)] = True
            else:
                mode_label = "euclidean_nearest_10"
                logger.warning(
                    "local_summary demand-overlap requested but city.catch_indptr "
                    "is None; falling back to 10-nearest Euclidean competitors."
                )
                adj = self._euclidean_neighbor_adjacency(10)
        else:
            mode_label = f"euclidean_nearest_{int(n_set)}"
            # Euclidean distance between store locations (demand uses transit time).
            adj = self._euclidean_neighbor_adjacency(int(n_set))

        from scipy.sparse import csr_matrix

        needs_same_type = any(sn == "same_type" for sn, _ in self._ls_channels)

        comp_csr = csr_matrix(adj)
        self._comp_indptr = comp_csr.indptr.astype(np.int64)
        self._comp_indices = comp_csr.indices.astype(np.int64)
        self._price_bin_edges = np.linspace(
            self.price_grid.min(), self.price_grid.max(), self.n_price_bins + 1
        )

        if needs_same_type:
            ct = np.array(
                [getattr(f, "chain_type", "") for f in self.firms], dtype=object
            )
            same_adj = adj & (ct[:, None] == ct[None, :])
            same_csr = csr_matrix(same_adj)
            self._comp_indptr_same = same_csr.indptr.astype(np.int64)
            self._comp_indices_same = same_csr.indices.astype(np.int64)

        if self.local_summary_detailed:
            mean_all = float(comp_csr.sum(axis=1).mean()) if N > 0 else 0.0
            mean_same = float(same_csr.sum(axis=1).mean()) if N > 0 else 0.0
            logger.info(
                "local_summary_detailed: all=%.1f/store, same_type=%.1f/store, "
                "state_size=%d",
                mean_all,
                mean_same,
                self.state_size,
            )
        else:
            mean_deg = float(comp_csr.sum(axis=1).mean()) if N > 0 else 0.0
            logger.info(
                "local_summary competitors: mode=%s, mean=%.1f/store, state_size=%d",
                mode_label,
                mean_deg,
                self.state_size,
            )

    def _euclidean_neighbor_adjacency(self, n_nearest: int) -> np.ndarray:
        """Boolean (N, N) adjacency: each row has up to n_nearest competitors."""
        N = len(self.firms)
        adj = np.zeros((N, N), dtype=bool)
        if N <= 1:
            return adj
        locations = np.array([f.location for f in self.firms], dtype=np.float64)
        from scipy.spatial import cKDTree

        tree = cKDTree(locations)
        k_query = min(n_nearest + 1, N)
        _, idx = tree.query(locations, k=k_query)
        if k_query == 1:
            idx = idx.reshape(N, 1)
        for j in range(N):
            for nidx in idx[j]:
                nidx = int(nidx)
                if nidx != j:
                    adj[j, nidx] = True
        return adj

    def current_state_signal(self) -> np.ndarray:
        """Return the Q-learning state signal for all agents at the current actions."""
        if self.state_mode == "hybrid_profit_gap":
            return self._hybrid_state_signal()
        if self.state_mode == "design4_ownprice":
            return self._design4_state_signal()
        if self.state_mode == "design5_full":
            return self._design5_state_signal()
        if self.state_mode == "graph_states":
            return self._graph_state_signal()
        if self.state_mode == "calvano_local":
            return self._calvano_state_signal()
        if self.state_mode == "strategic_hybrid":
            return self._strategic_state_signal()
        if self.state_mode != "local_summary":
            return self.get_neighbor_actions_arr()
        pidx = self._current_joint_actions_arr // self.m_effort
        prices = self.decode_prices(pidx)
        B = self.n_price_bins
        bins = []
        for set_name, stat in self._ls_channels:
            if set_name == "all":
                indptr, indices = self._comp_indptr, self._comp_indices
            else:
                indptr, indices = self._comp_indptr_same, self._comp_indices_same
            mean_c, min_c = _local_price_summary(prices, indptr, indices)
            v = mean_c if stat == "mean" else min_c
            bins.append(
                np.clip(np.digitize(v, self._price_bin_edges) - 1, 0, B - 1)
            )
        mult = B ** np.arange(len(bins), dtype=np.int64)
        return (np.stack(bins, axis=1).astype(np.int64) * mult[None, :]).sum(axis=1)

    def _design4_state_signal(self) -> np.ndarray:
        """Compute Design 4 state: (own_price_idx, binned_same_type_competitor_mean).

        Returns (N,) int64 state indices in [0, m * n_comp_bins - 1].
        """
        from hotelling.env.competitors import weighted_mean_prices, bin_prices

        cs = self._comp_sets
        assert cs is not None, "CompetitorSets not initialised for design4_ownprice"

        b_own = (self._current_joint_actions_arr // self.m_effort).astype(np.int64)

        pidx = self._current_joint_actions_arr // self.m_effort
        N = len(self.firms)
        if self._store_price_grids is not None:
            prices = self._store_price_grids[np.arange(N), pidx].astype(np.float64)
        else:
            prices = self.price_grid[pidx].astype(np.float64)

        comp_mean = weighted_mean_prices(
            prices, cs.same_indptr, cs.same_indices, cs.same_weights,
        )
        b_comp = bin_prices(comp_mean, cs.bin_edges_same, cs.n_bins)

        return (b_own * self.n_comp_bins + b_comp).astype(np.int64)

    def _design5_state_signal(self) -> np.ndarray:
        """Design 5: (own_price, same_type_comp_mean, cross_type_comp_mean).

        Returns (N,) int64 in [0, m * B * B - 1].
        """
        from hotelling.env.competitors import weighted_mean_prices, bin_prices

        cs = self._comp_sets
        assert cs is not None, "CompetitorSets not initialised for design5_full"

        b_own = (self._current_joint_actions_arr // self.m_effort).astype(np.int64)
        pidx = self._current_joint_actions_arr // self.m_effort
        N = len(self.firms)
        if self._store_price_grids is not None:
            prices = self._store_price_grids[np.arange(N), pidx].astype(np.float64)
        else:
            prices = self.price_grid[pidx].astype(np.float64)

        B = self.n_comp_bins
        same_mean = weighted_mean_prices(prices, cs.same_indptr, cs.same_indices, cs.same_weights)
        cross_mean = weighted_mean_prices(prices, cs.cross_indptr, cs.cross_indices, cs.cross_weights)
        b_same = bin_prices(same_mean, cs.bin_edges_same, B)
        b_cross = bin_prices(cross_mean, cs.bin_edges_cross, B)

        return (b_own * (B * B) + b_same * B + b_cross).astype(np.int64)

    def _calvano_state_signal(self) -> np.ndarray:
        """Calvano local: (own_price, rival_1_price, ..., rival_k_price).

        Returns (N,) int64 in [0, m^(k+1) - 1].
        """
        from hotelling.env.competitors import rival_price_indices

        cs = self._comp_sets
        assert cs is not None and cs.nearest_same_type is not None, \
            "CompetitorSets not initialised for calvano_local"

        k = self.calvano_k
        all_pidx = (self._current_joint_actions_arr // self.m_effort).astype(np.int64)
        rivals = np.ascontiguousarray(cs.nearest_same_type[:, :k])  # (N, k) int64

        b_rivals = rival_price_indices(all_pidx, rivals, self.m)  # (N, k) int64

        # Mixed-radix: s = b_own * m^k + b_r1 * m^(k-1) + ... + b_rk
        m = self.m
        state = all_pidx.copy()
        for r in range(k):
            state = state * m + b_rivals[:, r]

        return state.astype(np.int64)

    def _graph_state_signal(self) -> np.ndarray:
        """graph_states: (own_price_idx, rival_1_bin, ..., rival_k_bin).

        Own price uses the full action grid (m levels); each rival's last price is
        re-projected onto the common EUR axis (graph_n_rival_bins bins). Absent rivals
        (-1) take the neutral middle bin. Returns (N,) int64 in [0, m·B^k − 1].
        Encoding (mixed radix, own most significant):
            s = b_own·B^k + b_r1·B^(k-1) + ... + b_rk
        """
        assert self.graph_rivals is not None, (
            "graph_states requires graph_rivals (set by the runner via rival_graph)."
        )
        N = len(self.firms)
        pidx = (self._current_joint_actions_arr // self.m_effort).astype(np.int64)
        b_own = np.clip(pidx, 0, self.m - 1)
        if self._store_price_grids is not None:
            prices = self._store_price_grids[np.arange(N), pidx].astype(np.float64)
        else:
            prices = self.price_grid[pidx].astype(np.float64)
        B = self.graph_n_rival_bins
        k = self.graph_k
        rivals = np.ascontiguousarray(self.graph_rivals[:, :k], dtype=np.int64)
        b_rivals = _gather_rival_price_bins(
            prices, rivals, self._graph_state_bin_edges, B
        )  # (N, k) int64
        state = b_own.copy()
        for r in range(k):
            state = state * B + b_rivals[:, r]
        return state.astype(np.int64)

    def _strategic_state_signal(self) -> np.ndarray:
        """Strategic hybrid: (own_price, same_type_comp_mean, market_regime).

        regime ∈ {0=competitive, 1=neutral, 2=supra-competitive}.
        Returns (N,) int64 in [0, m * B * 3 - 1].
        """
        from hotelling.env.competitors import weighted_mean_prices, bin_prices

        cs = self._comp_sets
        assert cs is not None, "CompetitorSets not initialised for strategic_hybrid"

        b_own = (self._current_joint_actions_arr // self.m_effort).astype(np.int64)
        pidx = self._current_joint_actions_arr // self.m_effort
        N = len(self.firms)
        if self._store_price_grids is not None:
            prices = self._store_price_grids[np.arange(N), pidx].astype(np.float64)
        else:
            prices = self.price_grid[pidx].astype(np.float64)

        B = self.n_comp_bins
        same_mean = weighted_mean_prices(prices, cs.same_indptr, cs.same_indices, cs.same_weights)
        b_comp = bin_prices(same_mean, cs.bin_edges_same, B)

        all_mean = weighted_mean_prices(prices, cs.all_indptr, cs.all_indices, cs.all_weights)

        regime = np.ones(N, dtype=np.int64)  # default: neutral
        if self._p_nash_arr is not None and self._p_mono_arr is not None:
            p_nash = self._p_nash_arr.astype(np.float64)
            p_mono = self._p_mono_arr.astype(np.float64)
            span = np.maximum(p_mono - p_nash, 1e-6)
            threshold = 0.15 * span
            regime = np.where(all_mean < p_nash - threshold, np.int64(0), regime)
            regime = np.where(all_mean > p_nash + threshold, np.int64(2), regime)
        else:
            logger.warning(
                "strategic_hybrid: p_nash_arr not available; regime always 'neutral'."
            )

        return (b_own * (B * 3) + b_comp * 3 + regime).astype(np.int64)

    def _hybrid_state_signal(self) -> np.ndarray:
        """Hybrid profit-gap state: (own_price, prev_profit_bin, gap1, gap2).

        Encoding (big-endian, price most significant):
          s = b_price*(n_profit*n_gap²) + b_profit*(n_gap²) + b_gap1*n_gap + b_gap2

        Dimensions
        ----------
        b_price ∈ [0, m−1]             own price bin = current action // m_effort
        b_profit ∈ [0, n_profit−1]     quantile bin of PREVIOUS period's profit
                                        (from self._prev_rewards_arr; 0 on first step)
        b_gap1 ∈ [0, n_gap−1]          gap to nearest same-type rival
        b_gap2 ∈ [0, n_gap−1]          gap to 2nd nearest same-type rival
                                        (neutral bin n_gap//2 if rival absent)

        gap = (p_own − p_rival) / p_rival   (percentual deviation)

        State size: m * n_profit * n_gap^2 = 7290 for defaults (18, 5, 9, 9).
        """
        from hotelling.env.competitors import compute_rival_gap_bins, bin_profits_by_chain

        assert self._comp_sets is not None and self._comp_sets.nearest_same_type is not None
        assert self._prev_rewards_arr is not None
        assert self._profit_quantile_thresholds is not None
        assert self._chain_type_ids is not None

        N = len(self.firms)

        # ── Own price bin ────────────────────────────────────────────────────
        pidx = self._current_joint_actions_arr // self.m_effort
        b_price = pidx.astype(np.int64)

        # ── EUR prices for gap computation ───────────────────────────────────
        if self._store_price_grids is not None:
            prices = self._store_price_grids[np.arange(N), pidx].astype(np.float64)
        else:
            prices = self.price_grid[pidx].astype(np.float64)

        # ── Rival price gap bins ─────────────────────────────────────────────
        # nearest_same_type is (N, k_max≥3) int64; take first two columns
        rivals_2 = np.ascontiguousarray(
            self._comp_sets.nearest_same_type[:, :2], dtype=np.int64
        )
        b_gap1, b_gap2 = compute_rival_gap_bins(
            prices, rivals_2, self._hybrid_gap_edges, self.hybrid_n_gap,
        )

        # ── Profit bins (previous period) ────────────────────────────────────
        b_profit = bin_profits_by_chain(
            self._prev_rewards_arr,
            self._chain_type_ids,
            self._profit_quantile_thresholds,
            self.hybrid_n_profit,
        )

        # ── Mixed-radix encoding ─────────────────────────────────────────────
        n_profit = self.hybrid_n_profit
        n_gap = self.hybrid_n_gap
        state = (
            b_price * np.int64(n_profit * n_gap * n_gap)
            + b_profit * np.int64(n_gap * n_gap)
            + b_gap1 * np.int64(n_gap)
            + b_gap2
        )
        return state.astype(np.int64)

    def update_prev_rewards_for_hybrid(self, rewards: np.ndarray) -> None:
        """Store the current period's rewards for use as t−1 profit in the next state.

        Timing contract (enforced by BatchSimulationEngine):
          1. env.step_array(actions)  → computes rewards_arr for period t
          2. env.current_state_signal()  → encodes s_{t+1} using _prev_rewards_arr (= r_{t−1})
          3. env.update_prev_rewards_for_hybrid(rewards_arr)  ← THIS CALL
          4. agent.act(s_{t+1})  → next step uses updated t profits

        Step 3 MUST happen AFTER step 2 and BEFORE the next step 1/2 cycle.
        No-op when state_mode != "hybrid_profit_gap" (_prev_rewards_arr is None).
        """
        if self._prev_rewards_arr is not None:
            self._prev_rewards_arr[:] = np.asarray(rewards, dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_neighbor_indices(self) -> dict[str, list[int]]:
        """Return k nearest neighbor indices (into self.firms) for each agent.

        Uses Euclidean distance between firm.location tuples (EPSG:3035 metres).
        A firm is never its own neighbor (self excluded).
        If there are fewer than k other firms, all other firms are used.

        Returns
        -------
        dict mapping agent_id str → list of k int indices into self.firms
        """
        n = len(self.firms)
        if n == 0:
            return {}

        # Build (n, 2) array of firm locations in metres
        locations = np.array([f.location for f in self.firms], dtype=np.float64)  # (n, 2)

        result: dict[str, list[int]] = {}
        for i, firm in enumerate(self.firms):
            # Euclidean distances from firm i to all others
            diffs = locations - locations[i]  # (n, 2)
            dists = np.hypot(diffs[:, 0], diffs[:, 1])  # (n,)
            dists[i] = np.inf  # exclude self

            k_actual = min(self.k_neighbors, n - 1)
            if k_actual == 0:
                neighbors = []
            else:
                neighbors = np.argsort(dists)[:k_actual].tolist()
            result[str(firm.id)] = neighbors

        return result

    def _build_neighbor_idx(self) -> np.ndarray:
        """Build ``(N, k)`` int32 neighbor index array from the dict representation.

        Padding positions (firms with fewer than k actual neighbors) contain the
        sentinel value ``N = len(self.firms)``.  In :meth:`step_array` a padded
        lookup array of length ``N+1`` is used so that ``padded[N]`` resolves to
        the mid-action value — replicating the dict-path padding behaviour exactly.

        Returns
        -------
        ndarray of shape (N, k_neighbors), dtype int32
        """
        n = len(self.firms)
        k = self.k_neighbors
        neighbor_idx = np.full((n, k), fill_value=n, dtype=np.int32)
        for i, firm in enumerate(self.firms):
            nbrs = self._neighbor_indices.get(str(firm.id), [])
            for j, nidx in enumerate(nbrs[:k]):
                neighbor_idx[i, j] = nidx
        return neighbor_idx

    def _build_observation(self, agent_id: str) -> dict:
        """Build the observation dict for one agent from current joint actions.

        Returns
        -------
        dict with:
            "own_prev_action"       : int — own last joint action index
            "neighbor_prev_actions" : list of int — k neighbors' last joint action indices
        """
        own = self._current_joint_actions[agent_id]
        neighbor_idxs = self._neighbor_indices.get(agent_id, [])
        neighbor_firm_ids = [str(self.firms[idx].id) for idx in neighbor_idxs]
        neighbor_actions = [
            self._current_joint_actions.get(nid, self._action_size // 2)
            for nid in neighbor_firm_ids
        ]
        # Pad with mid action if fewer neighbors than k (e.g. singleton market)
        while len(neighbor_actions) < self.k_neighbors:
            neighbor_actions.append(self._action_size // 2)
        return {
            "own_prev_action": own,
            "neighbor_prev_actions": neighbor_actions,
        }

    # ------------------------------------------------------------------
    # PettingZoo lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        """Reset the market to initial prices and return observations.

        Parameters
        ----------
        seed : optional int, currently unused (present for PettingZoo compat)

        Returns
        -------
        tuple (observations, infos) where observations is dict[agent_id → obs_dict]
        and infos is an empty dict.
        """
        mid_price_idx = self.m // 2
        mid_effort_idx = 0
        mid_joint = mid_price_idx * self.m_effort + mid_effort_idx

        self._current_joint_actions = {a: mid_joint for a in self.possible_agents}
        self._current_joint_actions_arr[:] = mid_joint
        # Reset hybrid profit buffer so new episodes start with "very bad" profit signal
        if self._prev_rewards_arr is not None:
            self._prev_rewards_arr[:] = 0.0
        self.agents = list(self.possible_agents)

        observations = {aid: self._build_observation(aid) for aid in self.agents}
        infos: dict = {}
        return observations, infos

    # ------------------------------------------------------------------
    # Fast array-only step (hot path)
    # ------------------------------------------------------------------

    def step_array(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute one period — fully vectorized, no dict construction.

        Parameters
        ----------
        actions : (N,) int64
            Joint action indices for all N firms in canonical order.

        Returns
        -------
        neighbor_actions : (N, k) int64
            Each firm's k neighbors' new joint action indices.  Firms with
            fewer than k actual neighbors receive the mid-action padding value
            (replicating :meth:`_build_observation` behaviour exactly).
        rewards : (N,) float64
            Per-firm profits this period.
        demands : (N,) float64
            Per-firm logit market shares this period.

        Notes
        -----
        *   Updates ``self._current_joint_actions_arr`` in-place.
        *   Does **not** update the dict ``_current_joint_actions``; call
            :meth:`step` (which wraps this method) for the dict API, or read
            the array directly for the batch path.
        *   Per-firm attribute arrays (qualities, mc, …) are taken from
            ``self._firm_arrays``, which is built once in ``__init__``.
        """
        N = len(self.firms)
        m_effort = self.m_effort
        m = self.m

        actions = np.asarray(actions, dtype=np.int64)

        # Decode and clamp joint action indices
        pidx = np.clip(actions // m_effort, 0, m - 1)
        eidx = np.clip(actions % m_effort, 0, m_effort - 1)

        # Re-encode (clamping may change the joint action value)
        self._current_joint_actions_arr[:] = pidx * m_effort + eidx

        # Decode prices and efforts — no Python loop
        if self._store_price_grids is not None:
            prices = self._store_price_grids[np.arange(N), pidx]
        else:
            prices = self.price_grid[pidx]
        efforts = self.effort_grid[eidx]

        # Market clearing: pre-built firm arrays, no per-call allocation
        demands, rewards = market_clearing_arrays(
            prices=prices,
            efforts=efforts,
            city=self.city,
            transport_cost=self.transport_cost,
            firm_arrays=self._firm_arrays,
        )

        # Neighbor actions via fancy indexing on a padded (N+1,) lookup array.
        # padded[N] = mid_action acts as the sentinel for padded neighbor slots.
        padded = np.empty(N + 1, dtype=np.int64)
        padded[:N] = self._current_joint_actions_arr
        padded[N] = self._action_size // 2  # padding value for sparse neighborhoods
        neighbor_actions = padded[self._neighbor_idx]  # (N, k) int64

        return neighbor_actions, rewards, demands

    def get_neighbor_actions_arr(self) -> np.ndarray:
        """Return the current ``(N, k)`` neighbor-action array without dict obs.

        Uses the same padded fancy-index logic as :meth:`step_array`.  Useful
        for initialising the batch engine after :meth:`reset` without going
        through ``_obs_dict_to_matrix``.

        Returns
        -------
        (N, k) int64 array of current neighbor joint action indices.
        """
        N = len(self.firms)
        padded = np.empty(N + 1, dtype=np.int64)
        padded[:N] = self._current_joint_actions_arr
        padded[N] = self._action_size // 2
        return padded[self._neighbor_idx].astype(np.int64)

    # ------------------------------------------------------------------
    # Dict-based step (slow path, implemented on top of step_array)
    # ------------------------------------------------------------------

    def step(self, actions: dict[str, int]) -> tuple[dict, dict, dict, dict, dict]:
        """Advance one period: decode actions, compute demand + profits, return.

        Parameters
        ----------
        actions : dict[agent_id str → int] — joint action indices chosen this period

        Returns
        -------
        5-tuple: (observations, rewards, terminations, truncations, infos)
        All dicts keyed by agent_id string.

        Notes
        -----
        Internally calls :meth:`step_array` for the numerical computation,
        then syncs ``_current_joint_actions`` dict and builds response dicts.
        Per-firm loops here are acceptable because this path is only used by
        the single-agent :class:`~hotelling.simulation.engine.SimulationEngine`.
        """
        N = len(self.firms)

        # Build actions array from dict, applying the same fallback as before
        actions_arr = np.empty(N, dtype=np.int64)
        for i, firm in enumerate(self.firms):
            aid = str(firm.id)
            actions_arr[i] = int(actions.get(aid, self._current_joint_actions[aid]))

        # Fast numerical path
        _neighbor_actions, profits, demands = self.step_array(actions_arr)

        # Sync dict from array (needed by _build_observation and SimulationEngine)
        for i, firm in enumerate(self.firms):
            self._current_joint_actions[str(firm.id)] = int(
                self._current_joint_actions_arr[i]
            )

        # Decode prices/efforts for the infos dict (vectorised, no extra loop)
        joint = self._current_joint_actions_arr
        pidx = joint // self.m_effort
        eidx = joint % self.m_effort
        if self._store_price_grids is not None:
            prices = self._store_price_grids[np.arange(N), pidx]
        else:
            prices = self.price_grid[pidx]
        efforts = self.effort_grid[eidx]

        observations = {aid: self._build_observation(aid) for aid in self.agents}
        rewards = {str(f.id): float(profits[i]) for i, f in enumerate(self.firms)}
        terminations = {aid: False for aid in self.agents}
        truncations = {aid: False for aid in self.agents}
        infos = {
            str(f.id): {
                "demand": float(demands[i]),
                "price": float(prices[i]),
                "effort": float(efforts[i]),
            }
            for i, f in enumerate(self.firms)
        }
        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Gymnasium-style spaces
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> Any:
        """Return the observation space for one agent.

        Returns a gymnasium.spaces.Dict if gymnasium is available,
        otherwise returns a plain dict describing the space.
        """
        try:
            import gymnasium
            from gymnasium.spaces import Dict, Discrete, MultiDiscrete

            joint_space = self.m * self.m_effort
            return Dict({
                "own_prev_action": Discrete(joint_space),
                "neighbor_prev_actions": MultiDiscrete([joint_space] * self.k_neighbors),
            })
        except ImportError:
            joint_space = self.m * self.m_effort
            return {
                "own_prev_action": {"type": "Discrete", "n": joint_space},
                "neighbor_prev_actions": {
                    "type": "MultiDiscrete",
                    "nvec": [joint_space] * self.k_neighbors,
                },
            }

    def action_space(self, agent: str) -> Any:
        """Return the action space for one agent.

        The action space is Discrete(m_price * m_effort): each integer encodes
        a joint (price_idx, effort_idx) pair via:
            joint_action_idx = price_idx * m_effort + effort_idx

        Returns a gymnasium.spaces.Discrete if gymnasium is available,
        otherwise returns a plain dict.
        """
        try:
            import gymnasium
            from gymnasium.spaces import Discrete

            return Discrete(self.m * self.m_effort)
        except ImportError:
            return {"type": "Discrete", "n": self.m * self.m_effort}
