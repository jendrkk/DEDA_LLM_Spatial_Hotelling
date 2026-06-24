"""Reciprocal rival-graph construction for the ``--graph-states`` Q-learning mode.

Builds an undirected, degree-bounded *observation graph* over stores so that each store
observes the price of the rivals that compete most strongly for its demand. Because the
graph is undirected, reciprocity is structural (A observes B ⟺ B observes A), and a
max-degree-``k`` subgraph is exactly a disjoint union of paths and cycles: a remote triad
becomes a 3-cycle, a remote duopoly a single edge, and a true local monopolist an isolated
vertex (degree 0 → its rival state slots stay neutral → it learns its monopoly best response).

Edge weight = symmetric average diversion ratio evaluated at Bertrand-Nash:

    M[a,b] = Σ_{i,h} w_ih s_ia s_ib            (symmetric demand-weighted co-choice mass
                                                = μ·∂q_a/∂p_b = μ·∂q_b/∂p_a)
    E[a]   = Σ_{i,h} w_ih s_ia (1 - s_ia)      (own demand semi-elasticity)
    D_{a←b} = M[a,b] / E[b]                     (Conlon-Mortimer diversion ratio)
    W[a,b] = ½ ( D_{a←b} + D_{b←a} ) = ½ M[a,b] (1/E[a] + 1/E[b])   (symmetric)

The 1/E normalisation up-weights remote/isolated rivalries (small E) and down-weights dense
urban rivalries (large E) — i.e. it prioritises exactly the clusters where local collusion is
plausible. The degree-≤k subgraph maximising total weight is a max-weight b-matching, solved
exactly as a binary ILP (scipy.optimize.milp; greedy fallback).

Public API
----------
compute_competition_matrix(city, prices, efforts) -> (M, E)
diversion_edge_weights(M, E)                       -> W
build_rival_graph(W, k, *, match_mode, chain_types, candidate_topn, min_edge_weight) -> RivalGraph

References: Anderson, de Palma & Thisse (1992); Conlon & Mortimer (2018, diversion ratios).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numba as nb
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RivalGraph:
    """Result of reciprocal rival-graph construction.

    Attributes
    ----------
    rivals : (N, k) int64
        Per-store matched rival store indices, sorted by edge weight descending,
        padded with -1 where a store has fewer than k matched rivals. This is the
        array consumed by the env's ``graph_states`` signal.
    matched_edges : (E_m, 2) int64
        Undirected matched pairs (u < v) selected by the b-matching ("optimal" edges).
    matched_weights : (E_m,) float64
        Edge weights of ``matched_edges`` (same order).
    candidate_edges : (E_c, 2) int64
        Undirected candidate pairs (u < v) considered by the matcher (top-N per node,
        post chain-type/threshold filtering) — drawn faintly on the map.
    candidate_weights : (E_c,) float64
        Edge weights of ``candidate_edges`` (same order).
    degree : (N,) int64
        Matched degree per store (0 = local monopolist).
    k : int
        Max degree (= graph_k).
    """

    rivals: np.ndarray
    matched_edges: np.ndarray
    matched_weights: np.ndarray
    candidate_edges: np.ndarray
    candidate_weights: np.ndarray
    degree: np.ndarray
    k: int


@nb.njit(cache=True, fastmath=True)
def _cochoice_elasticity_catchment(
    g, A, a0_scaled, inv_mu, w_L, w_H, indptr, indices, catch_C, N
):
    """Serial Numba kernel: co-choice mass M (N×N) and own elasticity E (N,) at state ``g``.

    Mirrors the utility computation and log-sum-exp stabilisation of
    ``_catchment_demand_and_elasticity_jit`` but accumulates pairwise share products.
    Serial (writes a dense N×N matrix); this is a one-time precompute, O(Σ_i k_i²).

    g = beta*efforts - prices  (N,)
    """
    Mcells = len(indptr) - 1
    M = np.zeros((N, N))
    E = np.zeros(N)
    for i in range(Mcells):
        start = indptr[i]
        end = indptr[i + 1]
        k = int(end - start)
        if k == 0:
            continue
        ev = np.empty(k)
        for h in range(2):
            w_h = w_L[i] if h == 0 else w_H[i]
            if w_h == 0.0:
                continue
            # Pass 1 — vmax
            vmax = a0_scaled
            for q in range(k):
                p = start + q
                j = indices[p]
                v = (A[h, j] + g[j] + catch_C[p]) * inv_mu
                if v > vmax:
                    vmax = v
            # Pass 2 — exp values, denominator
            denom = np.exp(a0_scaled - vmax)
            for q in range(k):
                p = start + q
                j = indices[p]
                v = (A[h, j] + g[j] + catch_C[p]) * inv_mu
                ev[q] = np.exp(v - vmax)
                denom += ev[q]
            inv_d = 1.0 / denom
            # Accumulate elasticity and pairwise co-choice mass
            for qa in range(k):
                ja = indices[start + qa]
                sa = ev[qa] * inv_d
                E[ja] += w_h * sa * (1.0 - sa)
                for qb in range(qa + 1, k):
                    jb = indices[start + qb]
                    sb = ev[qb] * inv_d
                    val = w_h * sa * sb
                    M[ja, jb] += val
                    M[jb, ja] += val
    return M, E


def compute_competition_matrix(city, prices, efforts):
    """Compute (M, E) at the given prices/efforts over the city's sparse catchment.

    Parameters
    ----------
    city : City with a populated sparse catchment (``catch_indptr is not None``).
    prices, efforts : (N,) float64 — typically the Bertrand-Nash prices and zeros.

    Returns
    -------
    M : (N, N) float64 — symmetric co-choice mass.
    E : (N,)   float64 — own demand semi-elasticity.
    """
    if city.catch_indptr is None:
        raise ValueError(
            "compute_competition_matrix requires a sparse catchment "
            "(city.catch_indptr is None). Set catchment_minutes in the env config."
        )
    from hotelling.core.market import _ensure_catchment_precompute

    tc = (
        city.catch_C_transport_cost
        if city.catch_C_transport_cost is not None
        else 0.0
    )
    _ensure_catchment_precompute(city, tc)

    N = len(city.firms)
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    efforts = np.ascontiguousarray(efforts, dtype=np.float64)
    g = city.beta * efforts - prices
    inv_mu = 1.0 / float(city.mu)
    a0_scaled = float(city.a0) * inv_mu
    indptr = city.catch_indptr.astype(np.int64, copy=False)
    indices = city.catch_indices.astype(np.int32, copy=False)
    catch_C = np.ascontiguousarray(city.catch_C, dtype=np.float64)
    A = np.ascontiguousarray(city.A_quality, dtype=np.float64)
    w_L = np.ascontiguousarray(city.w_L, dtype=np.float64)
    w_H = np.ascontiguousarray(city.w_H, dtype=np.float64)

    M, E = _cochoice_elasticity_catchment(
        g, A, a0_scaled, inv_mu, w_L, w_H, indptr, indices, catch_C, N
    )
    logger.info(
        "Competition matrix: N=%d, nnz(M)=%d (%.1f%%), E in [%.3g, %.3g].",
        N, int((M > 0).sum()), 100.0 * (M > 0).mean(),
        float(E.min()), float(E.max()),
    )
    return M, E


def diversion_edge_weights(M: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Symmetric average diversion weight  W[a,b] = ½ M[a,b] (1/E[a] + 1/E[b])."""
    inv_e = np.where(E > 1e-12, 1.0 / E, 0.0)
    W = 0.5 * M * (inv_e[:, None] + inv_e[None, :])
    np.fill_diagonal(W, 0.0)
    return W


def _candidate_edges(W, topn, match_mode, chain_types, min_edge_weight):
    """Undirected candidate edge set: top-``topn`` neighbours per node, filtered.

    Returns (u, v, w) with u < v, deduplicated. Pruning to top-N per node keeps the
    subsequent ILP sparse and the map readable; since the optimal degree-≤k matching only
    ever uses each node's strongest edges, pruning with topn ≥ a few × k is near-lossless.
    """
    N = W.shape[0]
    ct = np.asarray(chain_types, dtype=object) if chain_types is not None else None
    floor = max(float(min_edge_weight), 0.0)
    pairs: set[tuple[int, int]] = set()
    for i in range(N):
        row = W[i].copy()
        row[i] = -np.inf
        if match_mode == "SC" and ct is not None:
            row[ct != ct[i]] = -np.inf
        t = min(int(topn), N - 1)
        if t <= 0:
            continue
        # indices of the t largest weights for node i
        cand = np.argpartition(-row, t - 1)[:t] if t < N else np.arange(N)
        for j in cand:
            j = int(j)
            wij = row[j]
            if np.isfinite(wij) and wij > floor:
                a, b = (i, j) if i < j else (j, i)
                if a != b:
                    pairs.add((a, b))
    if not pairs:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    arr = np.array(sorted(pairs), dtype=np.int64)
    u, v = arr[:, 0], arr[:, 1]
    w = W[u, v].astype(np.float64)
    return u, v, w


def _greedy_b_matching(u, v, w, N, k):
    """Greedy max-degree-k matching: take edges by descending weight if both ends have a free slot."""
    order = np.argsort(-w)
    deg = np.zeros(N, dtype=np.int64)
    sel = np.zeros(len(w), dtype=bool)
    for e in order:
        a, b = int(u[e]), int(v[e])
        if deg[a] < k and deg[b] < k:
            sel[e] = True
            deg[a] += 1
            deg[b] += 1
    return sel


def _solve_b_matching(u, v, w, N, k):
    """Exact max-weight degree-≤k b-matching via scipy.optimize.milp; greedy fallback.

    ILP: maximise Σ_e w_e x_e  s.t.  Σ_{e ∋ v} x_e ≤ k ∀v,  x_e ∈ {0,1}.
    """
    E = len(w)
    if E == 0:
        return np.zeros(0, dtype=bool)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        rows = np.concatenate([u, v])
        cols = np.concatenate([np.arange(E), np.arange(E)])
        data = np.ones(2 * E, dtype=np.float64)
        A = coo_matrix((data, (rows, cols)), shape=(N, E)).tocsr()
        constr = LinearConstraint(A, lb=0.0, ub=float(k) * np.ones(N))
        res = milp(
            c=-np.asarray(w, dtype=np.float64),
            constraints=[constr],
            integrality=np.ones(E),
            bounds=Bounds(0, 1),
            options={"time_limit": 120.0},
        )
        if getattr(res, "success", False) and res.x is not None:
            return np.asarray(res.x) > 0.5
        logger.warning(
            "milp b-matching did not succeed (%s); falling back to greedy.",
            getattr(res, "message", "no message"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("milp unavailable/failed (%s); using greedy b-matching.", exc)
    return _greedy_b_matching(u, v, w, N, k)


def build_rival_graph(
    W: np.ndarray,
    k: int,
    *,
    match_mode: str = "A",
    chain_types=None,
    candidate_topn: int = 8,
    min_edge_weight: float = 0.0,
) -> RivalGraph:
    """Build the undirected, degree-≤k reciprocal observation graph from edge weights W.

    Parameters
    ----------
    W : (N, N) float64 symmetric edge weights (zero diagonal).
    k : int — max degree per store (= graph_k).
    match_mode : {"A","SC"} — "SC" restricts edges to same chain type.
    chain_types : (N,) sequence of chain-type strings (required when match_mode == "SC").
    candidate_topn : per-node candidate cap before matching (≥ k recommended).
    min_edge_weight : drop candidate edges with weight ≤ this floor.

    Returns
    -------
    RivalGraph
    """
    N = W.shape[0]
    if match_mode == "SC" and chain_types is None:
        raise ValueError("match_mode='SC' requires chain_types.")
    topn = max(int(candidate_topn), int(k))
    u, v, w = _candidate_edges(W, topn, match_mode, chain_types, min_edge_weight)
    sel = _solve_b_matching(u, v, w, N, int(k))
    mu_, mv_, mw_ = u[sel], v[sel], w[sel]

    rivals = np.full((N, int(k)), -1, dtype=np.int64)
    degree = np.zeros(N, dtype=np.int64)
    nbrs: dict[int, list[tuple[float, int]]] = {i: [] for i in range(N)}
    for e in range(len(mu_)):
        a, b, wt = int(mu_[e]), int(mv_[e]), float(mw_[e])
        nbrs[a].append((wt, b))
        nbrs[b].append((wt, a))
    for i in range(N):
        lst = sorted(nbrs[i], key=lambda t: -t[0])[: int(k)]
        degree[i] = len(lst)
        for r, (_wt, nb_idx) in enumerate(lst):
            rivals[i, r] = nb_idx

    n_iso = int((degree == 0).sum())
    logger.info(
        "Rival graph (mode=%s, k=%d): %d candidate edges → %d matched edges; "
        "degree mean=%.2f max=%d; %d isolated stores (local monopolists).",
        match_mode, k, len(w), len(mu_),
        float(degree.mean()), int(degree.max()), n_iso,
    )
    return RivalGraph(
        rivals=rivals,
        matched_edges=np.stack([mu_, mv_], axis=1).astype(np.int64)
        if len(mu_) else np.empty((0, 2), dtype=np.int64),
        matched_weights=mw_.astype(np.float64),
        candidate_edges=np.stack([u, v], axis=1).astype(np.int64)
        if len(u) else np.empty((0, 2), dtype=np.int64),
        candidate_weights=w.astype(np.float64),
        degree=degree,
        k=int(k),
    )
