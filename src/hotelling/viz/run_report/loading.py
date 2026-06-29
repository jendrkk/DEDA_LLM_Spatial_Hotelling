"""Run loading + reconstruction for the run-report visualisation pipeline.

:class:`RunBundle` is the single source of truth a plotting module needs: it
rebuilds the :class:`~hotelling.core.city.City` with the *exact* env kwargs the
run used (so the sparse-catchment truncation — and therefore the benchmark
cache key — matches), loads the :class:`~hotelling.simulation.dense_log.DenseLog`,
recomputes per-store Bertrand-Nash / joint-monopoly benchmarks, reloads the
demand-grid and store geometries in canonical order, and exposes
chain-specific-grid-aware decoders for prices / demands / profits at any
recorded step.

Why not reuse ``hotelling.viz.spatial_map.load_run``?
    * It rebuilds the city without ``catchment_k_min/k_max``,
      ``precompute_expweights`` or ``low_precision_storage`` — a different
      catchment truncation than the run, which both desynchronises demand and
      misses the benchmark cache.
    * Its ``prices_efforts_at`` decodes via the *global* ``price_grid`` only,
      silently producing wrong prices for ``--chs-grid`` runs.
RunBundle fixes both.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# repo root: src/hotelling/viz/run_report/loading.py -> parents[4]
_REPO_ROOT: Path = Path(__file__).resolve().parents[4]

CHAIN_TYPES = ("discount", "standard", "bio")


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else _REPO_ROOT / p


@dataclass
class RunBundle:
    run_dir: Path
    run_cfg: Dict[str, Any]
    env_cfg: Dict[str, Any]
    agent_cfg: Dict[str, Any]
    meta: Dict[str, Any]

    city: Any
    firms: list
    dense_log: Any
    N: int
    M: int
    transport_cost: float
    with_effort: bool
    lean: bool
    is_strategic: bool

    # geometry (canonical order: row j == firm j; cell i == inside-mass row i)
    grid_gdf_3857: Any
    grid_gdf_3035: Any
    stores_gdf_3857: Any
    cell_centroids_3035: np.ndarray   # (M, 2) metres

    # chain bookkeeping
    chain_types: np.ndarray           # (N,) object
    chains: np.ndarray                # (N,) object (brand)
    marginal_costs: np.ndarray        # (N,) float64
    type_masks: Dict[str, np.ndarray] # 'discount'/'standard'/'bio'/'global' -> (N,) bool

    # grids
    price_grid: np.ndarray            # (m,) global grid
    store_price_grids: Optional[np.ndarray]  # (N, m) chain-specific or None
    price_grid_xi: float

    # benchmarks (per store)
    p_nash: np.ndarray
    p_mono: np.ndarray
    e_bench: np.ndarray
    demand_nash: np.ndarray
    demand_mono: np.ndarray
    profit_nash: np.ndarray
    profit_mono: np.ndarray

    # frames / time
    recorded_steps: np.ndarray        # (R,) absolute sim steps for each written row
    frame_rows: np.ndarray            # (F,) DenseLog row indices to animate
    step_spacing: int                 # median spacing between recorded steps

    # windowed converged per-store state
    learned_prices: np.ndarray
    learned_demands: np.ndarray
    learned_profits: np.ndarray

    # optional artefacts
    graph_rivals: Optional[np.ndarray] = None   # (N, k) int64, -1 padded

    # private caches
    _qual: np.ndarray = field(default=None, repr=False)
    _fa: Any = field(default=None, repr=False)

    # analysis-row subsampling (populated by _init_analysis; see AnalysisCfg)
    analysis_rows: np.ndarray = field(default=None, repr=False)
    analysis_steps: np.ndarray = field(default=None, repr=False)
    analysis_step_spacing: int = 1
    analysis_prices: np.ndarray = field(default=None, repr=False)
    _analysis_demands: Optional[np.ndarray] = field(default=None, repr=False)
    _analysis_profits: Optional[np.ndarray] = field(default=None, repr=False)

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_run_dir(cls, run_dir: str | Path, viz_cfg) -> "RunBundle":
        import geopandas as gpd
        import yaml

        from hotelling.core.market import precompute_firm_arrays
        from hotelling.simulation.dense_log import DenseLog

        run_dir = Path(run_dir)
        if not (run_dir / "config.yaml").exists():
            raise FileNotFoundError(f"No config.yaml in {run_dir}")
        run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
        meta_path = run_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        env_cfg = run_cfg.get("env", {}) or {}
        agent_cfg = run_cfg.get("agents", {}) or {}

        # mu / a0 key normalisation (saved runs already map these; be defensive)
        mu = float(env_cfg.get("mu", env_cfg.get("logit_scale", 0.25)))
        a0 = float(env_cfg.get("a0", env_cfg.get("outside_option", -1.0)))
        tc = float(env_cfg.get("transport_cost", 0.01))

        is_strategic = (
            str(meta.get("mode", "")) == "strategic"
            or "strategic_runs" in str(run_dir).replace("\\", "/")
            or (run_dir / "envelopes.parquet").exists()
        )
        with_effort = bool(run_cfg.get("with_effort", False)) or \
            int(agent_cfg.get("m_effort", 1)) > 1

        # lambda parity: reproduce the runner's auto-calibration if a placeholder slipped through
        lambda_val = float(env_cfg.get("lambda_val", 0.0))
        if lambda_val == 1500.0:
            try:
                from hotelling.spatial.assembly import calibrate_lambda
                from hotelling.spatial.loader import _compute_phi_i
                grid = gpd.read_parquet(_resolve(env_cfg.get(
                    "grid_path", "data/processed/demand_grid.parquet")))
                if "phi_i" not in grid.columns:
                    grid = grid.copy()
                    grid["phi_i"] = _compute_phi_i(grid).values
                lambda_val = float(calibrate_lambda(grid, target_footfall_share=0.125))
                logger.info("Reproduced auto-calibrated lambda=%.4f.", lambda_val)
            except Exception as exc:  # noqa: BLE001
                logger.warning("lambda auto-calibration failed (%s); using 1500.", exc)

        # ── rebuild City with FULL env parity (matches runner.run_single_session) ──
        from hotelling.spatial.loader import load_berlin_city
        catchment_minutes = env_cfg.get("catchment_minutes", None)
        city, firms = load_berlin_city(
            grid_path=_resolve(env_cfg.get("grid_path", "data/processed/demand_grid.parquet")),
            stores_path=_resolve(env_cfg.get("stores_path", "data/processed/supermarkets.parquet")),
            travel_times_path=_resolve(env_cfg.get("travel_times_path", "data/processed/travel_times.parquet")),
            lambda_val=lambda_val,
            q_S=float(env_cfg.get("q_S", 0.8)), q_B=float(env_cfg.get("q_B", 1.5)),
            alpha_L=float(env_cfg.get("alpha_L", 0.5)), alpha_H=float(env_cfg.get("alpha_H", 1.5)),
            beta_effort=float(env_cfg.get("beta_effort", 0.001)),
            kappa0=float(env_cfg.get("kappa0", 1.0)),
            store_size=float(env_cfg.get("store_size", 600.0)),
            transport_cost=tc, a0=a0, mu=mu,
            nan_fill_minutes=float(env_cfg.get("nan_fill_minutes", 120.0)),
            marginal_cost_D=float(env_cfg.get("marginal_cost_D", 0.0)),
            marginal_cost_S=float(env_cfg.get("marginal_cost_S", 0.0)),
            marginal_cost_B=float(env_cfg.get("marginal_cost_B", 0.0)),
            rent_scale=float(env_cfg.get("rent_scale", 0.0)),
            rent_normalization=str(env_cfg.get("rent_normalization", "mean_ratio")),
            dense_distances=bool(env_cfg.get("dense_distances", True)),
            catchment_minutes=(float(catchment_minutes) if catchment_minutes is not None else None),
            catchment_k_min=int(env_cfg.get("catchment_k_min", 12)),
            catchment_k_max=int(env_cfg.get("catchment_k_max", 80)),
            precompute_expweights=bool(env_cfg.get("precompute_expweights", False)),
            low_precision_storage=bool(env_cfg.get("low_precision_storage", False)),
        )
        N = len(firms)
        M = int(city.cell_pop.shape[0])

        # ── DenseLog + reconstruction binding ──────────────────────────────────
        dense_log = DenseLog.load(run_dir)
        dense_log.attach_city(city, transport_cost=tc)
        lean = dense_log.demands is None

        # ── benchmarks (cache-keyed; matches the run because the city matches) ──
        from hotelling.core.equilibrium import bertrand_nash, joint_monopoly
        from hotelling.core.market import market_clearing_arrays
        cache = _resolve(env_cfg.get("grid_path", "data/processed/demand_grid.parquet")).parent / "benchmarks_cache.npz"
        p_nash, e_nash = bertrand_nash(city, transport_cost=tc, cache_path=cache)
        p_mono, _ = joint_monopoly(
            city, transport_cost=tc, cache_path=cache,
            effort_fixed=(e_nash if with_effort else None),
        )
        e_bench = (np.asarray(e_nash, dtype=np.float64) if with_effort
                   else np.zeros(N, dtype=np.float64))
        fa = precompute_firm_arrays(firms)
        d_nash, _ = market_clearing_arrays(p_nash, e_bench, city, tc, fa)
        d_mono, _ = market_clearing_arrays(p_mono, e_bench, city, tc, fa)
        costs = np.array([f.marginal_cost for f in firms], dtype=np.float64)
        profit_nash = (p_nash - costs) * d_nash
        profit_mono = (p_mono - costs) * d_mono

        # ── geometry (canonical order identical to loader) ─────────────────────
        grid_raw = gpd.read_parquet(_resolve(env_cfg.get("grid_path", "data/processed/demand_grid.parquet")))
        grid_3035 = (grid_raw.sort_values("GITTER_ID_100m")
                     .drop_duplicates(subset="GITTER_ID_100m", keep="first")
                     .reset_index(drop=True))
        stores_3035 = gpd.read_parquet(_resolve(env_cfg.get("stores_path", "data/processed/supermarkets.parquet"))).reset_index(drop=True)
        grid_3857 = grid_3035.to_crs(epsg=3857)
        stores_3857 = stores_3035.to_crs(epsg=3857)
        cents = grid_3035.geometry.centroid
        cell_centroids_3035 = np.column_stack([cents.x.values, cents.y.values]).astype(np.float64)

        # ── chain bookkeeping (firms are authoritative) ────────────────────────
        chain_types = np.array([str(getattr(f, "chain_type", "standard")) for f in firms], dtype=object)
        chains = np.array([str(getattr(f, "chain", "") or f.id) for f in firms], dtype=object)
        type_masks = {ct: (chain_types == ct) for ct in CHAIN_TYPES}
        type_masks["global"] = np.ones(N, dtype=bool)

        # ── grids ──────────────────────────────────────────────────────────────
        price_grid = np.asarray(dense_log.price_grid, dtype=np.float64)
        store_price_grids = (np.asarray(dense_log.store_price_grids, dtype=np.float64)
                             if dense_log.store_price_grids is not None else None)
        price_grid_xi = float(agent_cfg.get("price_grid_xi", 0.1))

        # ── frames / time axis ─────────────────────────────────────────────────
        recorded_steps = np.asarray(dense_log.recorded_steps, dtype=np.int64)
        R = len(recorded_steps)
        if R == 0:
            raise ValueError(f"DenseLog in {run_dir} has zero written rows.")
        step_spacing = int(np.median(np.diff(recorded_steps))) if R > 1 else 1
        step_spacing = max(step_spacing, 1)
        frame_rows = cls._select_frames(R, viz_cfg)

        # ── graph rivals (graph_states baseline runs) ──────────────────────────
        graph_rivals = None
        gr_path = run_dir / "graph_rivals.npy"
        if gr_path.exists():
            graph_rivals = np.load(gr_path)

        obj = cls(
            run_dir=run_dir, run_cfg=run_cfg, env_cfg=env_cfg, agent_cfg=agent_cfg, meta=meta,
            city=city, firms=firms, dense_log=dense_log, N=N, M=M, transport_cost=tc,
            with_effort=with_effort, lean=lean, is_strategic=is_strategic,
            grid_gdf_3857=grid_3857, grid_gdf_3035=grid_3035, stores_gdf_3857=stores_3857,
            cell_centroids_3035=cell_centroids_3035,
            chain_types=chain_types, chains=chains, marginal_costs=costs, type_masks=type_masks,
            price_grid=price_grid, store_price_grids=store_price_grids, price_grid_xi=price_grid_xi,
            p_nash=p_nash, p_mono=p_mono, e_bench=e_bench,
            demand_nash=d_nash, demand_mono=d_mono, profit_nash=profit_nash, profit_mono=profit_mono,
            recorded_steps=recorded_steps, frame_rows=frame_rows, step_spacing=step_spacing,
            learned_prices=np.zeros(N), learned_demands=np.zeros(N), learned_profits=np.zeros(N),
            graph_rivals=graph_rivals,
        )
        obj._qual = np.array([f.quality for f in firms], dtype=np.float64)
        obj._fa = fa
        obj._init_analysis(viz_cfg)
        obj._compute_window(viz_cfg)
        return obj

    # ── frame selection ──────────────────────────────────────────────────────

    @staticmethod
    def _select_frames(R: int, viz_cfg) -> np.ndarray:
        fc = viz_cfg.frames
        if fc.mode == "all":
            rows = np.arange(R)
            if R > fc.max_frames:
                rows = np.unique(np.linspace(0, R - 1, fc.max_frames).round().astype(int))
            return rows
        n = min(int(fc.n_frames), R)
        return np.unique(np.linspace(0, R - 1, n).round().astype(int))

    # ── decoders ──────────────────────────────────────────────────────────────

    def decode_prices_rows(self, rows: np.ndarray) -> np.ndarray:
        """(len(rows), N) EUR prices, chain-specific-grid aware."""
        P = np.asarray(self.dense_log.price_idx[rows], dtype=np.intp)
        if P.ndim == 1:
            P = P[None, :]
        if self.store_price_grids is not None:
            return self.store_price_grids[np.arange(self.N)[None, :], P]
        return self.price_grid[P]

    def prices_at(self, t: int) -> np.ndarray:
        return self.decode_prices_rows(np.array([t]))[0]

    def efforts_at(self, t: int) -> np.ndarray:
        eidx = np.asarray(self.dense_log.effort_idx[t], dtype=np.intp)
        return self.dense_log.effort_grid[eidx].astype(np.float64)

    def demands_profits_at(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.dense_log.demands is not None:
            return (self.dense_log.demands[t].astype(np.float64),
                    self.dense_log.profits[t].astype(np.float64))
        from hotelling.core.market import market_clearing_arrays
        d, p = market_clearing_arrays(self.prices_at(t), self.efforts_at(t),
                                      self.city, self.transport_cost, self._fa)
        return d.astype(np.float64), p.astype(np.float64)

    def inside_mass_static(self, prices: np.ndarray, efforts: np.ndarray) -> np.ndarray:
        """(M, N) per-cell consumer allocation at the given config (path-aware)."""
        from hotelling.core.market import catchment_cell_mass, cell_choice_mass
        if self.city.catch_indptr is not None:
            inside, _ = catchment_cell_mass(self.city, prices, efforts, self.transport_cost)
        else:
            inside, _ = cell_choice_mass(
                prices=prices, efforts=efforts, dist2_km2=self.city.dist2_km2,
                cell_pop=self.city.cell_pop, lambda_phi=self.city.lambda_phi,
                pi_H=self.city.pi_H, pi_H_lambda_phi=self.city.pi_H_lambda_phi,
                alpha=self.city.alpha, quality=self._qual, beta=self.city.beta,
                transport_cost=self.transport_cost, mu=self.city.mu, a0=self.city.a0,
                transport_exponent=getattr(self.city, "transport_exponent", 1.0),
            )
        return inside

    def inside_mass_at(self, t: int) -> np.ndarray:
        return self.inside_mass_static(self.prices_at(t), self.efforts_at(t))

    # ── windowed converged per-store state ─────────────────────────────────────

    def _compute_window(self, viz_cfg) -> None:
        wc = viz_cfg.window
        last_step = int(self.recorded_steps[-1])
        if wc.tail_steps is not None:
            cutoff = last_step - int(wc.tail_steps)
        else:
            total = last_step - int(self.recorded_steps[0])
            cutoff = last_step - wc.tail_fraction * total
        tail_rows = np.where(self.recorded_steps >= cutoff)[0]
        if tail_rows.size == 0:
            tail_rows = np.array([len(self.recorded_steps) - 1])

        self.learned_prices = self.decode_prices_rows(tail_rows).mean(axis=0)
        if self.dense_log.demands is not None:
            self.learned_demands = self.dense_log.demands[tail_rows].astype(np.float64).mean(axis=0)
            self.learned_profits = self.dense_log.profits[tail_rows].astype(np.float64).mean(axis=0)
        else:
            from hotelling.core.market import market_clearing_arrays
            eff = self.efforts_at(int(tail_rows[-1]))
            d, p = market_clearing_arrays(self.learned_prices, eff, self.city,
                                          self.transport_cost, self._fa)
            self.learned_demands, self.learned_profits = d, p

    # ── per-chain-type benchmark aggregates ────────────────────────────────────

    def chain_nash_mono_prices(self, ct: str) -> Tuple[float, float]:
        """(mean p^N_tau, mean p^M_tau) over the chain type's stores."""
        m = self.type_masks[ct]
        return float(self.p_nash[m].mean()), float(self.p_mono[m].mean())

    def all_steps(self) -> np.ndarray:
        return self.recorded_steps

    # ── analysis-row subsampling & cached reconstruction ──────────────────────

    def _init_analysis(self, viz_cfg) -> None:
        """Pick the analysis rows (stride / auto-stride) and decode prices eagerly.

        Prices are cheap (index gather) so they are materialised now.
        Demands/profits are reconstructed lazily on first use via
        :meth:`get_analysis_demands` / :meth:`get_analysis_profits`, so a
        price-only report never pays the spatial market-clearing cost.
        """
        ac = viz_cfg.analysis
        R = len(self.recorded_steps)
        if getattr(ac, "auto_stride", True):
            stride = max(1, int(round(ac.target_steps_per_point / max(self.step_spacing, 1))))
        else:
            stride = max(1, int(ac.stride))
        rows = np.arange(0, R, stride)
        if rows.size == 0 or rows[-1] != R - 1:
            rows = np.append(rows, R - 1)
        rows = np.unique(rows)
        if rows.size > int(ac.max_points):
            rows = np.unique(np.linspace(0, R - 1, int(ac.max_points)).round().astype(int))
        self.analysis_rows = rows
        self.analysis_steps = self.recorded_steps[rows]
        self.analysis_step_spacing = (int(np.median(np.diff(self.analysis_steps)))
                                      if rows.size > 1 else self.step_spacing)
        self.analysis_step_spacing = max(self.analysis_step_spacing, 1)
        self.analysis_prices = self.decode_prices_rows(rows)
        self._analysis_demands = None
        self._analysis_profits = None
        logger.info("Analysis sampling: %d of %d recorded rows "
                    "(target ~%d steps/point, effective spacing %d steps).",
                    rows.size, R, getattr(ac, "target_steps_per_point", 0),
                    self.analysis_step_spacing)

    def _ensure_analysis_dp(self) -> None:
        """Populate the cached (A, N) demands & profits over the analysis rows once."""
        if self._analysis_demands is not None and self._analysis_profits is not None:
            return
        rows = self.analysis_rows
        A = len(rows)
        if self.dense_log.demands is not None:
            # Non-lean: slice the memmaps once (no reconstruction).
            self._analysis_demands = np.asarray(self.dense_log.demands[rows], dtype=np.float64)
            self._analysis_profits = np.asarray(self.dense_log.profits[rows], dtype=np.float64)
            return
        # Lean: reconstruct via the spatial kernel once over the analysis rows.
        from hotelling.core.market import market_clearing_arrays
        D = np.empty((A, self.N), dtype=np.float64)
        P = np.empty((A, self.N), dtype=np.float64)
        log_every = max(1, A // 10)
        for i, t in enumerate(rows):
            prices = self.analysis_prices[i]
            efforts = self.efforts_at(int(t))
            d, p = market_clearing_arrays(prices, efforts, self.city,
                                          self.transport_cost, self._fa)
            D[i] = d
            P[i] = p
            if i % log_every == 0:
                logger.info("  reconstructing demands/profits %d/%d ...", i, A)
        self._analysis_demands = D
        self._analysis_profits = P

    def get_analysis_demands(self) -> np.ndarray:
        """(A, N) demands over the analysis rows (reconstructed once, cached)."""
        self._ensure_analysis_dp()
        return self._analysis_demands

    def get_analysis_profits(self) -> np.ndarray:
        """(A, N) gross profits over the analysis rows (reconstructed once, cached)."""
        self._ensure_analysis_dp()
        return self._analysis_profits
