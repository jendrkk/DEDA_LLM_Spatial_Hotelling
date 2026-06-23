"""Memory-mapped dense per-step simulation log (T × N).

Scaling controls
----------------
store_demand_profit : bool, default True
    If False, demand/profit memmaps are not allocated and ``write_step``
    silently ignores the ``demands`` / ``profits`` arguments.  These
    quantities are fully recomputable post-hoc from the stored
    ``price_idx`` / ``effort_idx`` arrays together with the price/effort
    grids and city geometry (via ``market_clearing`` on any subset of
    recorded steps), so omitting them saves roughly 50 % disk without
    any information loss.

float_dtype : {"float32", "float64"}, default "float32"
    NumPy dtype for demand/profit memmaps.  ``float32`` halves disk usage
    versus ``float64`` and is ample precision for all policy-analysis
    tasks.  Use ``"float64"`` only when asserting exact numerical values
    in tests.

dense_stride : int ≥ 1, default 1
    Record only every *dense_stride*-th simulation step (0-indexed).
    ``dense_stride=1`` records every step (current behaviour).
    ``dense_stride=1000`` on a 1M-step run writes 1 000 rows instead of
    1 000 000, reducing data by three orders of magnitude while still
    capturing the full pricing trajectory at coarse resolution.

dense_tail : int | None, default None
    Always densely record the last *dense_tail* simulation steps,
    regardless of ``dense_stride``.  Useful for capturing the converged
    (collusive) regime at full time-step resolution.  Steps that satisfy
    *both* the stride condition and the tail condition are stored once.

Pre-allocated rows
------------------
The set of scheduled recording steps is computed at construction time::

    scheduled = {t for t in range(0, T, dense_stride)}
    if dense_tail:
        scheduled |= {t for t in range(max(0, T - dense_tail), T)}

The memmaps are pre-allocated with exactly ``len(scheduled)`` rows.
If the simulation converges early, the trailing rows remain zero-filled
and ``rows_written`` in ``dense_log_meta.json`` reflects how many rows
were actually filled.

Disk layout (run_dir/)
----------------------
    price_idx.npy      — (R, N) int8    where R = len(scheduled)
    effort_idx.npy     — (R, N) int8
    demands.npy        — (R, N) float_dtype  (absent when store_demand_profit=False)
    profits.npy        — (R, N) float_dtype  (absent when store_demand_profit=False)
    steps.npy          — (R,) int64  actual simulation step for each row
    agent_ids.npy      — (N,) str
    price_grid.npy     — (m,) float32
    effort_grid.npy    — (m_effort,) float32
    dense_log_meta.json — metadata including all scaling parameters

Backward compatibility
----------------------
Logs written by earlier versions of DenseLog (without ``steps.npy`` and
without the new meta keys) are loaded transparently: the ``steps`` array
is reconstructed as ``np.arange(T_written)`` (stride=1 assumed), and
missing meta keys fall back to their defaults.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Warn if projected on-disk footprint exceeds this threshold.
_SIZE_WARN_GB: float = 5.0


class DenseLog:
    """Memory-mapped per-step simulation log: (R, N) arrays for all quantities.

    See module docstring for full description of scaling controls.

    Parameters
    ----------
    run_dir             : directory where all array files are written.
    T                   : total simulation steps (upper bound; actual run may
                          terminate earlier via convergence).
    N                   : number of stores / agents.
    agent_ids           : canonical agent ID strings in column order.
    price_grid          : (m,) price grid used by the Q-learners.
    effort_grid         : (m_effort,) effort grid.
    store_demand_profit : allocate/write demand and profit arrays (default True).
    float_dtype         : dtype for demand/profit ("float32" or "float64").
    dense_stride        : record every dense_stride-th step (default 1 = all steps).
    dense_tail          : always record the last dense_tail steps (default None).
    """

    def __init__(
        self,
        run_dir: Path,
        T: int,
        N: int,
        agent_ids: list[str],
        price_grid: np.ndarray,
        effort_grid: np.ndarray,
        *,
        store_price_grids: np.ndarray | None = None,
        store_demand_profit: bool = True,
        float_dtype: str = "float32",
        dense_stride: int = 1,
        dense_tail: int | None = None,
        store_effort: bool = True,
    ) -> None:
        if dense_stride < 1:
            raise ValueError(f"dense_stride must be >= 1, got {dense_stride}")

        self.run_dir = Path(run_dir)
        self.T = T
        self.N = N
        self.agent_ids = agent_ids
        self.price_grid  = np.asarray(price_grid,  dtype=np.float32)
        self.effort_grid = np.asarray(effort_grid, dtype=np.float32)
        # Per-store chain-specific grids: (N, m) float32 or None
        self.store_price_grids: np.ndarray | None = (
            np.asarray(store_price_grids, dtype=np.float32)
            if store_price_grids is not None
            else None
        )
        self._store_demand_profit = store_demand_profit
        self._float_dtype_str     = str(np.dtype(float_dtype))
        self._dense_stride        = dense_stride
        self._dense_tail          = dense_tail
        self._flush_every         = 10_000
        self._store_effort = store_effort
        # Backing store for effort_idx memmap; None when store_effort=False.
        # External access ALWAYS goes through the effort_idx property.
        self._effort_idx_mm: np.ndarray | None = None
        # Lazy-initialised zero array returned by the effort_idx property when
        # no memmap exists.  Allocated once on first property access.
        self._effort_idx_zeros: np.ndarray | None = None
        # City reference for post-hoc demand/profit reconstruction.
        # Populated via attach_city() (added in Step 2); None until then.
        self._city: object | None = None
        self._transport_cost: float | None = None
        self._firm_arrays: object | None = None

        # ── Compute scheduled recording steps ─────────────────────────────
        scheduled: set[int] = set(range(0, T, dense_stride))
        if dense_tail is not None and dense_tail > 0:
            scheduled |= set(range(max(0, T - dense_tail), T))
        self._recorded_steps = np.array(sorted(scheduled), dtype=np.int64)
        self._step_to_row: dict[int, int] = {
            int(s): int(r) for r, s in enumerate(self._recorded_steps)
        }
        n_rows = len(self._recorded_steps)
        self._rows_written = 0

        self.run_dir.mkdir(parents=True, exist_ok=True)

        # ── Size estimation ────────────────────────────────────────────────
        idx_bytes = n_rows * N * 1               # price_idx: always stored (int8)
        if store_effort:
            idx_bytes += n_rows * N * 1          # effort_idx: stored only when active
        dp_bytes  = 0
        if store_demand_profit:
            dp_bytes = 2 * n_rows * N * np.dtype(float_dtype).itemsize
        steps_bytes = n_rows * 8                 # int64
        total_gb = (idx_bytes + dp_bytes + steps_bytes) / (1024 ** 3)

        print(
            f"DenseLog: {n_rows:,} recorded rows × {N} stores | "
            f"stride={dense_stride}, tail={dense_tail}, "
            f"dp={store_demand_profit}, effort={store_effort}, dtype={float_dtype} → "
            f"~{total_gb:.3f} GB projected"
        )
        if total_gb > _SIZE_WARN_GB:
            logger.warning(
                "DenseLog projected on-disk size %.2f GB exceeds %.0f GB. "
                "Consider increasing dense_stride or setting "
                "store_demand_profit=False.",
                total_gb, _SIZE_WARN_GB,
            )

        # ── Allocate memmaps ───────────────────────────────────────────────
        self.price_idx = np.memmap(
            self.run_dir / "price_idx.npy",
            dtype="int8", mode="w+", shape=(n_rows, N),
        )
        if store_effort:
            self._effort_idx_mm = np.memmap(
                self.run_dir / "effort_idx.npy",
                dtype="int8", mode="w+", shape=(n_rows, N),
            )
        # else: self._effort_idx_mm remains None (set in Task 1.2);
        #       effort_idx property returns zeros on demand.

        if store_demand_profit:
            _np_dtype = np.dtype(float_dtype)
            self.demands = np.memmap(
                self.run_dir / "demands.npy",
                dtype=_np_dtype, mode="w+", shape=(n_rows, N),
            )
            self.profits = np.memmap(
                self.run_dir / "profits.npy",
                dtype=_np_dtype, mode="w+", shape=(n_rows, N),
            )
        else:
            self.demands = None
            self.profits = None

        # ── Persist grids and planned steps ───────────────────────────────
        np.save(self.run_dir / "agent_ids.npy",  np.array(agent_ids, dtype=str))
        np.save(self.run_dir / "price_grid.npy", self.price_grid)
        if store_effort:
            np.save(self.run_dir / "effort_grid.npy", self.effort_grid)
        if self.store_price_grids is not None:
            np.save(self.run_dir / "store_price_grids.npy", self.store_price_grids)
        np.save(self.run_dir / "steps.npy",       self._recorded_steps)

    # ------------------------------------------------------------------
    # effort_idx property — transparent zeros fallback
    # ------------------------------------------------------------------

    @property
    def effort_idx(self) -> np.ndarray:
        """(R, N) int8 effort grid indices.

        Returns the backing memmap when effort was stored (``store_effort=True``
        at construction or loaded from a run that has ``effort_idx.npy``).
        Returns a lazily-allocated zero-filled array of the same shape otherwise,
        so all read paths work identically regardless of whether effort was
        recorded.

        Write path: always use ``self._effort_idx_mm`` directly with a None
        guard — never write through this property.
        """
        if self._effort_idx_mm is None:
            if self._effort_idx_zeros is None:
                n = len(self._recorded_steps)
                self._effort_idx_zeros = np.zeros((n, self.N), dtype="int8")
            return self._effort_idx_zeros
        return self._effort_idx_mm

    # ------------------------------------------------------------------
    # Hot-path write
    # ------------------------------------------------------------------

    def write_step(
        self,
        t: int,
        price_idxs: np.ndarray,
        effort_idxs: np.ndarray,
        demands: np.ndarray,
        profits: np.ndarray,
    ) -> None:
        """Write simulation step *t* to its pre-assigned memmap row.

        Steps not in the recording schedule are silently skipped.  When
        ``store_demand_profit=False``, *demands* and *profits* are accepted
        but not stored.

        Parameters
        ----------
        t           : 0-based simulation step index.
        price_idxs  : (N,) int array of price-grid indices.
        effort_idxs : (N,) int array of effort-grid indices.
        demands     : (N,) float demand array (ignored if not storing).
        profits     : (N,) float profit array (ignored if not storing).
        """
        row = self._step_to_row.get(t)
        if row is None:
            return  # step not scheduled for recording

        self.price_idx[row] = price_idxs.astype("int8")
        if self._effort_idx_mm is not None:
            self._effort_idx_mm[row] = effort_idxs.astype("int8")
        if self.demands is not None:
            self.demands[row] = demands.astype(self._float_dtype_str)
        if self.profits is not None:
            self.profits[row] = profits.astype(self._float_dtype_str)

        self._rows_written = row + 1

        if row > 0 and row % self._flush_every == 0:
            self.flush()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush all memmap arrays and write updated ``dense_log_meta.json``."""
        self.price_idx.flush()
        if self._effort_idx_mm is not None:
            self._effort_idx_mm.flush()
        if self.demands is not None:
            self.demands.flush()
        if self.profits is not None:
            self.profits.flush()

        meta = {
            # ── Core dimensions ─────────────────────────────────────────
            "T_allocated":         self.T,
            "n_rows_allocated":    len(self._recorded_steps),
            "rows_written":        self._rows_written,
            "N":                   self.N,
            # ── Scaling parameters ──────────────────────────────────────
            "store_demand_profit": self._store_demand_profit,
            "store_effort":        self._store_effort,
            "float_dtype":         self._float_dtype_str,
            "dense_stride":        self._dense_stride,
            "dense_tail":          self._dense_tail,
            "has_store_price_grids": self.store_price_grids is not None,
            # ── Legacy alias (older loaders read T_written) ─────────────
            "T_written":           self._rows_written,
        }
        with (self.run_dir / "dense_log_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, run_dir: Path) -> "DenseLog":
        """Load an existing DenseLog from *run_dir* (read-only).

        Handles both new-format logs (with ``steps.npy`` and all meta keys)
        and legacy logs produced by older versions of DenseLog.
        """
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "dense_log_meta.json").read_text())

        T   = meta["T_allocated"]
        N   = meta["N"]

        # ── Scaling params with backward-compat defaults ─────────────────
        store_dp    = meta.get("store_demand_profit", True)
        store_effort = meta.get("store_effort", True)   # True = backward-compat default
        float_dtype = meta.get("float_dtype", "float32")
        dense_stride = meta.get("dense_stride", 1)
        dense_tail   = meta.get("dense_tail",   None)

        # rows_written: prefer new key, fall back to legacy T_written
        rows_written      = meta.get("rows_written", meta.get("T_written", T))
        n_rows_allocated  = meta.get("n_rows_allocated", T)

        # ── Reconstruct recorded-steps array ─────────────────────────────
        steps_path = run_dir / "steps.npy"
        if steps_path.exists():
            recorded_steps = np.load(steps_path)
        else:
            # Legacy log: steps are contiguous 0..rows_written-1
            recorded_steps = np.arange(rows_written, dtype=np.int64)

        # ── Load grids ───────────────────────────────────────────────────
        agent_ids   = list(np.load(run_dir / "agent_ids.npy"))
        price_grid  = np.load(run_dir / "price_grid.npy")
        _effort_grid_path = run_dir / "effort_grid.npy"
        effort_grid = (
            np.load(_effort_grid_path)
            if _effort_grid_path.exists()
            else np.zeros(1, dtype=np.float32)   # effort-off run: single level at 0.0
        )
        
        # ── Load per-store chain-specific grids (backward-compat: absent in older logs)
        spg_path = run_dir / "store_price_grids.npy"
        store_price_grids = np.load(spg_path) if spg_path.exists() else None

        obj = object.__new__(cls)
        obj.run_dir          = run_dir
        obj.T                = T
        obj.N                = N
        obj.agent_ids        = agent_ids
        obj.price_grid       = price_grid
        obj.effort_grid      = effort_grid
        obj.store_price_grids = store_price_grids
        obj._rows_written    = rows_written
        obj._flush_every     = 10_000
        obj._recorded_steps  = recorded_steps
        obj._step_to_row     = {
            int(s): int(r) for r, s in enumerate(recorded_steps)
        }
        obj._store_demand_profit = store_dp
        obj._float_dtype_str     = float_dtype
        obj._dense_stride        = dense_stride
        obj._dense_tail          = dense_tail
        obj._store_effort        = store_effort
        obj._effort_idx_zeros: np.ndarray | None = None
        # City reference for post-hoc reconstruction (populated via attach_city())
        obj._city                = None
        obj._transport_cost      = None
        obj._firm_arrays         = None

        # ── Memory-map the arrays (read-only) ────────────────────────────
        shape = (n_rows_allocated, N)
        obj.price_idx = np.memmap(
            run_dir / "price_idx.npy", dtype="int8", mode="r", shape=shape
        )
        _effort_path = run_dir / "effort_idx.npy"
        if store_effort and _effort_path.exists():
            obj._effort_idx_mm = np.memmap(
                _effort_path, dtype="int8", mode="r", shape=shape
            )
        else:
            # effort_idx.npy absent (store_effort=False run, or legacy log without
            # the file): effort_idx property returns zeros on demand.
            obj._effort_idx_mm = None

        if store_dp:
            obj.demands = np.memmap(
                run_dir / "demands.npy",
                dtype=float_dtype, mode="r", shape=shape,
            )
            obj.profits = np.memmap(
                run_dir / "profits.npy",
                dtype=float_dtype, mode="r", shape=shape,
            )
        else:
            obj.demands = None
            obj.profits = None

        return obj

    # ------------------------------------------------------------------
    # City binding for post-hoc demand/profit reconstruction
    # ------------------------------------------------------------------

    def attach_city(
        self,
        city: object,
        transport_cost: float,
        firm_arrays: object | None = None,
    ) -> None:
        """Bind a City object for on-demand demand/profit reconstruction.

        After this call, ``to_dataframe()`` will reconstruct demand and
        profit columns even when the log was created with
        ``store_demand_profit=False`` (lean run).

        Parameters
        ----------
        city : City
            Loaded spatial market container (must match the N stores in this
            log exactly).
        transport_cost : float
            Transport disutility coefficient used during the original run
            (read from the run's ``config.yaml`` or the env YAML).
        firm_arrays : FirmArrays | None
            Pre-built per-firm attribute struct from
            ``hotelling.core.market.precompute_firm_arrays``.  If None,
            it is built automatically from ``city.firms`` on first use.

        Notes
        -----
        This call is a no-op on logs that already store demands/profits
        (``self.demands is not None``): reconstruction is never triggered
        when the stored arrays exist.
        """
        from hotelling.core.market import precompute_firm_arrays

        self._city = city
        self._transport_cost = float(transport_cost)
        self._firm_arrays = (
            firm_arrays
            if firm_arrays is not None
            else precompute_firm_arrays(city.firms)
        )

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @property
    def recorded_steps(self) -> np.ndarray:
        """Actual simulation steps stored in this log (1-D int64 array).

        Length equals ``rows_written``: only rows that were actually written
        (i.e. the simulation reached that step) are included.
        """
        return self._recorded_steps[: self._rows_written]

    def _decode_prices_flat(
        self,
        pidx_flat: np.ndarray,
        agent_idx_flat: np.ndarray | None = None,
        single_agent: int | None = None,
    ) -> np.ndarray:
        """Decode flat price-index array to EUR prices using stored grids.

        Parameters
        ----------
        pidx_flat : (K,) int array of price grid indices.
        agent_idx_flat : (K,) int array mapping each element to a store index.
            Required when store_price_grids is set and single_agent is None.
        single_agent : if not None, all elements belong to this store index.

        Returns
        -------
        (K,) float32 array of EUR prices.
        """
        if self.store_price_grids is not None:
            if single_agent is not None:
                return self.store_price_grids[single_agent, pidx_flat]
            if agent_idx_flat is None:
                raise ValueError(
                    "agent_idx_flat required when store_price_grids is set"
                )
            return self.store_price_grids[agent_idx_flat, pidx_flat]
        return self.price_grid[pidx_flat]

    def to_dataframe(
        self,
        agent_idx: int | None = None,
        step_slice: slice | None = None,
        city: object | None = None,
        transport_cost: float | None = None,
        firm_arrays: object | None = None,
    ) -> "pd.DataFrame":
        """Return a slice of the log as a :class:`~pandas.DataFrame`.

        The ``period`` column contains the **actual simulation step** for each
        recorded row (not a sequential row index), correctly reflecting
        ``dense_stride`` and ``dense_tail``.

        Parameters
        ----------
        agent_idx  : if given, return only that store's columns (long format
                     still, one row per recorded step).  ``None`` (default)
                     returns all agents in long format.
        step_slice : a Python :class:`slice` applied to the *row* axis of the
                     written data.  ``slice(None)`` (default) returns all
                     written rows.  Example: ``slice(-500, None)`` returns
                     the last 500 written rows.
        city : City | None
            City object for on-demand demand/profit reconstruction.  Only
            needed when ``store_demand_profit=False`` (lean run) and demand/
            profit columns are required.  Overrides a city bound via
            :meth:`attach_city` for this call only.
        transport_cost : float | None
            Transport disutility coefficient.  Required together with ``city``
            when reconstructing.  Ignored when demands are stored.
        firm_arrays : FirmArrays | None
            Pre-built per-firm attribute struct.  If ``None`` and
            reconstruction is needed, built automatically from ``city.firms``.

        Notes
        -----
        **Reconstruction** occurs when ``self.demands is None`` AND a city is
        reachable (via ``city=`` kwarg or a prior :meth:`attach_city` call).
        Each row in the slice is decoded independently via
        ``market_clearing_arrays`` — this is O(rows_in_slice) numba JIT calls
        and is fast for analysis-scale slices (thousands of rows) but not
        intended for full-length 20 M-step slices.

        When ``store_demand_profit=False`` and no city is available, the
        ``demand`` and ``profit`` columns are omitted silently (with a warning).
        """
        import pandas as pd

        rw = self._rows_written
        steps = self._recorded_steps[:rw]   # (rw,) actual step numbers

        sl = step_slice if step_slice is not None else slice(None)

        # Normalise to 2-D arrays regardless of slice shape (safety net for
        # integer-index slices, though the public API only accepts slice objects)
        steps_sl  = steps[sl]
        pidx_rows = self.price_idx[:rw][sl]   # (..., N)
        eidx_rows = self.effort_idx[:rw][sl]  # (..., N) — zeros when not stored
        if pidx_rows.ndim == 1:
            pidx_rows = pidx_rows[np.newaxis, :]
            eidx_rows = eidx_rows[np.newaxis, :]

        # ── Resolve effective city / transport_cost / firm_arrays ─────────────
        _city_eff   = city          if city           is not None else self._city
        _tc_eff     = float(transport_cost) if transport_cost is not None else self._transport_cost
        _fa_eff     = firm_arrays   if firm_arrays    is not None else self._firm_arrays

        # ── Determine demand/profit source ────────────────────────────────────
        _demands_src: np.ndarray | None = None
        _profits_src: np.ndarray | None = None

        if self.demands is not None:
            # Normal (non-lean) run: read from stored memmaps
            _demands_src = self.demands[:rw][sl]
            _profits_src = self.profits[:rw][sl]
            if _demands_src.ndim == 1:
                _demands_src = _demands_src[np.newaxis, :]
                _profits_src = _profits_src[np.newaxis, :]

        elif _city_eff is not None and _tc_eff is not None:
            # Lean run with city available: reconstruct row-by-row
            from hotelling.core.market import market_clearing_arrays, precompute_firm_arrays
            if _fa_eff is None:
                _fa_eff = precompute_firm_arrays(_city_eff.firms)

            n_rows_sl = pidx_rows.shape[0]
            _demands_src = np.empty((n_rows_sl, self.N), dtype=np.float32)
            _profits_src = np.empty((n_rows_sl, self.N), dtype=np.float32)
            _arange_N    = np.arange(self.N)

            for r in range(n_rows_sl):
                pidx_r = pidx_rows[r].astype(np.intp)
                eidx_r = eidx_rows[r].astype(np.intp)

                if self.store_price_grids is not None:
                    prices_r = self.store_price_grids[_arange_N, pidx_r].astype(np.float64)
                else:
                    prices_r = self.price_grid[pidx_r].astype(np.float64)

                efforts_r = self.effort_grid[eidx_r].astype(np.float64)

                d, p = market_clearing_arrays(prices_r, efforts_r, _city_eff, _tc_eff, _fa_eff)
                _demands_src[r] = d.astype(np.float32)
                _profits_src[r] = p.astype(np.float32)

        else:
            # Lean run, no city: omit demand/profit columns with a warning
            if self.demands is None:
                logger.warning(
                    "DenseLog.to_dataframe: log has store_demand_profit=False and "
                    "no city is available for reconstruction. demand/profit columns "
                    "will be omitted. Call attach_city() or pass city=, transport_cost=."
                )

        # ── Single-agent path ─────────────────────────────────────────────────
        if agent_idx is not None:
            pidx = pidx_rows[:, agent_idx]
            eidx = eidx_rows[:, agent_idx]
            row: dict = {
                "period":     steps_sl,
                "agent_id":   self.agent_ids[agent_idx],
                "price_idx":  pidx,
                "effort_idx": eidx,
                "price":      self._decode_prices_flat(pidx, single_agent=agent_idx),
                "effort":     self.effort_grid[eidx],
            }
            if _demands_src is not None:
                row["demand"] = _demands_src[:, agent_idx]
                row["profit"] = _profits_src[:, agent_idx]
            return pd.DataFrame(row)

        # ── All-agents path (long format) ─────────────────────────────────────
        T_sl           = pidx_rows.shape[0]
        periods        = np.repeat(steps_sl, self.N)
        agent_col      = np.tile(self.agent_ids, T_sl)
        pidx_flat      = pidx_rows.ravel()
        eidx_flat      = eidx_rows.ravel()
        agent_idx_flat = np.tile(np.arange(self.N), T_sl)
        row = {
            "period":     periods,
            "agent_id":   agent_col,
            "price_idx":  pidx_flat,
            "effort_idx": eidx_flat,
            "price":      self._decode_prices_flat(pidx_flat, agent_idx_flat=agent_idx_flat),
            "effort":     self.effort_grid[eidx_flat],
        }
        if _demands_src is not None:
            row["demand"] = _demands_src.ravel()
            row["profit"] = _profits_src.ravel()
        return pd.DataFrame(row)
