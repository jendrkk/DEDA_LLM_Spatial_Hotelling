"""2-D city container with population grid and distance matrix.

Responsibility: hold spatial market geometry and the firms located in it.

Public API: City

Key dependencies: numpy, hotelling.core.firm

References: Anderson, de Palma, Thisse (1992) Ch.3; Tabuchi (1994).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from hotelling.core.firm import Firm


@dataclass
class City:
    """2-D spatial market container.

    Supports two distance representations which are mutually exclusive in use
    (though both fields may be populated for validation purposes):

    Dense path (inner-ring / small grids, dense_distances=True):
        ``dist2_km2`` is a dense (M, N) array of travel-time minutes.
        Consumed by the existing numba logit kernels and the equilibrium
        solvers (bertrand_nash / joint_monopoly).

    Sparse catchment path (full Berlin grid, dense_distances=False):
        ``dist2_km2`` is None.  The three CSR fields ``catch_indptr``,
        ``catch_indices``, ``catch_tt`` encode a per-cell sparse subset of
        the N stores, built by ``loader.build_catchment()``.  Consumed by
        the catchment-aware demand kernels introduced in Prompt 4.
        Equilibrium benchmarks on the sparse path require the Prompt-4
        catchment kernels (not yet implemented).

    Parameters
    ----------
    boundary: (xmin, ymin, xmax, ymax) in metres (EPSG:3035)
    population_grid: Optional 2-D population density array (H×W); None when
        cell_pop is used directly.
    firms: List of Firm objects currently in the market.
    dist2_km2: (M, N) dense travel-time matrix in minutes, or None on the
        full-grid / sparse path.  Still a required positional argument; pass
        None explicitly on the sparse path.
    cell_pop: (M,) residential consumer mass per cell.
    lambda_phi: (M,) footfall consumer mass per cell (λ * φ_i).
    pi_H: (M,) fraction of high-income residents per cell.
    pi_H_lambda_phi: (M,) fraction of high-income footfall per cell.
    alpha: (2,) income-type quality sensitivities [α_L, α_H].
    beta: effort sensitivity parameter.
    mu: logit scale parameter (default 0.25).
    a0: outside-option utility intercept (default 0.0).
    transport_exponent: exponent on travel-time disutility; 1.0 = linear
        (ADR-020).
    catch_indptr: (M+1,) int64 CSR row-pointer array; catch_indptr[i] and
        catch_indptr[i+1] delimit the slice of catch_indices / catch_tt
        belonging to cell i.  None on the dense path.
    catch_indices: (NNZ,) int32 store column indices for each non-zero
        entry in the catchment.  None on the dense path.
    catch_tt: (NNZ,) float64 travel times (minutes) for each catchment
        entry, in the same order as catch_indices.  None on the dense path.
    catch_C: (NNZ,) float64 precomputed transport disutility per catchment
        entry: catch_C[p] = -transport_cost * catch_tt[p]**transport_exponent.
        Eliminates pow and multiply from the hot loop.  Set by
        loader.populate_catchment_precompute(); None until then.
    catch_C_transport_cost: the transport_cost value baked into catch_C.
        Used to detect whether tc has changed between build and kernel call.
    A_quality: (2, N) float64 matrix: A_quality[h, j] = alpha_h * quality_j.
        Period-invariant quality contribution, split by income type.
    w_H, w_L: (M,) float64 combined consumer-mass weights per cell, per type.
    precompute_expweights: when True, catch_Kexp_L / catch_Kexp_H are
        populated and the expweights fast kernel is used instead of the
        stable log-sum-exp kernel.  Only activated when max exponent < 700.
    low_precision_storage: store Kexp arrays in float32 to halve memory.
    catch_Kexp_L, catch_Kexp_H: (NNZ,) exp-weight arrays per income type.
    """

    boundary: Tuple[float, float, float, float]
    population_grid: Optional[np.ndarray]
    firms: List[Firm]

    # Dense distance matrix — None on the sparse/full-grid path.
    dist2_km2: Optional[np.ndarray]   # (M, N) travel-time minutes, or None

    # Demand arrays (always populated)
    cell_pop: np.ndarray              # (M,)
    lambda_phi: np.ndarray            # (M,)
    pi_H: np.ndarray                  # (M,)
    pi_H_lambda_phi: np.ndarray       # (M,)

    # Model parameters
    alpha: np.ndarray                 # (2,) — [α_L, α_H]
    beta: float
    mu: float = 0.25
    a0: float = 0.0
    transport_exponent: float = 1.0   # exponent on travel-time; 1.0 = linear (ADR-020)

    # Sparse catchment CSR — None on the dense path.
    # Built by loader.build_catchment() for the full-grid sparse path.
    catch_indptr:  Optional[np.ndarray] = None   # (M+1,) int64
    catch_indices: Optional[np.ndarray] = None   # (NNZ,) int32 store col indices
    catch_tt:      Optional[np.ndarray] = None   # (NNZ,) float64 travel-time (min)

    # ── Period-invariant precomputed arrays for the catchment demand kernel ──
    # All set in one pass by loader.populate_catchment_precompute().
    # None until that function is called; the kernels accept None and fall
    # back to inline computation when these are absent.
    #
    # catch_C[p] = -transport_cost * catch_tt[p]**transport_exponent
    #   Bakes transport_cost into a per-CSR-entry float so the hot loop
    #   contains only a multiply-accumulate (no pow, no per-entry tc multiply).
    # catch_C_transport_cost: the tc value used when catch_C was built.
    #   Used in market_clearing_arrays to detect whether tc has changed.
    # A_quality[h, j] = alpha_h * quality_j  (shape 2 × N)
    #   The period-invariant quality term, split by income type.
    # w_H[i] / w_L[i]: combined consumer-mass weights per cell per type.
    #   w_H[i] = cell_pop[i]*pi_H[i] + lambda_phi[i]*pi_H_lambda_phi[i]
    #   w_L[i] = cell_pop[i]*(1-pi_H[i]) + lambda_phi[i]*(1-pi_H_lambda_phi[i])
    catch_C:                Optional[np.ndarray] = None  # (NNZ,) float64
    catch_C_transport_cost: Optional[float]      = None  # tc baked into catch_C
    A_quality:              Optional[np.ndarray] = None  # (2, N) float64
    w_H:                    Optional[np.ndarray] = None  # (M,) float64
    w_L:                    Optional[np.ndarray] = None  # (M,) float64

    # ── Optional expweights fast path ────────────────────────────────────────
    # When precompute_expweights=True, two CSR-aligned exp-weight arrays are
    # built at load time.  The hot-loop then needs only N exp() calls per
    # period (for w_g = exp(g * inv_mu)) rather than one per catchment entry.
    #
    # catch_Kexp_L[p] = exp((A_quality[0, indices[p]] + catch_C[p]) * inv_mu)
    # catch_Kexp_H[p] = exp((A_quality[1, indices[p]] + catch_C[p]) * inv_mu)
    #
    # Guard: both arrays are built only when max(A[h,idx]+C) * inv_mu < 700.
    # If the guard fires, precompute_expweights is left False and the stable
    # log-sum-exp kernel is used instead (a warning is logged).
    precompute_expweights:  bool                 = False
    low_precision_storage:  bool                 = False   # use float32 for Kexp
    catch_Kexp_L:           Optional[np.ndarray] = None    # (NNZ,) float64/32
    catch_Kexp_H:           Optional[np.ndarray] = None    # (NNZ,) float64/32

    @property
    def width(self) -> float:
        """Horizontal extent of the market space."""
        return self.boundary[2] - self.boundary[0]

    @property
    def height(self) -> float:
        """Vertical extent of the market space."""
        return self.boundary[3] - self.boundary[1]

    @property
    def center(self) -> Tuple[float, float]:
        """Geometric center of the city."""
        return (self.boundary[0] + self.width / 2, self.boundary[1] + self.height / 2)

    @property
    def area(self) -> float:
        """Area of the city."""
        return self.width * self.height

