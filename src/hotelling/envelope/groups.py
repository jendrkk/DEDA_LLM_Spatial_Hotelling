"""Group division abstract base class and store-group assignment logic.

Responsibility: define the GroupDivision interface, maintain the division
registry, and assign composite group labels to stores at Phase 0 init.

Public API: GroupDivision, REGISTRY, assign_groups

Key dependencies: abc

References: ADR-009; docs/agent_simulation_technical_report.md §4.
"""
from __future__ import annotations

import itertools

import numpy as np
from abc import ABC, abstractmethod
from typing import ClassVar

REGISTRY: dict[str, type[GroupDivision]] = {}


class GroupDivision(ABC):
    """Abstract base for group-division strategies.

    Each concrete division classifies a store into one of exactly two
    categories based on static store metadata assigned at Phase 0 init.
    Register concrete subclasses in REGISTRY using the class's ``name`` key.
    """

    name: ClassVar[str]
    categories: ClassVar[tuple[str, str]]

    def __init__(self, **params) -> None:
        """Concrete divisions read thresholds etc. from ``params`` (config-driven)."""
        self.params = params

    @abstractmethod
    def assign(self, store_metadata: dict) -> str:
        """Return the category label for a single store."""
        raise NotImplementedError

    @abstractmethod
    def description(self) -> str:
        """Human-readable description included in the CEO system prompt."""
        raise NotImplementedError


def _instantiate_divisions(
    active_divisions: list[str],
    division_params: dict | None = None,
) -> list[GroupDivision]:
    """Instantiate active divisions from REGISTRY with their config params."""
    # Importing the divisions package populates REGISTRY via its submodules.
    import hotelling.envelope.divisions  # noqa: F401  (registration side effect)

    if len(active_divisions) > 2:
        raise ValueError(
            f"At most 2 simultaneous divisions allowed (ADR-009); got {active_divisions}"
        )
    division_params = division_params or {}
    out: list[GroupDivision] = []
    for name in active_divisions:
        if name not in REGISTRY:
            raise KeyError(f"Unknown division {name!r}; registered: {list(REGISTRY)}")
        out.append(REGISTRY[name](**division_params))
    return out


def composite_group_keys(
    active_divisions: list[str],
    division_params: dict | None = None,
) -> list[str]:
    """Canonical group-key vocabulary, matching the CEO prompt's group structure.

    0 divisions  -> ["default"]
    1 division   -> list(categories)            e.g. ["HEAVY", "EASY"]
    2 divisions  -> Cartesian product joined "_" e.g. ["HEAVY_RICH","HEAVY_POOR",...]
    Order is divisions in ``active_divisions`` order, categories in declared order.
    """
    if not active_divisions:
        return ["default"]
    divs = _instantiate_divisions(active_divisions, division_params)
    cats = [list(d.categories) for d in divs]
    return ["_".join(combo) for combo in itertools.product(*cats)]


def assign_groups(
    stores_metadata: list[dict],
    active_divisions: list[str],
    division_params: dict | None = None,
) -> dict[str, str]:
    """Assign a fixed composite group label to every store (Phase 0 init).

    Parameters
    ----------
    stores_metadata : list of per-store dicts (canonical store order); each must
        contain at least ``store_id`` plus the features the active divisions need
        (``n_rivals_within_R``, ``social_index``). Build via build_store_metadata.
    active_divisions : registry keys of active divisions (<= 2).
    division_params : thresholds/radius from the groups config.

    Returns
    -------
    dict store_id -> composite label (e.g. "HEAVY_RICH"); all "default" when no
    divisions are active.
    """
    if not active_divisions:
        return {str(md["store_id"]): "default" for md in stores_metadata}
    divs = _instantiate_divisions(active_divisions, division_params)
    labels: dict[str, str] = {}
    for md in stores_metadata:
        parts = [d.assign(md) for d in divs]
        labels[str(md["store_id"])] = "_".join(parts)
    return labels


def build_store_metadata(
    firms: list,
    grid_gdf=None,
    radius_m: float = 500.0,
) -> list[dict]:
    """Build per-store metadata used by group divisions (canonical store order).

    Features
    --------
    store_id            : str (firm.id)
    chain               : brand string (firm.chain)
    chain_type          : "discount" | "standard" | "bio"
    location            : (x, y) EPSG:3035 metres
    n_rivals_within_R   : count of OTHER stores within ``radius_m`` (KDTree, self excluded)
    social_index        : neighbourhood social-status index S_r in [0,1] for the
                          store's location (nearest demand-grid cell). 0.5 if
                          grid_gdf is None or has no usable social column.

    Parameters
    ----------
    firms : list[Firm] in canonical order (index j == firm.id "j").
    grid_gdf : the demand grid GeoDataFrame (EPSG:3035) with a per-cell social
        column ("pi_H_res" preferred, else "esix_normalized", else "si_normalized")
        and point/polygon geometry. If None, social_index defaults to 0.5.
    radius_m : metres for the rival-count feature (DIVISION_COMPETITION radius).
    """
    from scipy.spatial import cKDTree

    locs = np.array([f.location for f in firms], dtype=np.float64)  # (N, 2)
    n_rivals = np.zeros(len(firms), dtype=np.int64)
    if len(firms) > 1:
        tree = cKDTree(locs)
        # query_ball_point returns self too -> subtract 1
        counts = tree.query_ball_point(locs, r=float(radius_m), return_length=True)
        n_rivals = np.maximum(counts - 1, 0).astype(np.int64)

    social = np.full(len(firms), 0.5, dtype=np.float64)
    if grid_gdf is not None:
        col = next(
            (c for c in ("pi_H_res", "esix_normalized", "si_normalized")
             if c in grid_gdf.columns),
            None,
        )
        if col is not None:
            try:
                import geopandas as gpd
                from shapely.geometry import Point

                pts = gpd.GeoDataFrame(
                    {"store_id": [f.id for f in firms]},
                    geometry=[Point(xy) for xy in locs],
                    crs=grid_gdf.crs,
                )
                g = grid_gdf[[col, "geometry"]].copy()
                # Use cell centroids for a robust nearest join (cells may be polygons).
                g["geometry"] = g.geometry.centroid
                joined = gpd.sjoin_nearest(pts, g, how="left")
                joined = joined.drop_duplicates(subset="store_id").reset_index(drop=True)
                social = joined[col].fillna(0.5).to_numpy(dtype=np.float64)
            except Exception:  # noqa: BLE001 — diagnostic feature; never block the run
                social = np.full(len(firms), 0.5, dtype=np.float64)

    return [
        {
            "store_id": str(f.id),
            "chain": f.chain,
            "chain_type": str(f.chain_type),
            "location": tuple(f.location),
            "n_rivals_within_R": int(n_rivals[j]),
            "social_index": float(social[j]),
        }
        for j, f in enumerate(firms)
    ]
