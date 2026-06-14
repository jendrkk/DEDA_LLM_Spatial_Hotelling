"""Post-hoc store price heterogeneity diagnostics for completed Phase-0 runs."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from hotelling.simulation.dense_log import DenseLog

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_ID_COL_CANDIDATES = ("id", "store_id", "store_idx", "index")
_CHAIN_COL_CANDIDATES = ("chain_type", "type")
_SOCIAL_COL_CANDIDATES = ("esix_normalized", "si_normalized")


def _resolve_data_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else _REPO_ROOT / p


def _detect_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _converged_prices(dense_log: DenseLog, tail_frac: float) -> np.ndarray:
    rows_written = dense_log._rows_written
    tail_n = max(1, int(tail_frac * rows_written))
    price_idx_tail = dense_log.price_idx[:rows_written][-tail_n:]
    return dense_log.price_grid[price_idx_tail].mean(axis=0)


def _store_coordinates(stores: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "geometry" in stores.columns:
        return (
            stores.geometry.x.to_numpy(dtype=np.float64),
            stores.geometry.y.to_numpy(dtype=np.float64),
        )
    if "x" in stores.columns and "y" in stores.columns:
        return (
            stores["x"].to_numpy(dtype=np.float64),
            stores["y"].to_numpy(dtype=np.float64),
        )
    raise KeyError("Stores GeoDataFrame must have 'geometry' or 'x'/'y' columns.")


def _count_local_competitors(
    x: np.ndarray,
    y: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    coords = np.column_stack([x, y])
    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_point(coords, r=radius_m)
    return np.array(
        [len(neighbors) - 1 for neighbors in neighbor_lists],
        dtype=np.int64,
    )


def _attach_social_index(
    stores: gpd.GeoDataFrame,
    grid_path: Path,
) -> np.ndarray | None:
    try:
        grid = gpd.read_parquet(grid_path)
        social_col = _detect_column(grid, _SOCIAL_COL_CANDIDATES)
        if social_col is None or "geometry" not in grid.columns:
            return None

        stores_pts = stores.copy()
        if stores_pts.crs is None:
            stores_pts = stores_pts.set_crs("EPSG:3035")

        grid_gdf = grid[[social_col, "geometry"]].copy()
        if grid_gdf.crs is None:
            grid_gdf = grid_gdf.set_crs("EPSG:3035")

        joined = gpd.sjoin_nearest(
            stores_pts,
            grid_gdf,
            how="left",
        )
        if len(joined) > len(stores_pts):
            joined = joined[~joined.index.duplicated(keep="first")]
        return joined.sort_index()[social_col].to_numpy(dtype=np.float64)
    except Exception as exc:
        logger.debug("Social-index nearest join failed: %s", exc)
        return None


def _variance_decomposition(
    prices: np.ndarray,
    chain_types: np.ndarray,
) -> dict[str, float]:
    grand_mean = float(prices.mean())
    total_var = float(prices.var())
    between = 0.0
    within = 0.0
    n_total = len(prices)

    for chain_type in np.unique(chain_types):
        mask = chain_types == chain_type
        n_k = int(mask.sum())
        group = prices[mask]
        group_mean = float(group.mean())
        group_var = float(group.var()) if n_k > 1 else 0.0
        between += n_k * (group_mean - grand_mean) ** 2
        within += n_k * group_var

    between /= n_total
    within /= n_total
    ratio = between / total_var if total_var > 0.0 else float("nan")
    return {
        "price_between_var": between,
        "price_within_var": within,
        "price_between_total_ratio": ratio,
    }


def _ols_regression(
    y: np.ndarray,
    n_competitors: np.ndarray,
    chain_types: np.ndarray,
    social_index: np.ndarray | None,
) -> tuple[dict[str, float], float]:
    design_cols = [np.ones(len(y)), n_competitors.astype(np.float64)]
    coef_names = ["intercept", "n_local_competitors"]

    design_cols.append((chain_types == "standard").astype(np.float64))
    coef_names.append("dummy_standard")
    design_cols.append((chain_types == "bio").astype(np.float64))
    coef_names.append("dummy_bio")

    if social_index is not None:
        si = social_index.astype(np.float64).copy()
        if not np.all(np.isfinite(si)):
            median_si = float(np.nanmedian(si))
            si[~np.isfinite(si)] = median_si
        design_cols.append(si)
        coef_names.append("social_index")

    x_mat = np.column_stack(design_cols)
    coef, _, _, _ = np.linalg.lstsq(x_mat, y.astype(np.float64), rcond=None)
    fitted = x_mat @ coef
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    coefficients = {name: float(value) for name, value in zip(coef_names, coef)}
    return coefficients, r2


def store_price_heterogeneity(
    run_dir: str | Path,
    radius_m: float = 800.0,
    tail_frac: float = 0.10,
) -> dict:
    """Diagnose converged price heterogeneity for one Phase-0 run directory."""
    run_dir = Path(run_dir)
    dense_log = DenseLog.load(run_dir)
    converged_price = _converged_prices(dense_log, tail_frac)

    config = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    env_cfg = config.get("env", {})
    stores_path = _resolve_data_path(
        env_cfg.get("stores_path", "data/processed/supermarkets.parquet")
    )
    grid_path = _resolve_data_path(
        env_cfg.get("grid_path", "data/processed/demand_grid.parquet")
    )

    stores_raw = gpd.read_parquet(stores_path)
    id_col = _detect_column(stores_raw, _ID_COL_CANDIDATES)
    chain_col = _detect_column(stores_raw, _CHAIN_COL_CANDIDATES)
    if chain_col is None:
        raise KeyError(
            "Stores parquet must contain a chain-type column "
            f"({', '.join(_CHAIN_COL_CANDIDATES)})."
        )

    if id_col is None:
        stores = stores_raw.reset_index(drop=True)
        store_ids = [str(i) for i in range(len(stores))]
    else:
        stores = stores_raw.sort_values(id_col, ascending=True).reset_index(drop=True)
        store_ids = stores[id_col].astype(str).tolist()

    agent_ids = [str(a) for a in dense_log.agent_ids]
    n_stores = len(stores)
    n_agents = dense_log.N
    if n_stores != n_agents:
        raise ValueError(
            f"Store count ({n_stores}) does not match DenseLog N ({n_agents})."
        )

    agent_id_set = set(agent_ids)
    store_id_set = set(store_ids)
    if agent_id_set.issubset(store_id_set):
        id_to_price = dict(zip(agent_ids, converged_price))
        aligned_price = np.array(
            [id_to_price[sid] for sid in store_ids],
            dtype=np.float64,
        )
    else:
        logger.warning(
            "Positional store alignment: agent_ids do not all match store IDs; "
            "assuming canonical ascending loader order (index 0..N-1)."
        )
        if not all(agent_ids[i] == store_ids[i] for i in range(n_agents)):
            logger.warning(
                "agent_ids and store IDs differ positionally; still using "
                "column order from DenseLog.agent_ids."
            )
        aligned_price = converged_price.astype(np.float64)

    x, y = _store_coordinates(stores)
    n_local_competitors = _count_local_competitors(x, y, radius_m)
    social_index = _attach_social_index(stores, grid_path)
    chain_types = stores[chain_col].fillna("standard").astype(str).to_numpy()

    price_mean = float(aligned_price.mean())
    price_std = float(aligned_price.std())
    price_cv = price_std / price_mean if price_mean != 0.0 else float("nan")

    variance_stats = _variance_decomposition(aligned_price, chain_types)
    regression_coefs, regression_r2 = _ols_regression(
        aligned_price,
        n_local_competitors,
        chain_types,
        social_index,
    )

    per_store = pd.DataFrame(
        {
            "store_id": store_ids,
            "chain_type": chain_types,
            "converged_price": aligned_price,
            "n_local_competitors": n_local_competitors,
            "social_index": social_index if social_index is not None else np.nan,
        }
    )

    return {
        "n_stores": n_agents,
        "radius_m": float(radius_m),
        "tail_frac": float(tail_frac),
        "price_mean": price_mean,
        "price_std": price_std,
        "price_cv": price_cv,
        **variance_stats,
        "regression_coefficients": regression_coefs,
        "regression_r2": regression_r2,
        "social_index_available": social_index is not None,
        "per_store": per_store,
    }
