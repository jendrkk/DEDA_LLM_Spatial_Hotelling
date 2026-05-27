"""Final simulation-grid assembly: merge all spatial layers into one GeoDataFrame.

This module provides functions that combine the population grid, LOR attributes,
POI data, and socio-economic layers into the single GeoDataFrame consumed by the
simulation engine.

All inputs must be in EPSG:3035.

Key dependencies: geopandas, pandas (optional ``[spatial]`` extra).
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "add_lcc_layer",
    "add_lor_attributes",
    "add_poi_layer",
    "assemble_simulation_grid",
    "build_demand_grid",
    "enrich_supermarkets_with_brw",
    "normalize_social_indices",
]


def add_lor_attributes(
    grid: gpd.GeoDataFrame,
    lor: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Attach LOR planning-area attributes to each grid cell.

    Performs a spatial join of *grid* cell centroids against *lor* polygons
    and attaches ``PLR_ID``, ``PLR_NAME``, and any additional LOR columns
    (e.g. population-density scores from :func:`~hotelling.spatial.admin.refine_shapes_selection`)
    to *grid*.

    Parameters
    ----------
    grid:
        Population grid GeoDataFrame with polygon geometry in EPSG:3035.
    lor:
        LOR GeoDataFrame in EPSG:3035 with at minimum ``PLR_ID`` and
        ``PLR_NAME`` columns.

    Returns
    -------
    geopandas.GeoDataFrame
        *grid* with ``PLR_ID`` and ``PLR_NAME`` columns added.  Cells that
        do not fall within any LOR polygon receive ``NaN``.
    """
    if grid.empty or lor.empty:
        return grid.copy()

    out = grid.copy()
    # Compute centroid of each grid cell polygon for point-in-polygon join
    centroids = gpd.GeoDataFrame(
        {"_grid_idx": out.index},
        geometry=out.geometry.centroid,
        crs=out.crs,
    )
    if centroids.crs != lor.crs:
        centroids = centroids.to_crs(lor.crs)

    lor_attr_cols = [c for c in lor.columns if c != "geometry"]
    joined = gpd.sjoin(
        centroids,
        lor[["geometry"] + lor_attr_cols],
        how="left",
        predicate="within",
    )
    # If a centroid falls on a shared boundary it may match two polygons — keep first
    joined = joined[~joined.index.duplicated(keep="first")]

    for col in lor_attr_cols:
        if col in joined.columns:
            out[col] = joined[col].reindex(out.index).values

    n_matched = out["PLR_ID"].notna().sum() if "PLR_ID" in out.columns else 0
    logger.info("LOR attributes joined: %d/%d cells matched.", n_matched, len(out))
    return out


def add_poi_layer(
    grid: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
    chain_col: str = "chain",
) -> gpd.GeoDataFrame:
    """Count and classify POIs (supermarkets) per grid cell.

    Reprojects *pois* to match *grid* CRS, uses the ``point`` column for
    location (falling back to geometry centroid if ``point`` is absent),
    spatial-joins to *grid* polygons, and aggregates per cell.

    Expected output columns added to *grid*:

    * ``poi_count``          — total number of POIs in cell (int)
    * ``poi_chains``         — comma-separated list of chain names (str)
    * one column per chain   — boolean flag ``has_{chain_name}`` (e.g.
                               ``has_Rewe``, ``has_Lidl``, …)

    Parameters
    ----------
    grid:
        Population grid GeoDataFrame with polygon geometry in EPSG:3035.
    pois:
        OSM POI GeoDataFrame from :func:`~hotelling.spatial.osm.fetch_pois`,
        CRS EPSG:4326 (will be reprojected automatically).
    chain_col:
        Column in *pois* that holds the canonical chain name.

    Returns
    -------
    geopandas.GeoDataFrame
        *grid* with POI summary columns added.
    """
    out = grid.copy()
    out["_grid_idx"] = out.index

    if pois.empty:
        out["poi_count"] = 0
        out["poi_chains"] = ""
        return out.drop(columns=["_grid_idx"])

    # Reproject POIs to grid CRS
    pois_proj = pois.to_crs(grid.crs) if pois.crs != grid.crs else pois.copy()

    # Use 'point' column (representative point from fetch_pois) if available,
    # otherwise use centroid.  The 'point' column stores raw Shapely objects
    # in the *original* CRS (EPSG:4326), so we must reproject them separately
    # after the GeoDataFrame has been reprojected.
    pois_proj = pois_proj.copy()
    pois_proj["geometry"] = pois_proj.geometry.centroid

    chain_data = pois_proj[[chain_col]].copy() if chain_col in pois_proj.columns else None
    poi_pts = pois_proj[["geometry"]].copy()
    if chain_data is not None:
        poi_pts[chain_col] = chain_data[chain_col].values

    joined = gpd.sjoin(
        poi_pts,
        out[["_grid_idx", "geometry"]],
        how="left",
        predicate="within",
    )

    poi_counts = joined.groupby("_grid_idx").size().rename("poi_count")
    out = out.merge(poi_counts, left_on="_grid_idx", right_index=True, how="left")
    out["poi_count"] = out["poi_count"].fillna(0).astype(int)

    if chain_col in joined.columns:
        chain_s = joined.dropna(subset=[chain_col])
        chain_list = chain_s.groupby("_grid_idx")[chain_col].apply(list).rename("poi_chains")
        out = out.merge(chain_list, left_on="_grid_idx", right_index=True, how="left")
        out["poi_chains"] = out["poi_chains"].apply(
            lambda x: ",".join(sorted(set(x))) if isinstance(x, list) else ""
        )
        all_chains = sorted(chain_s[chain_col].dropna().unique())
        chains_per_cell = chain_s.groupby("_grid_idx")[chain_col].apply(set)
        for cname in all_chains:
            col = "has_" + cname.replace(" ", "_").replace("-", "_")
            out[col] = out["_grid_idx"].map(
                lambda i, c=cname: c in chains_per_cell.get(i, set())
            ).fillna(False)
    else:
        out["poi_chains"] = ""

    out = out.drop(columns=["_grid_idx"])
    logger.info(
        "POI layer added: %d POIs, %d cells with poi_count>0.",
        len(pois), (out["poi_count"] > 0).sum(),
    )
    return out


def assemble_simulation_grid(
    pop_grid: gpd.GeoDataFrame,
    lor: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Validate schema and finalise the simulation-ready grid.

    This function receives a *pop_grid* that has already been enriched by
    the pipeline (LOR attributes, ESIx/MSS scores, IHK employment, transit
    hub flags, CBD flags, and POI layer have all been joined in prior phases).
    Its job is to:

    1. Verify that all required columns are present, raising ``KeyError``
       with a clear message for any that are missing.
    2. Fill guaranteed-present columns with safe defaults where values are
       ``NaN`` (e.g. ``poi_count`` → 0, ``Einwohner`` → 0).
    3. Reset the index to a clean RangeIndex.
    4. Return the final GeoDataFrame ready for the simulation engine.

    Required output columns (will raise ``KeyError`` if absent):
        ``x_mp_100m``, ``y_mp_100m``, ``geometry``, ``Einwohner``,
        ``PLR_ID``, ``PLR_NAME``, ``poi_count``

    Parameters
    ----------
    pop_grid:
        Fully-enriched grid GeoDataFrame produced by all prior pipeline
        phases.  Must already contain the required columns listed above.
    lor:
        Selected LOR districts (passed for reference / logging only;
        not used for spatial joins here).
    pois:
        OSM POI GeoDataFrame (passed for reference / logging only;
        not used for spatial joins here).

    Returns
    -------
    geopandas.GeoDataFrame
        Schema-validated simulation grid in EPSG:3035, clean RangeIndex.

    Raises
    ------
    KeyError
        If any required column is missing from *pop_grid*.
    """
    REQUIRED = ["x_mp_100m", "y_mp_100m", "geometry", "Einwohner", "PLR_ID", "PLR_NAME", "poi_count"]
    missing = [c for c in REQUIRED if c not in pop_grid.columns]
    if missing:
        raise KeyError(
            f"assemble_simulation_grid: required columns missing: {missing}. "
            "Ensure add_lor_attributes and add_poi_layer have been run."
        )
    out = pop_grid.copy()
    out["poi_count"] = out["poi_count"].fillna(0).astype(int)
    out["Einwohner"] = out["Einwohner"].fillna(0).astype(np.int32)
    out = out.reset_index(drop=True)
    logger.info(
        "Simulation grid assembled: %d cells, %d columns, CRS=%s.",
        len(out), len(out.columns), out.crs,
    )
    return out


def add_lcc_layer(
    grid: gpd.GeoDataFrame,
    lcc_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add Local Commercial Centre mall-intersection data to the simulation grid.

    Processes only ``shop=mall`` features from *lcc_gdf* (from
    :func:`~hotelling.spatial.osm.fetch_pois` with ``type="LCC"``).
    For each grid cell, computes what fraction of each overlapping mall's
    area falls within that cell, and records the maximum fraction.

    Logic extracted from GEO_03_OSM.ipynb.
    Saves intermediate to ``data/processed/grid_malls.parquet``.

    Parameters
    ----------
    grid:
        Simulation grid GeoDataFrame with polygon geometry in EPSG:3035.
    lcc_gdf:
        LCC POI GeoDataFrame from :func:`~hotelling.spatial.osm.fetch_pois`
        with ``type="LCC"``.  CRS EPSG:4326 (reprojected internally).

    Returns
    -------
    geopandas.GeoDataFrame
        *grid* with added columns:

        * ``mall_area`` (float) — area of overlapping mall in m², 0 if none
        * ``mall_intersection_fraction`` (float) — fraction of that mall
          covered by the cell, NaN for cells with no mall
        * ``has_mall`` (bool) — True if any mall intersects the cell
    """
    lcc_proj = lcc_gdf.to_crs(grid.crs) if lcc_gdf.crs != grid.crs else lcc_gdf.copy()

    out = grid.copy()
    out["mall_area"] = 0.0
    out["mall_intersection_fraction"] = float("nan")
    out["has_mall"] = False

    if "shop" not in lcc_proj.columns:
        logger.warning("No 'shop' column in lcc_gdf — add_lcc_layer returning grid unchanged.")
        return out

    mall_gdf = lcc_proj[lcc_proj["shop"] == "mall"].copy()
    mall_gdf["mall_area"] = mall_gdf.geometry.area
    mall_gdf = mall_gdf[mall_gdf["mall_area"] > 0].reset_index(drop=True)

    if mall_gdf.empty:
        logger.warning("No mall features (shop=mall) in lcc_gdf.")
        return out

    # Spatial join: grid cells ← malls (left join, intersects)
    grid_malls = gpd.sjoin(
        grid.copy(),
        mall_gdf[["geometry", "mall_area"]],
        how="left",
        predicate="intersects",
    )

    def _fraction(row: pd.Series) -> float:
        if pd.isna(row.get("index_right")):
            return float("nan")
        mall_geom = mall_gdf.iloc[int(row["index_right"])].geometry
        inter = row.geometry.intersection(mall_geom).area
        return inter / row["mall_area"] if row["mall_area"] > 0 else 0.0

    grid_malls["mall_intersection_fraction"] = grid_malls.apply(_fraction, axis=1)
    grid_malls["has_mall"] = grid_malls["mall_intersection_fraction"].notna()

    # If a cell overlaps multiple malls, keep the row for the largest mall
    id_cols = [c for c in grid.columns if c != "geometry"]
    grid_malls = (
        grid_malls.sort_values("mall_area", ascending=False)
        .drop_duplicates(subset=id_cols)
        .reset_index(drop=True)
    )
    if "index_right" in grid_malls.columns:
        grid_malls = grid_malls.drop(columns=["index_right"])

    grid_malls["mall_area"] = grid_malls["mall_area"].fillna(0.0)

    out_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "grid_malls.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_malls.to_parquet(out_path, index=False)
    logger.info(
        "LCC mall layer: %d cells with mall overlap. Saved → %s.",
        int(grid_malls["has_mall"].sum()), out_path,
    )
    return grid_malls


def normalize_social_indices(
    demand_grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Normalize ESIx and MSS social-status indices for the demand model.

    Applies two normalization transformations to the raw index columns that
    were joined by :func:`build_demand_grid`:

    **ESIx** (``esix_wert``, continuous):
        Min-max normalization → ``esix_normalized`` ∈ [0, 1].

    **MSS** (``si_n``, ordinal 1–4):
        Ordinal-to-quantile mapping using the 20th / 40th / 60th / 80th
        percentiles of ``esix_normalized``.  Status class 1 (highest social
        status) maps to Q4 (80th percentile), class 4 (lowest) maps to Q1
        (20th percentile).  Sentinel value -9999 (missing) is replaced with
        ``NaN`` before mapping.

    The mapping is defined as::

        si_n=1 → esix_normalized.quantile(0.8)   (high status ↔ high ESIx)
        si_n=2 → esix_normalized.quantile(0.6)
        si_n=3 → esix_normalized.quantile(0.4)
        si_n=4 → esix_normalized.quantile(0.2)   (low status ↔ low ESIx)

    This cross-calibration anchors the coarser ordinal MSS scale to the
    distribution of the continuous ESIx metric, enabling both indices to be
    used interchangeably in the logit demand model.

    Parameters
    ----------
    demand_grid:
        GeoDataFrame that already contains ``esix_wert`` (float) and
        ``si_n`` (float, ordinal 1–4 plus -9999 sentinel) columns from the
        MSS/ESIx sjoin in :func:`build_demand_grid`.

    Returns
    -------
    geopandas.GeoDataFrame
        *demand_grid* with two new columns added in-place:
        ``esix_normalized`` and ``si_normalized``.
    """
    out = demand_grid.copy()

    # ESIx: min-max normalization
    emin = out["esix_wert"].min()
    emax = out["esix_wert"].max()
    denom = emax - emin
    if denom == 0 or pd.isna(denom):
        out["esix_normalized"] = float("nan")
    else:
        out["esix_normalized"] = (out["esix_wert"] - emin) / denom

    # MSS: ordinal → quantile mapping
    out["si_n"] = out["si_n"].replace({-9999: np.nan})
    q = out["esix_normalized"].quantile([0.2, 0.4, 0.6, 0.8]).values
    out["si_normalized"] = out["si_n"].replace({
        1.0: q[3],   # si_n=1 (highest status) → 80th percentile of ESIx
        2.0: q[2],
        3.0: q[1],
        4.0: q[0],   # si_n=4 (lowest status)  → 20th percentile of ESIx
    })

    logger.info(
        "Social indices normalized: esix_normalized mean=%.3f, "
        "si_normalized non-null=%d.",
        out["esix_normalized"].mean(skipna=True),
        out["si_normalized"].notna().sum(),
    )
    return out


def enrich_supermarkets_with_brw(
    supermarkets: gpd.GeoDataFrame,
    brw: gpd.GeoDataFrame,
    output_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Join BRW land-value attributes to incumbent supermarket locations.

    Performs a spatial join (intersects predicate) between *supermarkets*
    point geometry and *brw* zone polygons.  The result adds the BRW zone
    attributes (``bezirk``, ``brw``, ``nutzung``) to each store and adds an
    integer ``id`` column equal to the store's original integer index.

    Saves the enriched GeoDataFrame to
    ``data/processed/supermarkets_full.parquet`` (or *output_path*).

    Parameters
    ----------
    supermarkets:
        Supermarket GeoDataFrame from :func:`~hotelling.spatial.osm.process_supermarkets`.
        CRS must be EPSG:3035 or will be reprojected.
    brw:
        BRW land-value zones GeoDataFrame.  Must contain at minimum the
        columns ``bezirk``, ``brw``, ``nutzung``, and ``geometry``.
    output_path:
        Path where the parquet output is written.
        Default: ``data/processed/supermarkets_full.parquet``.

    Returns
    -------
    geopandas.GeoDataFrame
        *supermarkets* with BRW columns added and an ``id`` column.
    """
    out = supermarkets.copy()
    brw_proj = brw.to_crs(out.crs) if brw.crs != out.crs else brw.copy()

    brw_cols = [c for c in ["bezirk", "brw", "nutzung", "geometry"] if c in brw_proj.columns]
    out = out.sjoin(brw_proj[brw_cols], how="left", predicate="intersects")
    out["id"] = out.index

    _save = output_path or Path("data/processed/supermarkets_full.parquet")
    _save = Path(_save)
    _save.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_save, index=False)
    logger.info(
        "supermarkets_full saved → %s  (%d rows).", _save, len(out)
    )
    return out


def build_demand_grid(
    grid: gpd.GeoDataFrame,
    grid_malls: gpd.GeoDataFrame,
    grid_with_stations: gpd.GeoDataFrame,
    travel_times: pd.DataFrame,
    employment_clusters: gpd.GeoDataFrame,
    *,
    mss_path: Path | None = None,
    esix_path: Path | None = None,
    output_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Assemble the final demand grid from all spatial layers.

    Combines the population grid with transit, employment, travel-time, and
    social-index data into one GeoDataFrame ready for the logit demand model.

    Steps:
    1. Assign canonical cell IDs via :func:`~hotelling.spatial.census.make_cell_id`.
    2. Propagate ``has_mall`` and ``has_station`` / ``station_class`` /
       ``matched_db_station`` flags from *grid_malls* and *grid_with_stations*.
    3. Spatial-join employment-cluster attributes from *employment_clusters*.
    4. Attach a per-cell travel-time dict
       ``{store_id_str: travel_time_minutes}`` from *travel_times*.
    5. Spatial-join full MSS and ESIx polygon attributes.
    6. Normalize social indices via :func:`normalize_social_indices`.
    7. Save to parquet and return.

    Parameters
    ----------
    grid:
        Population grid GeoDataFrame in EPSG:3035, with polygon geometry.
        Must contain ``x_mp_100m``, ``y_mp_100m``, and ``GITTER_ID_100m``
        (or the latter is synthesised from the coordinates).
    grid_malls:
        Output of :func:`add_lcc_layer`, with columns ``GITTER_ID_100m``
        (or synthesised) and ``has_mall``.
    grid_with_stations:
        Output of :func:`~hotelling.spatial.city_data.identify_transport_hubs`,
        with columns ``station_class`` and ``matched_db_station``.
    travel_times:
        DataFrame with columns ``from_id`` (INSPIRE cell ID str),
        ``to_id`` (store integer index as str), ``travel_time`` (int, minutes).
        Loaded from ``data/processed/travel_times.parquet``.
    employment_clusters:
        Employment/prime-location cluster GeoDataFrame (EPSG:3035).
        All columns are attached to intersecting grid cells via sjoin.
    mss_path:
        Path to ``mss.gpkg``.  Default: ``data/raw/mss.gpkg``.
    esix_path:
        Path to ``esix.gpkg``.  Default: ``data/raw/esix.gpkg``.
    output_path:
        Parquet output path.  Default: ``data/processed/demand_grid.parquet``.

    Returns
    -------
    geopandas.GeoDataFrame
        Assembled demand grid in EPSG:3035 with columns:
        ``GITTER_ID_100m``, ``has_mall``, ``has_station``, ``station_class``,
        ``matched_db_station``, employment-cluster columns,
        ``travel_times`` (object, dict per cell),
        ``esix_wert``, ``si_n``, ``esix_normalized``, ``si_normalized``,
        plus all original *grid* columns.
    """
    from hotelling.spatial.census import make_cell_id  # noqa: PLC0415  (avoid circular import)

    _mss_path  = Path(mss_path  or "data/raw/mss.gpkg")
    _esix_path = Path(esix_path or "data/raw/esix.gpkg")
    _out_path  = Path(output_path or "data/processed/demand_grid.parquet")

    demand_grid = grid.copy()

    # ── Step 1: Assign canonical GITTER_ID_100m ───────────────────────────────
    demand_grid["GITTER_ID_100m"] = demand_grid.apply(make_cell_id, axis=1)
    grid_malls      = grid_malls.to_crs(demand_grid.crs).copy()
    grid_with_stations = grid_with_stations.to_crs(demand_grid.crs).copy()
    employment_clusters = employment_clusters.to_crs(demand_grid.crs).copy()

    grid_malls["GITTER_ID_100m"]       = grid_malls.apply(make_cell_id, axis=1)
    grid_with_stations["GITTER_ID_100m"] = grid_with_stations.apply(make_cell_id, axis=1)

    # ── Step 2: Mall and station flags ────────────────────────────────────────
    grid_with_stations["has_station"] = grid_with_stations["station_class"].notna()
    grid_malls_true    = grid_malls[grid_malls["has_mall"]]
    grid_stations_true = grid_with_stations[grid_with_stations["has_station"]]

    demand_grid["has_mall"]    = False
    demand_grid["has_station"] = False
    demand_grid.loc[
        demand_grid["GITTER_ID_100m"].isin(grid_malls_true["GITTER_ID_100m"]),
        "has_mall",
    ] = True
    demand_grid.loc[
        demand_grid["GITTER_ID_100m"].isin(grid_stations_true["GITTER_ID_100m"]),
        "has_station",
    ] = True

    station_info = grid_with_stations.set_index("GITTER_ID_100m")[
        ["station_class", "matched_db_station"]
    ]
    demand_grid = demand_grid.join(station_info, on="GITTER_ID_100m")

    # ── Step 3: Employment clusters (sjoin intersects) ────────────────────────
    demand_grid = demand_grid.sjoin(
        employment_clusters, how="left", predicate="intersects"
    )
    if "index_right" in demand_grid.columns:
        demand_grid = demand_grid.drop(columns=["index_right"])

    # ── Step 4: Travel-time dict per cell ─────────────────────────────────────
    demand_grid["travel_times"] = None
    travel_times = travel_times.copy()
    travel_times["to_id"] = travel_times["to_id"].apply(str)
    travel_time_lookup = (
        travel_times.groupby("from_id")
        .apply(lambda df: df.set_index("to_id")["travel_time"].to_dict())
    )
    demand_grid["travel_times"] = demand_grid["GITTER_ID_100m"].map(travel_time_lookup)

    # ── Step 5: MSS and ESIx full-attribute sjoin ─────────────────────────────
    for path, label in [(_mss_path, "MSS"), (_esix_path, "ESIx")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} file not found at {path}. "
                "Run download_index_data() first."
            )

    mss  = gpd.read_file(_mss_path).to_crs(demand_grid.crs)
    esix = gpd.read_file(_esix_path).to_crs(demand_grid.crs)

    if "plr_id" in mss.columns:
        mss = mss.rename(columns={"plr_id": "plr_id_mss"})
    if "plr_id" in esix.columns:
        esix = esix.rename(columns={"plr_id": "plr_id_esix"})

    # Drop residual index_right from previous sjoins before each new sjoin
    if "index_right" in demand_grid.columns:
        demand_grid = demand_grid.drop(columns=["index_right"])
    demand_grid = demand_grid.sjoin(mss, how="left", predicate="intersects")

    if "index_right" in demand_grid.columns:
        demand_grid = demand_grid.drop(columns=["index_right"])
    demand_grid = demand_grid.sjoin(esix, how="left", predicate="intersects")

    if "index_right" in demand_grid.columns:
        demand_grid = demand_grid.drop(columns=["index_right"])

    # ── Step 6: Normalize social indices ─────────────────────────────────────
    if "esix_wert" in demand_grid.columns and "si_n" in demand_grid.columns:
        demand_grid = normalize_social_indices(demand_grid)
    else:
        logger.warning(
            "esix_wert or si_n column not found after MSS/ESIx join — "
            "skipping normalization. Check the gpkg column names."
        )

    # ── Step 7: Save ──────────────────────────────────────────────────────────
    _out_path.parent.mkdir(parents=True, exist_ok=True)
    demand_grid.to_parquet(_out_path)
    logger.info(
        "demand_grid saved → %s  (%d cells, %d columns).",
        _out_path, len(demand_grid), len(demand_grid.columns),
    )
    return demand_grid
