"""Data fetching - ESIx, GESIx, CBD, Employment

Key dependencies: geopandas (optional ``[spatial]`` extra).
"""
from __future__ import annotations

import logging
import os
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)

__all__ = [
    "download_index_data",
    "download_stadtstruktur",
    "download_station_data",
    "download_IHK_data",
    "download_medianeinkommen_data",
]

def download_index_data() -> None:
    """Download the Index data from the Berlin Senate of Health."""
    
    logger.info("Starting Index data download and processing.")
    
    WFS_URL_ESIX = (
    "https://gdi.berlin.de/services/wfs/gssa_esix2022"
    "?service=WFS"
    "&version=2.0.0"
    "&request=GetFeature"
    "&typeNames=gssa_esix2022:gssa_esix2022"
    "&outputFormat=application/json"
    "&srsName=EPSG:25833"          # native CRS — ETRS89/UTM33N, metres
    )
    
    WFS_URL_MSS = (
    "https://gdi.berlin.de/services/wfs/mss_2023"
    "?service=WFS"
    "&version=2.0.0"
    "&request=GetFeature"
    "&typeNames=mss_2023:mss2023_indizes_542"
    "&outputFormat=application/json"
    "&srsName=EPSG:25833"          # native CRS — ETRS89/UTM33N, metres
    )
    gdf_esix = gpd.read_file(WFS_URL_ESIX)
    gdf_mss = gpd.read_file(WFS_URL_MSS)
    logger.info("ESIX and MSS data downloaded.")
    
    save_path = "data/raw/esix.gpkg"
    gdf_esix.to_file(save_path, driver="GPKG", layer="esix")
    logger.info("ESIX data saved to %s.", save_path)
    
    save_path = "data/raw/mss.gpkg"
    gdf_mss.to_file(save_path, driver="GPKG", layer="mss")
    logger.info("MSS data saved to %s.", save_path)

def download_stadtstruktur() -> None:
    """Download the Stadtstruktur data from the Berlin GeoPortal."""
    
    WFS_URL_1 = (
    "https://gdi.berlin.de/services/wfs/ua_stadtstruktur"
    "?service=WFS"
    "&version=2.0.0"
    "&request=GetFeature"
    "&typeNames=ua_stadtstruktur:b_stadtstruktur_differenziert_2024"
    "&outputFormat=application/json"
    "&srsName=EPSG:25833"          # native CRS — ETRS89/UTM33N, metres
    )
    
    WFS_URL_2 = (
    "https://gdi.berlin.de/services/wfs/alkis_gebaeude"
    "?service=WFS"
    "&version=2.0.0"
    "&request=GetFeature"
    "&typeNames=alkis_gebaeude:gebaeude"
    "&outputFormat=application/json"
    "&srsName=EPSG:25833"          # native CRS — ETRS89/UTM33N, metres
    )
    
    WFS_URL_3 = (
    "https://gdi.berlin.de/services/wfs/step_zen_2040"
    "?service=WFS"
    "&version=2.0.0"
    "&request=GetFeature"
    "&typeNames=step_zen_2040:step_zen_2040_fma"
    "&outputFormat=application/json"
    "&srsName=EPSG:25833"          # native CRS — ETRS89/UTM33N, metres
    )
    
    WFS_URL_4 = (
    "https://gdi.berlin.de/services/wfs/step_zen_2040"
    "?service=WFS"
    "&version=2.0.0"
    "&request=GetFeature"
    "&typeNames=step_zen_2040:step_zen_2040_zh"
    "&outputFormat=application/json"
    "&srsName=EPSG:25833"          # native CRS — ETRS89/UTM33N, metres
    )
    
    gdf_1 = gpd.read_file(WFS_URL_1)
    gdf_2 = gpd.read_file(WFS_URL_2)
    gdf_3 = gpd.read_file(WFS_URL_3)
    gdf_4 = gpd.read_file(WFS_URL_4)
    logger.info("Stadtstruktur, Gebaeude and Zentren data downloaded.")
    
    save_path = "data/raw/stadtstruktur.gpkg"
    gdf_1.to_file(save_path, driver="GPKG", layer="stadtstruktur")
    logger.info("Stadtstruktur data saved to %s.", save_path)
    
    save_path = "data/raw/gebaeude.gpkg"
    gdf_2.to_file(save_path, driver="GPKG", layer="gebaeude")
    logger.info("Gebaeude data saved to %s.", save_path)
    
    save_path = "data/raw/zentren.gpkg"
    gdf_3.to_file(save_path, driver="GPKG", layer="zentren_fma")
    logger.info("Zentren FMA data saved to %s.", save_path)
    
    save_path = "data/raw/zentren.gpkg"
    gdf_4.to_file(save_path, driver="GPKG", layer="zentren_zh")
    logger.info("Zentren ZH data saved to %s.", save_path)

def download_station_data() -> None:
    """Download the Station data from the DB InfraGo."""
    
    link_db = "https://www.dbinfrago.com/resource/blob/13518698/1cd204bc2c7a98b2490822ee6fc200ad/Stationspreisliste-2026-data.pdf"
    save_db_path = "data/raw/db_station_data.pdf"
    urllib.request.urlretrieve(link_db, save_db_path)
    logger.info("Station data saved to %s.", save_db_path)
    
    # Load the pdf file
    with pdfplumber.open(save_db_path) as pdf:
        all_data = []
        for page in pdf.pages:
            # We crop the page to avoid the footer/header text if necessary
            # page.crop((left, top, right, bottom))
            
            table = page.extract_table(table_settings={
                "vertical_strategy": "text", 
                "horizontal_strategy": "text",
                "snap_y_tolerance": 5,
            })
            
            if table:
                all_data.extend(table)
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    save_db_path = "data/raw/db_station_data.csv"
    df.to_csv(save_db_path, index=False)
    logger.info("Station data saved to %s.", save_db_path)

    # Download the zip GTFS file and unpack
    link_gtfs = "https://unternehmen.vbb.de/fileadmin/user_upload/VBB/Dokumente/API-Datensaetze/gtfs-2024.zip"
    save_gtfs_path = Path("data/raw/gtfs-2024.zip")
    urllib.request.urlretrieve(link_gtfs, save_gtfs_path)
    logger.info("GTFS data saved to %s.", save_gtfs_path)
    # Unpack the zip file
    with zipfile.ZipFile(save_gtfs_path, 'r') as zip_ref:
        zip_ref.extractall(save_gtfs_path.parent)
    logger.info("GTFS data unpacked to %s.", save_gtfs_path.parent)
    
    # Remove the zip file
    os.remove(save_gtfs_path)
    logger.info("GTFS zip file removed.")
    logger.info("Station data downloaded and processed.")
    
def download_IHK_data() -> None:
    """Download the IHK data from the Berlin Chamber of Commerce."""
    
    raise NotImplementedError("This method is not implemented. IHK data should be downloaded manually.")

def download_medianeinkommen_data() -> None:
    """Download the Medianeinkommen data from the Berlin Senate of Finance."""
    
    raise NotImplementedError("This method is not implemented. Medianeinkommen data should be downloaded manually.")


def process_ihk_data(
    grid: gpd.GeoDataFrame,
    ihk_path: Path,
) -> gpd.GeoDataFrame:
    """Load IHK Berlin business microdata and aggregate employment per grid cell.

    Reads the manually-downloaded IHK CSV at *ihk_path*, parses employee-count
    ranges (e.g. ``"1 - 3 Beschäftigte"`` → midpoint 2), reprojects to
    EPSG:3035, spatial-joins to *grid* polygons, sums employment per cell, and
    merges the result as an ``empl`` column.

    Parameters
    ----------
    grid:
        Population grid GeoDataFrame with polygon geometry in EPSG:3035.
    ihk_path:
        Path to the IHK CSV file (``2023_12_IHK_Berlin_Gewerbedaten.csv`` or
        equivalent).

    Returns
    -------
    geopandas.GeoDataFrame
        *grid* with an additional ``empl`` column (float, 0 for cells with no
        businesses).

    Notes
    -----
    IHK data cannot be downloaded automatically.  Place the file at
    ``data/raw/2023_12_IHK_Berlin_Gewerbedaten.csv`` before calling this
    function.
    """
    raise NotImplementedError(
        "process_ihk_data: load CSV, parse employee ranges via midpoint formula, "
        "gpd.points_from_xy → to_crs(3035), gpd.sjoin(grid, predicate='within'), "
        "groupby cell index and sum, merge empl back to grid."
    )


def process_esix_mss_data(
    grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Join ESIx 2022 and MSS 2023 social-status indices to grid cells.

    Loads ``data/raw/esix.gpkg`` and ``data/raw/mss.gpkg`` (written by
    :func:`download_index_data`), reprojects to EPSG:3035, spatially joins to
    *grid* polygons, and attaches the relevant index columns.

    Expected output columns added to *grid*:

    * ``esix_score``  — ESIx 2022 composite social-structure index (float)
    * ``mss_score``   — MSS 2023 social-development index (float)

    Parameters
    ----------
    grid:
        Population grid GeoDataFrame with polygon geometry in EPSG:3035.

    Returns
    -------
    geopandas.GeoDataFrame
        *grid* with ``esix_score`` and ``mss_score`` columns added.
        Cells that do not intersect any index polygon receive ``NaN``.
    """
    raise NotImplementedError(
        "process_esix_mss_data: gpd.read_file('data/raw/esix.gpkg'), to_crs(3035), "
        "gpd.sjoin(grid, predicate='intersects'), mean-aggregate the index columns "
        "per grid cell, merge back."
    )


def identify_transport_hubs(
    grid: gpd.GeoDataFrame,
    gtfs_dir: Path | None = None,
) -> gpd.GeoDataFrame:
    """Flag grid cells that contain or are adjacent to major transit nodes.

    Parses VBB GTFS ``stops.txt`` from *gtfs_dir* (default:
    ``data/raw/gtfs-2024/``), classifies stops by route type / service
    frequency, and adds columns to *grid* indicating transit accessibility.

    Expected output columns added to *grid*:

    * ``transit_stops``       — number of transit stops within the cell (int)
    * ``is_transit_hub``      — True if a major hub (S/U-Bahn interchange) is
                                present (bool)

    Parameters
    ----------
    grid:
        Population grid GeoDataFrame with polygon geometry in EPSG:3035.
    gtfs_dir:
        Directory containing unpacked GTFS files.  Defaults to
        ``data/raw/gtfs-2024/``.

    Returns
    -------
    geopandas.GeoDataFrame
        *grid* with ``transit_stops`` and ``is_transit_hub`` columns.
    """
    raise NotImplementedError(
        "identify_transport_hubs: read stops.txt from GTFS, create GeoDataFrame "
        "from stop_lat/stop_lon, to_crs(3035), sjoin to grid, count stops per cell, "
        "flag major S/U-Bahn interchanges."
    )


def identify_cbd(
    grid: gpd.GeoDataFrame,
    zentren_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Flag grid cells that fall within a Central Business District polygon.

    Loads the Zentren FMA layer from ``data/raw/zentren.gpkg`` (written by
    :func:`download_stadtstruktur`), reprojects to EPSG:3035, and marks grid
    cells that intersect a CBD-type urban-centre polygon.

    Expected output column added to *grid*:

    * ``is_cbd`` — True if the cell intersects a Hauptzentrum / CBD polygon (bool)

    Parameters
    ----------
    grid:
        Population grid GeoDataFrame with polygon geometry in EPSG:3035.
    zentren_path:
        Path to the zentren GeoPackage.  Defaults to ``data/raw/zentren.gpkg``.

    Returns
    -------
    geopandas.GeoDataFrame
        *grid* with an ``is_cbd`` boolean column.
    """
    raise NotImplementedError(
        "identify_cbd: gpd.read_file('data/raw/zentren.gpkg', layer='zentren_fma'), "
        "to_crs(3035), filter for 'Hauptzentrum' or equivalent CBD category, "
        "sjoin to grid with predicate='intersects', flag cells."
    )