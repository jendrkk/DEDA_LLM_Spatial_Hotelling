"""Sub-city administrative units (e.g. Berlin LOR) download and normalisation.

Key dependencies: geopandas (optional ``[spatial]`` extra); ``py7zr`` for .7z archives.
"""
from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)


def download_lor_shapes() -> None:
    """Download Berlin LOR shapes from SenStadt, extract, reproject to EPSG:3035, save parquet."""
    logger.info("Starting LOR shapes download and conversion.")
    url = (
        "https://www.berlin.de/sen/sbw/_assets/stadtdaten/stadtwissen/"
        "lebensweltlich-orientierte-raeume/lor_2019-01-01_shapefiles_nur_id.7z?ts=1770289260"
    )
    save_path = "data/raw/lor_shapes.7z"
    extract_dir = Path("data/raw/lor_shapes")
    urllib.request.urlretrieve(url, save_path)
    logger.info("Downloaded LOR archive to %s.", save_path)
    try:
        import py7zr  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "Missing optional dependency 'py7zr' required to extract .7z archives. "
            "Install it with: pip install py7zr",
        ) from exc

    with py7zr.SevenZipFile(save_path, mode="r") as archive:
        archive.extractall(path=str(extract_dir))
    logger.info("Extracted LOR archive to %s.", extract_dir)
    os.remove(save_path)

    shapefiles = list(extract_dir.glob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError(f"No shapefiles found in extracted LOR directory: {extract_dir}")

    priority_tokens = ["_PLR_", "_PGR_", "_BZR_"]
    selected = None
    for token in priority_tokens:
        selected = next((path for path in shapefiles if token in path.name), None)
        if selected is not None:
            break
    if selected is None:
        selected = shapefiles[0]

    logger.info("Selected LOR shapefile %s for conversion.", selected.name)
    data = gpd.read_file(selected)
    data = data.to_crs(crs="EPSG:3035")
    logger.info("Reprojected to CRS EPSG:3035.")

    parquet_path = Path("data/raw/lor_shapes.parquet")
    data.to_parquet(parquet_path)
    logger.info("Saved LOR parquet to %s.", parquet_path)
    shutil.rmtree(extract_dir)
    logger.info("Removed extracted LOR folder %s after parquet conversion.", extract_dir)


def download_local_shapes() -> None:
    """Download local planning-area shapes for cities other than Berlin (placeholder)."""
    raise NotImplementedError("This method is not implemented yet.")
