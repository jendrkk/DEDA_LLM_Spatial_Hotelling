"""Data fetching - ESIx, GESIx, CBD, Employment

Key dependencies: geopandas (optional ``[spatial]`` extra).
"""

import logging
import urllib.request
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)

def download_index_data():
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

def download_IHK_data():
    """Download the IHK data from the Berlin Chamber of Commerce."""
    
    raise NotImplementedError("This method is not implemented. IHK data should be downloaded manually.")

def download_stadtstruktur():
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