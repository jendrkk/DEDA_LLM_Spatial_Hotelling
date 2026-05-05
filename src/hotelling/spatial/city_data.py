"""Data fetching - ESIx, GESIx, CBD, Employment

Key dependencies: geopandas (optional ``[spatial]`` extra).
"""

import logging
import urllib.request
import pandas as pd
import geopandas as gpd
import pdfplumber
import zipfile
import os

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

def download_station_data():
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
    save_gtfs_path = "data/raw/gtfs-2024.zip"
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
    
def download_IHK_data():
    """Download the IHK data from the Berlin Chamber of Commerce."""
    
    raise NotImplementedError("This method is not implemented. IHK data should be downloaded manually.")

def download_medianeinkommen_data():
    """Download the Medianeinkommen data from the Berlin Senate of Finance."""
    
    raise NotImplementedError("This method is not implemented. Medianeinkommen data should be downloaded manually.")