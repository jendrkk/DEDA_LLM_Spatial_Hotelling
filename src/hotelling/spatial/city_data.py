"""Data fetching - ESIx, GESIx, CBD, Employment

Key dependencies: geopandas (optional ``[spatial]`` extra).
"""

import logging
import urllib.request

logger = logging.getLogger(__name__)

def download_gesix_data():
    """Download the GESIx data from the Berlin Senate of Health."""
    
    logger.info("Starting GESIx data download and processing.")
    
    link_bezirke = "https://www.berlin.de/sen/gesundheit/_assets/daten/gesundheits-und-sozialstrukturatlas/gssa_2022_bezirke.csv"
    link_planungsraeume = "https://www.berlin.de/sen/gesundheit/_assets/daten/gesundheits-und-sozialstrukturatlas/gssa_2022_planungsraeume.csv"
    
    save_path_bezirke = "data/raw/gesix_bezirke.csv"
    save_path_planungsraeume = "data/raw/gesix_planungsraeume.csv"
    urllib.request.urlretrieve(link_bezirke, save_path_bezirke)
    urllib.request.urlretrieve(link_planungsraeume, save_path_planungsraeume)
    
    logger.info("GESIx data downloaded.")

def download_IHK_data():
    """Download the IHK data from the Berlin Chamber of Commerce."""
    
    logger.info("Starting IHK data download and processing.")
    
    link_ihk = ...