"""Data fetching – ESIx, MSS, CBD, Employment, Transit.

Key dependencies: geopandas, requests (optional ``[spatial]`` extra).
Install ``tqdm`` for progress bars: ``pip install tqdm``.

WFS layer inventory (all verified against live GetCapabilities):
  gssa_esix2022      → gssa_esix2022:gssa_esix2022           (~542 features)
  mss_2023           → mss_2023:mss2023_indizes_542           (~542 features)
  ua_stadtstruktur   → ua_stadtstruktur:b_stadtstruktur_differenziert_2024  (26 613)
  alkis_gebaeude     → alkis_gebaeude:gebaeude                (783 071) ← large
  step_zen_2040      → step_zen_2040:step_zen_2040_fma        (small)
  step_zen_2040      → step_zen_2040:step_zen_2040_zh         (small)
"""
from __future__ import annotations

import logging
import os
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree

logger = logging.getLogger(__name__)

__all__ = [
    "download_IHK_data",
    "download_alkis_data",
    "download_index_data",
    "download_medianeinkommen_data",
    "download_stadtstruktur",
    "download_station_data",
    "identify_cbd",
    "identify_transport_hubs",
    "process_esix_mss_data",
    "process_gebaeude_stadtstruktur",
    "process_ihk_data",
    "run_prime_location_clustering",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GDI_WFS_BASE = "https://gdi.berlin.de/services/wfs/{service}"

_DB_STATION_PDF_URL = (
    "https://www.dbinfrago.com/resource/blob/13518698/"
    "1cd204bc2c7a98b2490822ee6fc200ad/Stationspreisliste-2026-data.pdf"
)

# Imprint / address block at the bottom of each PDF page (tabulated across several rows).
_DB_STATION_FOOTER_ROW_RE = re.compile(
    r"(?i)"
    r"stand\s*:\s*\d{2}\.\d{2}\.\d{4}|"
    r"b\s*infra|"
    r"\bgo\s+ag\b|"
    r"personenbahnh|"
    r"uropapl\s+atz\s+1|"
    r"\bhnhöfe\b|"
    r"eschäf\s+tsbereich\b"
)
_DB_STATION_PAGE_NO_RE = re.compile(r"(?i)\bseite\s+\d+\b")

# Seven logical columns from the PDF grid (``Bemerkung`` is dropped).  The two amount
# columns both read “Anteil” / “Serviceeinrichtung” in the table; the PDF distinguishes
# them with the “Stationspreis SPNV” / “Stationspreis SPFV” line above the table.
_DB_STATION_CSV_COLUMNS = (
    "Bf-Nr",
    "Aufgabenträger",
    "Bahnhof",
    "klasse",
    "Bundesland",
    "Anteil Serviceeinrichtung Stationspreis SPNV",
    "Anteil Serviceeinrichtung Stationspreis SPFV",
)


def _db_station_row_join(row: list) -> str:
    parts: list[str] = []
    for v in row:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    return " ".join(parts)


def _db_station_is_column_index_row(row: list) -> bool:
    """First PDF row is 0,1,2,… column markers."""
    parts: list[str] = []
    for c in row:
        if c is None or (isinstance(c, float) and pd.isna(c)):
            continue
        s = str(c).strip()
        if not s:
            continue
        parts.append(s)
    if len(parts) < 8:
        return False
    try:
        nums = [int(p) for p in parts[:20]]
    except ValueError:
        return False
    return nums == list(range(len(nums)))


def _db_station_is_header_title_row(row: list) -> bool:
    for v in row:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if str(v).strip() == "Bf-Nr":
            return True
    return False


def _db_station_is_data_row(row: list) -> bool:
    """True when the first non-empty cell is numeric Bf-Nr and the next cell is an operator (not PLZ)."""
    vals: list[str] = []
    for v in row:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            vals.append("")
            continue
        vals.append(str(v).strip())
    i = 0
    while i < len(vals) and vals[i] == "":
        i += 1
    if i >= len(vals) or not vals[i].isdigit():
        return False
    if i + 1 >= len(vals) or not vals[i + 1]:
        return False
    second = vals[i + 1]
    # Footer PLZ line: 1 | 0557 B | erlin | …
    if second[0].isdigit():
        return False
    return True


def _normalize_db_station_table_rows(rows: list) -> list:
    """Keep one header block and all data rows; drop per-page footers and repeated headers.

    Each PDF page ends with the DB InfraGo imprint and starts again with the column header;
    ``extract_table`` concatenates those into one long grid, so we filter **globally**, not only
    at the end of the file.
    """
    out: list = []
    header_title_emitted = False
    expecting_first_header_continuation = False
    skipping_repeat_header = False

    for row in rows:
        joined = _db_station_row_join(row)
        if not joined:
            continue
        if _DB_STATION_FOOTER_ROW_RE.search(joined):
            continue
        if _DB_STATION_PAGE_NO_RE.search(joined):
            continue
        if _db_station_is_column_index_row(row):
            continue

        if skipping_repeat_header:
            if _db_station_is_data_row(row):
                skipping_repeat_header = False
                out.append(row)
            continue

        if _db_station_is_header_title_row(row):
            if not header_title_emitted:
                header_title_emitted = True
                expecting_first_header_continuation = True
                out.append(row)
            else:
                skipping_repeat_header = True
            continue

        if expecting_first_header_continuation:
            if _db_station_is_data_row(row):
                expecting_first_header_continuation = False
                out.append(row)
            else:
                out.append(row)
            continue

        out.append(row)

    return out


def _db_station_row_cells_clean(raw: list) -> list[str]:
    """Leading/trailing empties stripped; standalone € cells removed (pdfplumber artefact)."""
    cells: list[str] = []
    for v in raw:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            cells.append("")
        else:
            cells.append(str(v).strip())
    while cells and cells[0] == "":
        cells.pop(0)
    while cells and cells[-1] == "":
        cells.pop()
    return [c for c in cells if c != "€"]


def _parse_station_euro(val: object) -> float:
    """German decimal comma → float; returns NaN if not parseable."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return float("nan")
    t = str(val).strip().replace("€", "").strip()
    if not t or t in {"-", "—", "–"}:
        return float("nan")
    t = t.replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _db_station_trim_trailing_bemerkung(cells: list[str]) -> list[str]:
    """Drop trailing text cells (``Bemerkung``) so the last two fields are euro amounts."""
    c = list(cells)
    while len(c) >= 3:
        p_last = _parse_station_euro(c[-1])
        p_prev = _parse_station_euro(c[-2])
        if not np.isnan(p_last) and not np.isnan(p_prev):
            break
        c.pop()
    return c


def _db_station_data_cells_to_row(cells: list[str]) -> list | None:
    """Map one cleaned data row to seven fields; amounts as floats (NaN if missing)."""
    c = _db_station_trim_trailing_bemerkung(cells)
    if len(c) < 7:
        return None
    p2 = _parse_station_euro(c[-1])
    p1 = _parse_station_euro(c[-2])
    land = c[-3]
    klasse_s = c[-4]
    bahnhof = " ".join(x for x in c[2:-4] if x)
    bf_s, aufg = c[0], c[1]
    if not bahnhof or not land:
        return None
    try:
        bf_nr = int(bf_s)
    except ValueError:
        return None
    try:
        klasse = int(klasse_s)
    except ValueError:
        klasse = klasse_s
    return [bf_nr, aufg, bahnhof, klasse, land, p1, p2]


def _db_station_normalized_rows_to_dataframe(normalized_rows: list) -> pd.DataFrame:
    """Build the seven-column station table from pdfplumber-normalized grid rows."""
    records: list[list] = []
    for raw in normalized_rows:
        if not _db_station_is_data_row(raw):
            continue
        cells = _db_station_row_cells_clean(raw)
        rec = _db_station_data_cells_to_row(cells)
        if rec is not None:
            records.append(rec)
    df = pd.DataFrame(records, columns=list(_DB_STATION_CSV_COLUMNS))
    df["Bf-Nr"] = df["Bf-Nr"].astype("Int64")
    df["klasse"] = pd.to_numeric(df["klasse"], errors="coerce").astype("Int64")
    return df


# VBB GTFS: the official public download link.
# Override via the VBB_GTFS_URL environment variable or pass gtfs_url= explicitly.
_GTFS_DEFAULT_URL: str = os.environ.get(
    "VBB_GTFS_URL",
    "https://unternehmen.vbb.de/fileadmin/user_upload/VBB/Dokumente/API-Datensaetze/gtfs-2024.zip",
)

# ALKIS helper: static registry from GetCapabilities (2026-05); lazy-filled if empty.
# Each tuple: (service, qualified_layer_name, gpkg_layer_name, human_desc).
__ALKIS_LAYERS: list[tuple[str, str, str, str]] = [
    ("alkis", "alkis:bauwerkeflaechen", "bauwerkeflaechen", "ALKIS bauwerkeflaechen"),
    ("alkis", "alkis:bauwerkelinien", "bauwerkelinien", "ALKIS bauwerkelinien"),
    ("alkis", "alkis:besondereflurstuecksgrenzen", "besondereflurstuecksgrenzen", "ALKIS besondereflurstuecksgrenzen"),
    ("alkis", "alkis:bezirk", "bezirk", "ALKIS bezirk"),
    ("alkis", "alkis:festlegungenflaechen", "festlegungenflaechen", "ALKIS festlegungenflaechen"),
    ("alkis", "alkis:flur", "flur", "ALKIS flur"),
    ("alkis", "alkis:flurstuecke", "flurstuecke", "ALKIS flurstuecke"),
    ("alkis", "alkis:gebaeudeflaechen", "gebaeudeflaechen", "ALKIS gebaeudeflaechen"),
    ("alkis", "alkis:gebaeudelinien", "gebaeudelinien", "ALKIS gebaeudelinien"),
    ("alkis", "alkis:gemarkung", "gemarkung", "ALKIS gemarkung"),
    ("alkis", "alkis:gewaesservegetationflaechen", "gewaesservegetationflaechen", "ALKIS gewaesservegetationflaechen"),
    ("alkis", "alkis:gewaesservegetationlinien", "gewaesservegetationlinien", "ALKIS gewaesservegetationlinien"),
    ("alkis", "alkis:land", "land", "ALKIS land"),
    ("alkis", "alkis:ortsteile", "ortsteile", "ALKIS ortsteile"),
    ("alkis", "alkis:relief", "relief", "ALKIS relief"),
    ("alkis", "alkis:tatsaechlichenutzungflaechen", "tatsaechlichenutzungflaechen", "ALKIS tatsaechlichenutzungflaechen"),
    ("alkis", "alkis:vegetationpunkte", "vegetationpunkte", "ALKIS vegetationpunkte"),
]

# WFS layer tuples: (service, qualified_layer_name)
# All names verified against live GetCapabilities responses.
_ESIX_LAYER          = ("gssa_esix2022",     "gssa_esix2022:gssa_esix2022")
_MSS_LAYER           = ("mss_2025",          "mss_2025:mss2025_indizes_542")
_STADTSTRUKTUR_LAYER = ("ua_stadtstruktur",  "ua_stadtstruktur:b_stadtstruktur_differenziert_2024")
_GEBAEUDE_LAYER      = ("alkis_gebaeude",    "alkis_gebaeude:gebaeude")
_ZENTREN_FMA_LAYER   = ("step_zen_2040",     "step_zen_2040:step_zen_2040_fma")
_ZENTREN_ZH_LAYER    = ("step_zen_2040",     "step_zen_2040:step_zen_2040_zh")
_FNP_LAYER           = ("fnp_2025",          "fnp_2025:fnp_2025_vektor")
_BRW_LAYER           = ("brw2025",           "brw2025:brw_2025_vector")
_ALKIS_LAYER         = ("alkis_gebaeude",    "alkis_gebaeude:gebaeude")

# Buildings layer has 783 071 features → paginate to avoid a >400 MB JSON blob.
# GeoPackage format (binary, ~3× smaller than JSON) is used for non-paginated layers.
_LARGE_LAYER_THRESHOLD = 100_000
_GEBAEUDE_PAGE_SIZE = 100_000  # → 8 pages for 783 K buildings


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _try_tqdm():
    """Return the tqdm class, or None if not installed."""
    try:
        from tqdm import tqdm  # noqa: PLC0415
        return tqdm
    except ImportError:
        return None


def _wfs_url(
    service: str,
    layer: str,
    fmt: str = "geopackage",
    epsg: int = 25833,
    count: int | None = None,
    start_index: int = 0,
) -> str:
    """Build a GDI Berlin WFS 2.0.0 GetFeature URL."""
    url = (
        f"{_GDI_WFS_BASE.format(service=service)}"
        "?service=WFS&version=2.0.0&request=GetFeature"
        f"&typeNames={layer}"
        f"&outputFormat={fmt}"
        f"&srsName=EPSG:{epsg}"
    )
    if count is not None:
        url += f"&COUNT={count}&STARTINDEX={start_index}"
    return url


def _wfs_count(service: str, layer: str) -> int:
    """Query *numberMatched* for a WFS layer without downloading any features."""
    import urllib.request  # noqa: PLC0415

    url = (
        f"{_GDI_WFS_BASE.format(service=service)}"
        "?service=WFS&version=2.0.0&request=GetFeature"
        f"&typeNames={layer}&resultType=hits"
    )
    # Use system proxy (required for DNS in many network environments).
    # gdi.berlin.de is allowed by standard system proxies.
    req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode()
    m = re.search(r'numberMatched="(\d+)"', raw)
    return int(m.group(1)) if m else 0


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _stream_to_file(url: str, dest: Path, desc: str = "") -> None:
    """Download *url* → *dest*, showing progress in the terminal.

    Strategy (tried in order):
    1. ``curl`` — uses macOS SecureTransport / system PAC proxy evaluation,
       exactly the same networking path as Safari.  Always available on macOS.
    2. ``urllib.request`` with ``ProxyHandler({})`` — plain Python, no-proxy
       fallback for environments where curl is unavailable.

    The browser User-Agent is sent in both cases so that servers that inspect
    the ``User-Agent`` header (e.g. dbinfrago.com, vbb.de) respond correctly.
    """
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    dest.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("curl"):
        label = desc or dest.name
        logger.info("Downloading %s via curl …", label)
        cmd = [
            "curl",
            "--location",           # follow redirects
            "--fail",               # non-zero exit on HTTP errors
            "--retry", "3",
            "--retry-delay", "2",
            "--connect-timeout", "30",
            "--max-time", "3600",
            "--progress-bar",       # show native curl progress in terminal
            "-A", _BROWSER_UA,
            "-o", str(dest),
            url,
        ]
        subprocess.run(cmd, check=True)
        return

    # ── fallback: urllib.request (no system proxy) ──────────────────────────
    tqdm = _try_tqdm()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _BROWSER_UA, "Accept": "*/*"},
    )
    with opener.open(req, timeout=600) as r:
        total_hdr = r.headers.get("Content-Length")
        total = int(total_hdr) if total_hdr else None
        # tqdm raises TypeError on `if bar:` when total=None — use `is not None`.
        bar = (
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc or dest.name,
            )
            if tqdm is not None
            else None
        )
        with dest.open("wb") as fh:
            while True:
                blk = r.read(1 << 17)
                if not blk:
                    break
                fh.write(blk)
                if bar is not None:
                    bar.update(len(blk))
        if bar is not None:
            bar.close()


def _read_wfs_gpkg(service: str, layer: str, desc: str = "") -> gpd.GeoDataFrame:
    """Download a WFS layer in GeoPackage format to a temp file, return GeoDataFrame.

    GeoPackage is the binary format supported by GDI Berlin — roughly 3× smaller
    than GeoJSON, which means ~3× faster network transfer and no JSON parsing cost.
    """
    import tempfile  # noqa: PLC0415

    url = _wfs_url(service, layer, fmt="geopackage")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "layer.gpkg"
        _stream_to_file(url, tmp, desc=desc or layer)
        return gpd.read_file(tmp)


def _read_wfs_paged(
    service: str,
    layer: str,
    desc: str = "",
    page_size: int = _GEBAEUDE_PAGE_SIZE,
) -> gpd.GeoDataFrame:
    """Download a large WFS layer using paginated GeoJSON requests.

    Uses ``COUNT`` + ``STARTINDEX`` WFS paging.  Each page is read directly
    by geopandas.  A tqdm progress bar shows which page is being fetched.
    """
    tqdm = _try_tqdm()

    n = _wfs_count(service, layer)
    label = desc or layer
    logger.info("%s: %d features, page size %d.", label, n, page_size)

    if n == 0:
        logger.warning("%s: count unavailable — falling back to single GeoPackage request.", label)
        return _read_wfs_gpkg(service, layer, desc=label)

    n_pages = (n + page_size - 1) // page_size
    pages: range | object = range(n_pages)
    if tqdm:
        pages = tqdm(pages, desc=label, unit="page", total=n_pages)

    import tempfile  # noqa: PLC0415

    gdfs: list[gpd.GeoDataFrame] = []
    for page in pages:
        url = _wfs_url(
            service,
            layer,
            fmt="application/json",
            count=page_size,
            start_index=page * page_size,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / "page.json"
            _stream_to_file(url, tmp, desc=f"p{page + 1}/{n_pages}")
            gdfs.append(gpd.read_file(tmp))

    if not gdfs:
        raise RuntimeError(f"No features downloaded for WFS layer {layer!r}.")

    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)


def _read_wfs(
    service: str,
    layer: str,
    desc: str = "",
    large_threshold: int = _LARGE_LAYER_THRESHOLD,
    page_size: int = _GEBAEUDE_PAGE_SIZE,
) -> gpd.GeoDataFrame:
    """Download a WFS layer, choosing the optimal strategy automatically.

    * Layers with ≤ *large_threshold* features → single GeoPackage request
      (binary, compact, progress shown as bytes).
    * Layers with > *large_threshold* features → paginated GeoJSON requests
      (avoids a single >400 MB blob, progress shown per page).
    """
    n = _wfs_count(service, layer)
    label = desc or layer
    logger.info("%s: %d features detected.", label, n)

    if n > large_threshold:
        return _read_wfs_paged(service, layer, desc=label, page_size=page_size)
    return _read_wfs_gpkg(service, layer, desc=label)


def _fetch_alkis_layers() -> list[tuple[str, str, str, str]]:
    """Discover ALKIS feature types from WFS GetCapabilities."""
    import urllib.request  # noqa: PLC0415
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    url = (
        f"{_GDI_WFS_BASE.format(service='alkis')}"
        "?request=GetCapabilities&service=WFS"
    )
    req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()

    root = ET.fromstring(raw)
    qualified_names: list[str] = []
    for ns in (
        "{http://www.opengis.net/wfs/2.0}",
        "{http://www.opengis.net/wfs}",
    ):
        for ft in root.findall(f".//{ns}FeatureType"):
            name_el = ft.find(f"{ns}Name")
            if name_el is not None and name_el.text:
                qualified_names.append(name_el.text.strip())
        if qualified_names:
            break

    layers: list[tuple[str, str, str, str]] = []
    for qualified in sorted(set(qualified_names)):
        gpkg_name = re.sub(
            r"[^a-z0-9]+",
            "_",
            qualified.split(":", 1)[-1].lower(),
        ).strip("_")
        layers.append(("alkis", qualified, gpkg_name, f"ALKIS {gpkg_name}"))

    logger.info("Discovered %d ALKIS layers.", len(layers))
    return layers


# ---------------------------------------------------------------------------
# Employee range parser
# ---------------------------------------------------------------------------

def _parse_employees_range(range_str: str) -> float:
    if pd.isna(range_str):
        return np.nan
    if range_str == "unbekannt":
        return np.nan
    try:
        parts = range_str.split("-")
        if len(parts) == 2:
            low = int(parts[0].strip())
            high = int(parts[1].strip().split()[0])
            return float((low + high) / 2)
        return float(range_str.strip().split()[0])
    except Exception as e:
        logger.error("Error parsing '%s': %s", range_str, e)
        return np.nan


# ---------------------------------------------------------------------------
# Public download functions
# ---------------------------------------------------------------------------

def download_index_data(skip_if_exists: bool = True) -> None:
    """Download ESIx 2022 and MSS 2023 social-structure indices from GDI Berlin.

    Saves
    -----
    ``data/raw/esix.gpkg`` (layer ``esix``)
    ``data/raw/mss.gpkg``  (layer ``mss``)

    Parameters
    ----------
    skip_if_exists:
        If *True* (default) and the output file already exists, skip the
        download for that layer.
    """
    logger.info("Downloading ESIx and MSS index data.")

    esix_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "esix.gpkg"
    mss_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "mss.gpkg"

    if skip_if_exists and esix_path.exists():
        logger.info("ESIx already exists at %s — skipping.", esix_path)
    else:
        gdf_esix = _read_wfs(*_ESIX_LAYER, desc="ESIx 2022")
        esix_path.parent.mkdir(parents=True, exist_ok=True)
        gdf_esix.to_file(esix_path, driver="GPKG", layer="esix")
        logger.info("ESIx saved → %s", esix_path)

    if skip_if_exists and mss_path.exists():
        logger.info("MSS already exists at %s — skipping.", mss_path)
    else:
        gdf_mss = _read_wfs(*_MSS_LAYER, desc="MSS 2023")
        mss_path.parent.mkdir(parents=True, exist_ok=True)
        gdf_mss.to_file(mss_path, driver="GPKG", layer="mss")
        logger.info("MSS saved → %s", mss_path)


def download_stadtstruktur(skip_if_exists: bool = True) -> None:
    """Download urban-structure, buildings and city-centre layers from GDI Berlin.

    Saves
    -----
    ``data/raw/stadtstruktur.gpkg`` (layer ``stadtstruktur``)
    ``data/raw/gebaeude.gpkg``      (layer ``gebaeude``)   ← 783 K features, slow
    ``data/raw/zentren.gpkg``       (layers ``zentren_fma``, ``zentren_zh``)

    Parameters
    ----------
    skip_if_exists:
        If *True* (default) and the output file already exists, skip that layer.

    Notes
    -----
    The ``alkis_gebaeude:gebaeude`` layer contains **783 071** individual building
    footprints.  Download is automatically split into pages of
    ``_GEBAEUDE_PAGE_SIZE`` features to avoid a single >400 MB JSON response.
    Expect several minutes for the buildings layer even on a fast connection.
    """
    stadtstruktur_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "stadtstruktur.gpkg"
    gebaeude_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "gebaeude.gpkg"
    zentren_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "zentren.gpkg"

    if skip_if_exists and stadtstruktur_path.exists():
        logger.info("Stadtstruktur already exists — skipping.")
    else:
        gdf = _read_wfs(*_STADTSTRUKTUR_LAYER, desc="Stadtstruktur 2024")
        stadtstruktur_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(stadtstruktur_path, driver="GPKG", layer="stadtstruktur")
        logger.info("Stadtstruktur saved → %s", stadtstruktur_path)

    if skip_if_exists and gebaeude_path.exists():
        logger.info("Gebäude already exists — skipping.")
    else:
        logger.info(
            "Downloading ALKIS buildings (783 K features, page size %d) …",
            _GEBAEUDE_PAGE_SIZE,
        )
        gdf = _read_wfs(*_GEBAEUDE_LAYER, desc="Gebäude (ALKIS)")
        gebaeude_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(gebaeude_path, driver="GPKG", layer="gebaeude")
        logger.info("Gebäude saved → %s", gebaeude_path)

    zentren_fma_done = skip_if_exists and zentren_path.exists()
    if zentren_fma_done:
        logger.info("Zentren already exists — skipping FMA and ZH layers.")
    else:
        gdf_fma = _read_wfs(*_ZENTREN_FMA_LAYER, desc="Zentren FMA")
        zentren_path.parent.mkdir(parents=True, exist_ok=True)
        gdf_fma.to_file(zentren_path, driver="GPKG", layer="zentren_fma")
        logger.info("Zentren FMA saved → %s", zentren_path)

        gdf_zh = _read_wfs(*_ZENTREN_ZH_LAYER, desc="Zentren ZH")
        gdf_zh.to_file(zentren_path, driver="GPKG", layer="zentren_zh")
        logger.info("Zentren ZH saved → %s", zentren_path)

def download_station_data(
    skip_if_exists: bool = True,
    gtfs_url: str | None = None,
) -> None:
    """Download DB station price list (PDF→CSV) and VBB GTFS stop data.

    The CSV has seven columns matching the DB InfraGo PDF table (``Bemerkung`` omitted):
    ``Bf-Nr``, ``Aufgabenträger``, ``Bahnhof``, ``klasse``, ``Bundesland``, and two
    ``Anteil Serviceeinrichtung`` amounts distinguished as *Stationspreis SPNV* vs *SPFV*
    in the document.  Euro symbols from the PDF grid are not written; amounts are floats
    with ``.`` as the decimal separator.

    Parameters
    ----------
    skip_if_exists:
        Skip any file that already exists on disk.
    gtfs_url:
        Direct URL to a GTFS ``.zip`` file.  If *None*, uses the
        ``VBB_GTFS_URL`` environment variable, or falls back to the
        Germany-wide GTFS from ``gtfs.de`` (~250 MB, includes all VBB stops).

        **To use VBB-specific GTFS** (smaller, ~30 MB):
        Register at https://www.vbb.de/vbb-services/api-entwicklerinfos/
        to obtain a personal download link, then pass it here or export it
        as ``VBB_GTFS_URL``.

    Notes
    -----
    The DB InfraGo PDF is parsed with ``pdfplumber``; install it with::

        pip install pdfplumber
    """
    try:
        import pdfplumber  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Missing optional dependency 'pdfplumber' required by download_station_data. "
            "Install it with: pip install pdfplumber"
        ) from exc

    # ── DB InfraGo station price list ──────────────────────────────────────
    db_pdf_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "db_station_data.pdf"
    db_csv_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "db_station_data.csv"

    if skip_if_exists and db_csv_path.exists():
        logger.info("DB station CSV already exists — skipping PDF download.")
    else:
        db_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading DB InfraGo station price list PDF …")
        _stream_to_file(_DB_STATION_PDF_URL, db_pdf_path, desc="DB Stationspreisliste")

        with pdfplumber.open(db_pdf_path) as pdf:
            all_data = []
            for page in pdf.pages:
                table = page.extract_table(
                    table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_y_tolerance": 5,
                    }
                )
                if table:
                    all_data.extend(table)

        station_df = _db_station_normalized_rows_to_dataframe(
            _normalize_db_station_table_rows(all_data)
        )
        station_df.to_csv(db_csv_path, index=False)
        logger.info("DB station data saved → %s", db_csv_path)

    # ── VBB GTFS ────────────────────────────────────────────────────────────
    gtfs_dir = Path(__file__).resolve().parents[3] / "data" / "raw" / "gtfs"
    stops_file = gtfs_dir / "stops.txt"

    if skip_if_exists and stops_file.exists():
        logger.info("GTFS stops already exist at %s — skipping.", gtfs_dir)
    else:
        url = gtfs_url or _GTFS_DEFAULT_URL
        gtfs_zip = Path(__file__).resolve().parents[3] / "data" / "raw" / "gtfs.zip"

        logger.info("Downloading GTFS from %s …", url)
        gtfs_zip.parent.mkdir(parents=True, exist_ok=True)
        _stream_to_file(url, gtfs_zip, desc="GTFS")

        logger.info("Extracting GTFS → %s …", gtfs_dir)
        gtfs_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(gtfs_zip) as zf:
            zf.extractall(gtfs_dir)

        gtfs_zip.unlink()
        logger.info("GTFS extracted → %s", gtfs_dir)

def download_fnp_data(skip_if_exists: bool = True) -> None:
    """Download the FNP 2025 data from the Berlin Geoportal.
    
    Saves
    -----
    ``data/raw/fnp.gpkg`` (layer ``fnp_2025``)

    Parameters
    ----------
    skip_if_exists:
        If *True* (default) and the output file already exists, skip the
        download for that layer.
    """
    fnp_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "fnp_2025.gpkg"
    
    if skip_if_exists and fnp_path.exists():
        logger.info("FNP 2025 already exists at %s — skipping.", fnp_path)
    else:
        gdf = _read_wfs(*_FNP_LAYER, desc="FNP 2025")
        fnp_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(fnp_path, driver="GPKG", layer="fnp_2025")
        logger.info("FNP 2025 saved → %s", fnp_path)

def download_brw_data(skip_if_exists: bool = True) -> None:
    """Download the BRW data from the Berlin Senate of Finance."""
    brw_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "brw_2025.gpkg"
    if skip_if_exists and brw_path.exists():
        logger.info("BRW already exists at %s — skipping.", brw_path)
    else:
        gdf = _read_wfs(*_BRW_LAYER, desc="BRW 2025")
        brw_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(brw_path, driver="GPKG", layer="brw_2025")


def download_alkis_data(skip_if_exists: bool = True) -> None:
    """Download ALL ALKIS layers from the GDI Berlin WFS and save to a single GeoPackage.

    Saves
    -----
    ``data/raw/alkis_full.gpkg``
        Multi-layer GeoPackage; one internal layer per ALKIS feature type.
        Layer names are the unqualified, normalised FeatureType names
        (e.g. ``ax_gebaeude``, ``ax_flurstueck``).

    Parameters
    ----------
    skip_if_exists:
        If *True* (default), skip layers already present inside the output
        GeoPackage (checked per layer via ``fiona.listlayers``).

    Notes
    -----
    The ALKIS WFS endpoint is ``https://gdi.berlin.de/services/wfs/alkis``.
    Layers are discovered dynamically from GetCapabilities at first call if
    ``__ALKIS_LAYERS`` is empty; otherwise the static registry is used.

    Some layers may be large (``ax_gebaeude`` has ~783 K features).
    ``_read_wfs`` handles pagination automatically; expect several minutes
    for large layers.

    CRS is EPSG:25833 (GDI Berlin native); reproject with ``.to_crs()`` if needed.
    """
    global __ALKIS_LAYERS  # noqa: PLW0603

    if not __ALKIS_LAYERS:
        __ALKIS_LAYERS = _fetch_alkis_layers()

    import fiona  # noqa: PLC0415

    alkis_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "alkis_full.gpkg"
    alkis_path.parent.mkdir(parents=True, exist_ok=True)

    existing_layers: set[str] = set()
    if skip_if_exists and alkis_path.exists():
        existing_layers = set(fiona.listlayers(alkis_path))

    attempted = 0
    downloaded = 0
    skipped = 0

    for service, qualified_layer_name, gpkg_layer_name, human_desc in __ALKIS_LAYERS:
        attempted += 1
        if skip_if_exists and gpkg_layer_name in existing_layers:
            logger.info(
                "ALKIS layer %s already in %s — skipping.",
                gpkg_layer_name,
                alkis_path,
            )
            skipped += 1
            continue

        try:
            gdf = _read_wfs(service, qualified_layer_name, desc=human_desc)
            gdf.to_file(alkis_path, driver="GPKG", layer=gpkg_layer_name)
            logger.info(
                "ALKIS %s saved (%d features) → %s",
                gpkg_layer_name,
                len(gdf),
                alkis_path,
            )
            downloaded += 1
        except Exception as exc:
            logger.error(
                "ALKIS layer %s (%s) failed: %s",
                gpkg_layer_name,
                qualified_layer_name,
                exc,
            )

    logger.info(
        "ALKIS download complete: %d attempted, %d downloaded, %d skipped.",
        attempted,
        downloaded,
        skipped,
    )


def download_IHK_data() -> None:
    """Download the IHK data from the Berlin Chamber of Commerce."""
    raise NotImplementedError(
        "IHK data must be downloaded manually — it is not publicly accessible."
    )


def download_medianeinkommen_data() -> None:
    """Download the Medianeinkommen data from the Berlin Senate of Finance."""
    raise NotImplementedError(
        "Medianeinkommen data must be downloaded manually."
    )

# ---------------------------------------------------------------------------
# Processing stubs (not yet implemented)
# ---------------------------------------------------------------------------

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
        ``grid`` with an ``empl`` column (float, 0 for cells with no businesses).

    Notes
    -----
    IHK data cannot be downloaded automatically.  Place the file at
    ``data/raw/2023_12_IHK_Berlin_Gewerbedaten.csv`` before calling this
    function.
    """
    logger.info("Loading IHK data from %s.", ihk_path)
    ihk = pd.read_csv(ihk_path)

    ihk_gdf = gpd.GeoDataFrame(
        ihk,
        geometry=gpd.points_from_xy(ihk["longitude"], ihk["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3035")
    ihk_gdf["empl"] = ihk_gdf["employees_range"].apply(_parse_employees_range)

    grid_indexed = grid.copy()
    grid_indexed["_grid_idx"] = grid_indexed.index
    joined = gpd.sjoin(
        ihk_gdf[["geometry", "empl"]],
        grid_indexed[["_grid_idx", "geometry"]],
        how="left",
        predicate="within",
    )
    empl_by_cell = joined.groupby("_grid_idx")["empl"].sum().rename("empl").reset_index()

    out = grid.copy()
    out["_grid_idx"] = out.index
    out = out.merge(empl_by_cell, on="_grid_idx", how="left")
    out["empl"] = out["empl"].fillna(0.0)
    out = out.drop(columns=["_grid_idx"])
    logger.info(
        "IHK employment joined: total=%.0f, cells with empl>0=%d.",
        out["empl"].sum(), (out["empl"] > 0).sum(),
    )
    return out


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
    esix_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "esix.gpkg"
    mss_path  = Path(__file__).resolve().parents[3] / "data" / "raw" / "mss.gpkg"
    if not esix_path.exists():
        raise FileNotFoundError(
            f"ESIx GeoPackage not found at {esix_path}. "
            "Run download_index_data() first."
        )
    if not mss_path.exists():
        raise FileNotFoundError(
            f"MSS GeoPackage not found at {mss_path}. "
            "Run download_index_data() first."
        )
    esix = gpd.read_file(esix_path).to_crs("EPSG:3035")
    mss  = gpd.read_file(mss_path).to_crs("EPSG:3035")

    def _first_numeric_col(gdf: gpd.GeoDataFrame) -> str | None:
        for col in gdf.columns:
            if col == "geometry":
                continue
            if pd.api.types.is_numeric_dtype(gdf[col]):
                return col
        return None

    out = grid.copy()
    out["_idx"] = out.index

    esix_col = _first_numeric_col(esix)
    if esix_col:
        j = gpd.sjoin(
            out[["_idx", "geometry"]],
            esix[["geometry", esix_col]],
            how="left", predicate="intersects",
        )
        out = out.join(
            j.groupby("_idx")[esix_col].mean().rename("esix_score"),
            on="_idx", how="left",
        )
        logger.info("ESIx joined (col=%s).", esix_col)
    else:
        out["esix_score"] = float("nan")
        logger.warning("No numeric column in ESIx layer; esix_score=NaN.")

    mss_col = _first_numeric_col(mss)
    if mss_col:
        j = gpd.sjoin(
            out[["_idx", "geometry"]],
            mss[["geometry", mss_col]],
            how="left", predicate="intersects",
        )
        out = out.join(
            j.groupby("_idx")[mss_col].mean().rename("mss_score"),
            on="_idx", how="left",
        )
        logger.info("MSS joined (col=%s).", mss_col)
    else:
        out["mss_score"] = float("nan")
        logger.warning("No numeric column in MSS layer; mss_score=NaN.")

    return out.drop(columns=["_idx"])


def identify_transport_hubs(
    grid: gpd.GeoDataFrame,
    gtfs_dir: Path | None = None,
) -> gpd.GeoDataFrame:
    """Flag grid cells that contain or are adjacent to major transit nodes.

    Parses VBB GTFS ``stops.txt`` from *gtfs_dir* (default: ``data/raw/gtfs/``),
    classifies stops by route type / service frequency, and adds columns to
    *grid* indicating transit accessibility.

    Expected output columns added to *grid*:

    * ``transit_stops``  — number of transit stops within the cell (int)
    * ``is_transit_hub`` — True if a major hub (S/U-Bahn interchange) is
                           present (bool)

    Parameters
    ----------
    grid:
        Population grid GeoDataFrame with polygon geometry in EPSG:3035.
    gtfs_dir:
        Directory containing unpacked GTFS files.  Defaults to
        ``data/raw/gtfs/``.

    Returns
    -------
    geopandas.GeoDataFrame
        *grid* with ``transit_stops`` and ``is_transit_hub`` columns.
    """
    import re
    import unicodedata
    from hotelling.spatial.osm import fetch_pois  # lazy import

    db_csv_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "db_station_data.csv"

    # ── OSM stations ──────────────────────────────────────────────────────────
    logger.info("Loading OSM station POIs.")
    osm_stations = fetch_pois(type="stations", city="Berlin").to_crs(grid.crs)
    osm_stations = osm_stations.copy()
    osm_stations["geometry"] = osm_stations.geometry.centroid

    # ── Spatial join stations → grid ─────────────────────────────────────────
    grid_indexed = grid.copy()
    grid_indexed["_grid_idx"] = grid_indexed.index

    joined = gpd.sjoin(
        osm_stations[["geometry", "name"]],
        grid_indexed[["_grid_idx", "geometry"]],
        how="left", predicate="within",
    )
    station_counts = joined.groupby("_grid_idx").size().rename("station_count")
    station_names  = joined.groupby("_grid_idx")["name"].apply(list).rename("station_names")

    out = grid.copy()
    out["_grid_idx"] = out.index
    out = out.merge(station_counts, left_on="_grid_idx", right_index=True, how="left")
    out = out.merge(station_names,  left_on="_grid_idx", right_index=True, how="left")
    out["station_count"]  = out["station_count"].fillna(0).astype(int)
    out["station_names"]  = out["station_names"].apply(lambda x: x if isinstance(x, list) else [])
    out["station_class"]  = float("nan")
    out["matched_db_station"] = None
    out = out.drop(columns=["_grid_idx"])

    # ── DB station class matching ─────────────────────────────────────────────
    if not db_csv_path.exists():
        logger.warning(
            "DB station CSV not found at %s — station_class will be NaN.", db_csv_path
        )
        return out

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower()
        s = re.sub(r"\(.*?\)", "", s)
        s = re.sub(r"^(s|u)\s+", "", s)
        s = re.sub(r"^berlin[\s\-]+", "", s)
        s = s.replace("ß", "ss")
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    _OVERRIDES: dict[str, str] = {
        "Schöneweide":        "Berlin-Schöneweide Pbf",
        "Berlin-Schöneweide": "Berlin-Schöneweide Pbf",
        "Wittenau":           "Berlin-Wittenau (Wilhelmsruher Damm)",
    }

    db_stations = pd.read_csv(db_csv_path)
    db_stations = db_stations[db_stations["Bundesland"] == "Berlin"].copy()
    db_stations["Bahnhof"] = db_stations["Bahnhof"].apply(
        lambda n: n[4:] if isinstance(n, str) and n.startswith("lin ") else n
    )

    db_lookup: dict[str, str] = {}
    for name in db_stations["Bahnhof"].dropna().unique():
        db_lookup.setdefault(_norm(name), name)

    osm_names = osm_stations["name"].dropna().unique()
    osm_to_db: dict[str, str | None] = {
        n: _OVERRIDES.get(n, db_lookup.get(_norm(n))) for n in osm_names
    }

    for idx, row in out.iterrows():
        if row["station_count"] > 0:
            db_names: set[str] = set()
            for osm_name in row["station_names"]:
                db_name = osm_to_db.get(osm_name)
                if db_name:
                    db_names.add(db_name)
            if len(db_names) == 1:
                db_name = next(iter(db_names))
                out.at[idx, "matched_db_station"] = db_name
                db_info = db_stations[db_stations["Bahnhof"] == db_name]
                if not db_info.empty:
                    out.at[idx, "station_class"] = int(db_info["klasse"].iloc[0])

    out_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "grid_with_stations.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info(
        "Transport hubs identified: %d cells with stations, %d with DB class. Saved to %s.",
        (out["station_count"] > 0).sum(), out["station_class"].notna().sum(), out_path,
    )
    return out


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
    import warnings
    warnings.warn(
        "identify_cbd() is deprecated and will be removed. "
        "CBD concept retired in ADR-018. Use run_prime_location_clustering() "
        "for employment-dense prime locations, and add_lcc_layer() for "
        "local commercial centres (OSM-extracted).",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning("identify_cbd() called but is deprecated (ADR-018). Returning grid unchanged.")
    return grid


def process_gebaeude_stadtstruktur(
    gebaeude_path: Path | None = None,
    stadtstruktur_path: Path | None = None,
    ihk_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Build the enriched building-level GeoDataFrame used by the AABPL pipeline.

    Joins ALKIS building footprints (``gebaeude.gpkg``) with the urban-structure
    classification layer (``stadtstruktur.gpkg``), applies ALKIS GFK-derived
    floor-space efficiency factors and employee hard caps from
    :mod:`hotelling.spatial.gebaeude_capacity`, then maps IHK business-microdata
    geocoordinates to their nearest building via an STRtree query and enforces
    the physical hard cap.

    Saved output: ``data/processed/gebaeude_stadtstruktur.parquet``.

    Parameters
    ----------
    gebaeude_path:
        Path to ``gebaeude.gpkg``.  Default: ``data/raw/gebaeude.gpkg``.
    stadtstruktur_path:
        Path to ``stadtstruktur.gpkg``.  Default: ``data/raw/stadtstruktur.gpkg``.
    ihk_path:
        Path to the IHK CSV.  If ``None`` or absent, ``empl`` and
        ``approx_empl`` default to 0.

    Returns
    -------
    geopandas.GeoDataFrame
        Building polygons enriched with columns: ``gfk``, ``aog``, ``hoh``,
        ``nutzung``, ``efficiency``, ``usable_area_m2``, ``employee_hard_cap``,
        ``empl``, ``approx_empl``.  CRS matches the source files.

    Raises
    ------
    FileNotFoundError
        If ``gebaeude_path`` or ``stadtstruktur_path`` do not exist.
    """
    from hotelling.spatial.gebaeude_capacity import (
        compute_employee_hard_cap,
        compute_usable_floor_area,
        get_efficiency_factor,
    )

    _geb_path = gebaeude_path    or Path(__file__).resolve().parents[3] / "data" / "raw" / "gebaeude.gpkg"
    _ss_path  = stadtstruktur_path or Path(__file__).resolve().parents[3] / "data" / "raw" / "stadtstruktur.gpkg"
    _ihk      = ihk_path or Path(__file__).resolve().parents[3] / "data" / "raw" / "2023_12_IHK_Berlin_Gewerbedaten.csv"

    for p in (_geb_path, _ss_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run download_stadtstruktur() first."
            )

    # ── 1. Load ───────────────────────────────────────────────────────────────
    logger.info("Loading gebaeude and stadtstruktur.")
    gebaeude      = gpd.read_file(_geb_path)
    stadtstruktur = gpd.read_file(_ss_path)

    filtered = (
        gebaeude[gebaeude["bezeich"] == "AX_Gebaeude"]
        .copy().reset_index(drop=True)
    )
    filtered["_bpos"] = filtered.index

    ss = stadtstruktur.copy().reset_index(drop=True)
    ss["_spos"] = ss.index

    # ── 2. Intersects sjoin ───────────────────────────────────────────────────
    logger.info("Joining %d buildings to stadtstruktur via intersects.", len(filtered))
    sjoin_res = gpd.sjoin(
        filtered[["_bpos", "geometry"]],
        ss[["_spos", "geometry"]],
        how="left", predicate="intersects",
    )

    # ── 3. Tie-break: keep the stadtstruktur polygon with largest intersection ─
    matched = sjoin_res[sjoin_res["_spos"].notna()].copy()
    if not matched.empty:
        bgeoms = filtered.loc[matched.index, "geometry"].values
        sgeoms = ss.loc[matched["_spos"].astype(int).values, "geometry"].values
        matched["_iarea"] = np.array([b.intersection(s).area for b, s in zip(bgeoms, sgeoms)])
        best = (
            matched.sort_values("_iarea", ascending=False)
            .drop_duplicates(subset=["_bpos"])
        )
    else:
        best = matched

    unmatched = sjoin_res[~sjoin_res["_bpos"].isin(best["_bpos"]) & sjoin_res["_spos"].isna()]
    unmatched_dedup = unmatched.drop_duplicates(subset=["_bpos"])
    combined = pd.concat([best, unmatched_dedup], ignore_index=True)

    # ── 4. Merge stadtstruktur attributes onto buildings ──────────────────────
    ss_attrs = ss.drop(columns=["geometry"])
    enriched = filtered.merge(
        combined[["_bpos", "_spos"]].merge(ss_attrs, left_on="_spos", right_on="_spos", how="left"),
        on="_bpos", how="left",
    ).drop(columns=["_bpos", "_spos"], errors="ignore")

    # ── 5. AOG imputation for unmatched buildings ─────────────────────────────
    _BEZGFK_TO_AOG: dict[str, int] = {
        "Tiefgarage": 0, "Garage": 0, "Gebäude zum Parken": 0,
        "Umformer": 1, "Schutzbunker": 0, "Heizwerk": 1,
        "Gebäude zur Elektrizitätsversorgung": 1, "Parkdeck": 0,
        "Speichergebäude": 1, "Gebäude für Vorratshaltung": 1,
        "Pumpstation": 0, "Lagerhalle, Lagerschuppen, Lagerhaus": 1,
        "Gebäude zum Sportplatz": 1, "Sport-, Turnhalle": 1,
        "Gebäude für Sportzwecke": 1, "Wasserbehälter": 0,
        "Pumpwerk (nicht für Wasserversorgung)": 0,
        "Gebäude zur Wasserversorgung": 1, "Gebäude zur Abwasserbeseitigung": 1,
        "Hallenbad": 1, "Schuppen": 1, "Gartenhaus": 1, "Wasserwerk": 1,
        "Gebäude zur Abfallbehandlung": 1, "Gebäude für Land- und Forstwirtschaft": 1,
        "Bootshaus": 1, "Tierschauhaus": 1, "Müllbunker": 1, "Parkhaus": 1,
    }
    if "aog" in enriched.columns and "bezgfk" in enriched.columns:
        null_mask = enriched["aog"].isna()
        enriched.loc[null_mask, "aog"] = enriched.loc[null_mask, "bezgfk"].map(_BEZGFK_TO_AOG)
        enriched = enriched[~enriched["aog"].isna()].copy()

    # ── 6. HOH bool conversion ────────────────────────────────────────────────
    if "hoh" in enriched.columns:
        enriched["hoh"] = enriched["hoh"].apply(
            lambda x: False
            if (x == "false" or x == "" or x is None
                or (isinstance(x, float) and np.isnan(x)))
            else True
        )

    # ── 7. Efficiency, usable area, hard cap ──────────────────────────────────
    logger.info("Computing floor-space efficiency and employee hard caps.")
    gfk_col  = "gfk"   if "gfk"   in enriched.columns else None
    hoh_col  = "hoh"   if "hoh"   in enriched.columns else None
    aog_col  = "aog"   if "aog"   in enriched.columns else None
    area_col = "shape_area" if "shape_area" in enriched.columns else None

    def _area(row: pd.Series) -> float:
        return row[area_col] if area_col else row.geometry.area

    if gfk_col:
        enriched["efficiency"] = enriched.apply(
            lambda r: get_efficiency_factor(r[gfk_col], bool(r[hoh_col]) if hoh_col else False), axis=1
        )
        enriched["usable_area_m2"] = enriched.apply(
            lambda r: compute_usable_floor_area(
                _area(r), r[aog_col] if aog_col else None,
                r[gfk_col], bool(r[hoh_col]) if hoh_col else False,
            ), axis=1
        )
        enriched["employee_hard_cap"] = enriched.apply(
            lambda r: compute_employee_hard_cap(
                _area(r), r[aog_col] if aog_col else None,
                r[gfk_col], bool(r[hoh_col]) if hoh_col else False,
            ), axis=1
        )
    else:
        enriched["efficiency"] = float("nan")
        enriched["usable_area_m2"] = float("nan")
        enriched["employee_hard_cap"] = float("nan")

    enriched = enriched.reset_index(drop=True)
    enriched["empl"]       = 0.0
    enriched["approx_empl"] = 0.0

    # ── 8. IHK nearest-building matching ─────────────────────────────────────
    if _ihk.exists():
        logger.info("Matching IHK employment to buildings via STRtree nearest.")
        ihk_raw = pd.read_csv(_ihk)
        ihk_gdf = gpd.GeoDataFrame(
            ihk_raw,
            geometry=gpd.points_from_xy(ihk_raw["longitude"], ihk_raw["latitude"]),
            crs="EPSG:4326",
        ).to_crs(enriched.crs)
        ihk_gdf["empl"] = ihk_gdf["employees_range"].apply(_parse_employees_range)

        # Deduplicate by (lon, lat), sum employment per unique location
        ihk_gdf["_ll"] = list(zip(ihk_raw["longitude"], ihk_raw["latitude"]))
        empl_by_loc = (
            ihk_gdf.groupby("_ll")
            .agg(empl=("empl", "sum"), geometry=("geometry", "first"))
            .reset_index(drop=True)
        )
        empl_by_loc = gpd.GeoDataFrame(empl_by_loc, geometry="geometry", crs=enriched.crs)

        bldg_geoms   = enriched.geometry.values
        tree         = STRtree(bldg_geoms)
        pt_geoms     = empl_by_loc.geometry.values
        nearest_idxs = tree.nearest(pt_geoms)
        distances    = np.array([pt.distance(bldg_geoms[i]) for pt, i in zip(pt_geoms, nearest_idxs)])

        empl_by_loc["_bldg_idx"] = nearest_idxs
        empl_by_loc["_dist"]     = distances
        empl_by_loc = empl_by_loc[empl_by_loc["_dist"] <= 500.0].copy()

        empl_by_bldg = empl_by_loc.groupby("_bldg_idx")["empl"].sum()
        enriched["empl"] = enriched.index.map(empl_by_bldg).fillna(0.0)

        # Apply hard cap: approx_empl = min(empl, employee_hard_cap)
        cap = enriched["employee_hard_cap"].values
        enriched["approx_empl"] = np.minimum(
            enriched["empl"].values,
            np.where(np.isinf(cap), enriched["empl"].values, cap),
        )
        logger.info(
            "IHK mapped: total approx_empl=%.0f, buildings>0=%d.",
            enriched["approx_empl"].sum(), (enriched["approx_empl"] > 0).sum(),
        )
    else:
        logger.warning("IHK path %s absent; empl/approx_empl=0.", _ihk)

    # ── 9. Save and return ────────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "gebaeude_stadtstruktur.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(out_path, index=False)
    logger.info("gebaeude_stadtstruktur saved → %s (%d rows).", out_path, len(enriched))
    return enriched


def run_prime_location_clustering(
    gebaeude_stadtstruktur: gpd.GeoDataFrame,
    k_percentile: float = 99.5,
    min_empl: float = 10.0,
    radius_m: int = 500,
) -> gpd.GeoDataFrame:
    """Detect employment-dense prime-location clusters using AABPL.

    Wraps ``scripts/aabpl_wrapper.detect_employment_clusters``.
    Output saved to ``data/processed/prime_location_clusters.parquet``.

    Parameters
    ----------
    gebaeude_stadtstruktur:
        Output of :func:`process_gebaeude_stadtstruktur`.
        Must contain ``approx_empl`` column and polygon/point geometry.
    k_percentile:
        AABPL percentile threshold (default 99.5).
    min_empl:
        Buildings with ``approx_empl <= min_empl`` excluded from AABPL.
        Default 10 (removes residential/vacant).
    radius_m:
        AABPL search radius in metres (default 500).

    Returns
    -------
    geopandas.GeoDataFrame
        Cluster centroids in EPSG:3035 with columns: ``cluster_id``, ``sum``,
        ``n_cells``, ``centroid_x`` (lon), ``centroid_y`` (lat), ``geometry``.

    Notes
    -----
    Requires the ``aabpl`` package (not in pyproject.toml — install separately).
    The AABPL algorithm defines "prime locations" for the φ_i^prime demand
    component, as per ADR-018.
    """
    import sys
    _scripts = Path(__file__).resolve().parents[3] / "scripts"
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    try:
        from aabpl_wrapper import detect_employment_clusters
    except ImportError as exc:
        raise ImportError(
            "Cannot import aabpl_wrapper from scripts/. "
            "Ensure 'aabpl' is installed and scripts/ is at the repo root."
        ) from exc

    logger.info(
        "Running AABPL (k=%.1f, min_empl=%.0f, r=%dm).",
        k_percentile, min_empl, radius_m,
    )
    scatter = gebaeude_stadtstruktur.copy()
    scatter["geometry"] = scatter.geometry.centroid
    scatter["approx_empl"] = scatter["approx_empl"].fillna(0).clip(lower=0)

    clusters_df, summary = detect_employment_clusters(
        scatter,
        weight_col="approx_empl",
        k_percentile=k_percentile,
        min_empl=min_empl,
        radius_m=radius_m,
    )
    logger.info("AABPL: %d cluster(s) found.", len(clusters_df))

    cluster_gdf = gpd.GeoDataFrame(
        clusters_df,
        geometry=gpd.points_from_xy(clusters_df["centroid_x"], clusters_df["centroid_y"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3035")

    out_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "prime_location_clusters.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_gdf.to_parquet(out_path, index=False)
    logger.info("Prime-location clusters saved → %s.", out_path)
    return clusters_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    download_stadtstruktur()
    download_station_data()
    


if __name__ == "__main__":
    main()
