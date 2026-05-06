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

logger = logging.getLogger(__name__)

__all__ = [
    "download_IHK_data",
    "download_index_data",
    "download_medianeinkommen_data",
    "download_stadtstruktur",
    "download_station_data",
    "identify_cbd",
    "identify_transport_hubs",
    "process_esix_mss_data",
    "process_ihk_data",
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

# WFS layer tuples: (service, qualified_layer_name)
# All names verified against live GetCapabilities responses.
_ESIX_LAYER          = ("gssa_esix2022",    "gssa_esix2022:gssa_esix2022")
_MSS_LAYER           = ("mss_2023",          "mss_2023:mss2023_indizes_542")
_STADTSTRUKTUR_LAYER = ("ua_stadtstruktur",  "ua_stadtstruktur:b_stadtstruktur_differenziert_2024")
_GEBAEUDE_LAYER      = ("alkis_gebaeude",    "alkis_gebaeude:gebaeude")
_ZENTREN_FMA_LAYER   = ("step_zen_2040",     "step_zen_2040:step_zen_2040_fma")
_ZENTREN_ZH_LAYER    = ("step_zen_2040",     "step_zen_2040:step_zen_2040_zh")

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

    esix_path = Path("data/raw/esix.gpkg")
    mss_path = Path("data/raw/mss.gpkg")

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
    stadtstruktur_path = Path("data/raw/stadtstruktur.gpkg")
    gebaeude_path = Path("data/raw/gebaeude.gpkg")
    zentren_path = Path("data/raw/zentren.gpkg")

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
    db_pdf_path = Path("data/raw/db_station_data.pdf")
    db_csv_path = Path("data/raw/db_station_data.csv")

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
    gtfs_dir = Path("data/raw/gtfs")
    stops_file = gtfs_dir / "stops.txt"

    if skip_if_exists and stops_file.exists():
        logger.info("GTFS stops already exist at %s — skipping.", gtfs_dir)
    else:
        url = gtfs_url or _GTFS_DEFAULT_URL
        gtfs_zip = Path("data/raw/gtfs.zip")

        logger.info("Downloading GTFS from %s …", url)
        gtfs_zip.parent.mkdir(parents=True, exist_ok=True)
        _stream_to_file(url, gtfs_zip, desc="GTFS")

        logger.info("Extracting GTFS → %s …", gtfs_dir)
        gtfs_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(gtfs_zip) as zf:
            zf.extractall(gtfs_dir)

        gtfs_zip.unlink()
        logger.info("GTFS extracted → %s", gtfs_dir)


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
