"""Interactive Leaflet map of the reciprocal rival observation graph (``--graph-states``).

Standalone HTML (Leaflet from CDN, OSM tiles) written into the run folder. Stores are
chain-colour-coded circle markers; edges are hidden until a store is clicked, then that
store's matched rivals are drawn in solid black and its candidate rivals in faint dimgray,
with line width proportional to the diversion edge weight.

No folium dependency — the HTML is assembled by substituting JSON blobs into a template
string (JS braces are left intact). Coordinates are reprojected EPSG:3035 → EPSG:4326.

Public API: write_rival_graph_map
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

CHAIN_TYPE_COLORS: dict[str, str] = {
    "discount": "royalblue",
    "standard": "firebrick",
    "bio": "forestgreen",
}

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { margin: 0; height: 100%; }
  #map { width: 100%; height: 100vh; }
  .legend {
    background: rgba(255,255,255,0.92); padding: 8px 10px; border-radius: 6px;
    font: 12px/1.4 system-ui, sans-serif; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }
  .legend b { display:block; margin-bottom:4px; }
  .legend .sw { display:inline-block; width:11px; height:11px; border-radius:50%;
    margin-right:6px; vertical-align:-1px; }
  .legend .ln { display:inline-block; width:20px; height:0; margin-right:6px;
    vertical-align:3px; }
  .hint { font-size:11px; color:#444; margin-top:6px; }
</style>
</head>
<body>
<div id="map"></div>
<script>
const STORES = __STORES_JSON__;
const MATCHED = __MATCHED_JSON__;     // [[a, b, w], ...]
const CANDIDATE = __CANDIDATE_JSON__; // [[a, b, w], ...]
const WMIN = __WMIN__, WMAX = __WMAX__;

const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

// Fit to stores
const lats = STORES.map(s => s.lat), lons = STORES.map(s => s.lon);
const bounds = [[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]];
map.fitBounds(bounds, {padding: [30, 30]});

// Build adjacency (undirected) from edge lists
const adjM = {}, adjC = {};
function addAdj(adj, a, b, w) {
  (adj[a] = adj[a] || []).push([b, w]);
  (adj[b] = adj[b] || []).push([a, w]);
}
MATCHED.forEach(e => addAdj(adjM, e[0], e[1], e[2]));
CANDIDATE.forEach(e => addAdj(adjC, e[0], e[1], e[2]));

function lineWidth(w) {
  if (WMAX <= WMIN) return 2.0;
  return 1.0 + 5.0 * (w - WMIN) / (WMAX - WMIN);
}

const edgeLayer = L.layerGroup().addTo(map);
let selected = null;

function clearEdges() { edgeLayer.clearLayers(); selected = null; }

function drawEdgesFor(i) {
  edgeLayer.clearLayers();
  const cand = adjC[i] || [];
  for (const [j, w] of cand) {
    L.polyline([[STORES[i].lat, STORES[i].lon], [STORES[j].lat, STORES[j].lon]],
      {color: 'dimgray', weight: lineWidth(w), opacity: 0.30}).addTo(edgeLayer);
  }
  const matched = adjM[i] || [];
  for (const [j, w] of matched) {
    L.polyline([[STORES[i].lat, STORES[i].lon], [STORES[j].lat, STORES[j].lon]],
      {color: 'black', weight: lineWidth(w), opacity: 1.0}).addTo(edgeLayer);
    L.circleMarker([STORES[j].lat, STORES[j].lon],
      {radius: 7, color: 'black', weight: 2, fill: false}).addTo(edgeLayer);
  }
  L.circleMarker([STORES[i].lat, STORES[i].lon],
    {radius: 9, color: 'black', weight: 3, fill: false}).addTo(edgeLayer);
  selected = i;
}

STORES.forEach((s, i) => {
  const mk = L.circleMarker([s.lat, s.lon], {
    radius: 5, color: '#222', weight: 0.6, fillColor: s.color, fillOpacity: 0.9
  }).addTo(map);
  mk.bindTooltip('store ' + i + ' \u00b7 ' + s.ct + ' \u00b7 deg ' + s.deg);
  mk.on('click', (ev) => {
    L.DomEvent.stopPropagation(ev);
    if (selected === i) { clearEdges(); } else { drawEdgesFor(i); }
  });
});

map.on('click', clearEdges);

// Legend
const legend = L.control({position: 'topright'});
legend.onAdd = function () {
  const div = L.DomUtil.create('div', 'legend');
  div.innerHTML =
    '<b>__TITLE__</b>' +
    '<div><span class="sw" style="background:royalblue"></span>Discount</div>' +
    '<div><span class="sw" style="background:firebrick"></span>Standard</div>' +
    '<div><span class="sw" style="background:forestgreen"></span>Bio</div>' +
    '<div style="margin-top:5px;"><span class="ln" style="border-top:3px solid black"></span>matched rival</div>' +
    '<div><span class="ln" style="border-top:3px solid dimgray;opacity:0.4"></span>candidate rival</div>' +
    '<div class="hint">click a store to reveal its edges \u00b7 width \u221d diversion weight</div>';
  return div;
};
legend.addTo(map);
</script>
</body>
</html>
"""


def write_rival_graph_map(
    output_path: "str | Path",
    firms: Sequence[Any],
    graph: Any,
    *,
    title: str = "Rival observation graph",
) -> Path:
    """Write a standalone interactive Leaflet HTML map of the rival graph.

    Parameters
    ----------
    output_path : destination .html path.
    firms : sequence of Firm (``.location`` (x,y) EPSG:3035, ``.chain_type``); order
        must match the positional store indices in ``graph``.
    graph : RivalGraph from hotelling.env.rival_graph (matched_edges/candidate_edges/...).
    title : map / legend title.

    Returns
    -------
    Path to the written file.
    """
    from pyproj import Transformer

    output_path = Path(output_path)
    N = len(firms)
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

    stores = []
    for idx, f in enumerate(firms):
        x, y = float(f.location[0]), float(f.location[1])
        lon, lat = transformer.transform(x, y)
        ct = str(getattr(f, "chain_type", "standard"))
        stores.append({
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "ct": ct,
            "color": CHAIN_TYPE_COLORS.get(ct, "gray"),
            "deg": int(graph.degree[idx]) if hasattr(graph, "degree") else 0,
        })

    def _edges(edges, weights):
        out = []
        for e in range(len(weights)):
            a, b = int(edges[e][0]), int(edges[e][1])
            out.append([a, b, round(float(weights[e]), 6)])
        return out

    matched = _edges(graph.matched_edges, graph.matched_weights)
    candidate = _edges(graph.candidate_edges, graph.candidate_weights)

    all_w = [e[2] for e in candidate] + [e[2] for e in matched]
    wmin = min(all_w) if all_w else 0.0
    wmax = max(all_w) if all_w else 1.0

    html = (
        _TEMPLATE
        .replace("__TITLE__", title)
        .replace("__STORES_JSON__", json.dumps(stores))
        .replace("__MATCHED_JSON__", json.dumps(matched))
        .replace("__CANDIDATE_JSON__", json.dumps(candidate))
        .replace("__WMIN__", repr(float(wmin)))
        .replace("__WMAX__", repr(float(wmax)))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(
        "Rival-graph map: %d stores, %d matched + %d candidate edges -> %s",
        N, len(matched), len(candidate), output_path,
    )
    return output_path
