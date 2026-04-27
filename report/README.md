# Hotelling LLM Spatial Competition – Seminar Report

## Structure

```
report/
├── figures/          # Generated figures (PNG/PDF from viz/ modules)
├── tables/           # LaTeX / CSV tables from analysis/ modules
└── README.md         # This file
```

## Generating Figures

After running a simulation sweep, generate all report figures with:

```bash
python -c "
from pathlib import Path
from hotelling.analysis.results_db import ResultsDB
from hotelling.viz.static import plot_price_timeseries, plot_irf, plot_profit_heatmap

db = ResultsDB(Path('results/runs'))
db.connect()
# ... see notebooks/05_results_analysis.ipynb for full figure generation code
"
```

## Figure Descriptions

| File | Description |
|------|-------------|
| `figures/fig01_price_timeseries.pdf` | Price trajectories over 1M training steps (Q-learning duopoly) |
| `figures/fig02_irf.pdf` | Impulse-response function after one-shot deviation |
| `figures/fig03_delta_heatmap.pdf` | Profit gain Δ across (α, β) sweep |
| `figures/fig04_dose_response.pdf` | Δ as a function of transport cost t |
| `figures/fig05_spatial_voronoi.pdf` | Voronoi market areas for Berlin Mitte (OSM data) |
| `figures/fig06_llm_vs_qlearning.pdf` | LLM agent price paths vs Q-learning benchmark |
