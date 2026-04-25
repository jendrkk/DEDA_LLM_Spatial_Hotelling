# DEDA_LLM_Spatial_Hotelling

> **Python toolkit for 2-D Hotelling spatial competition simulations with LLM and Q-learning agents.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository implements a flexible simulation framework for the classical
[Hotelling model of spatial competition](docs/theory/hotelling_model.md)
extended to **two dimensions** and enriched with intelligent market actors:

| Agent type | Description |
|------------|-------------|
| **LLM agent** | Uses a large language model (OpenAI-compatible API) to decide location and price each period |
| **Q-learning agent** | Learns a tabular Q-policy over discretised location/price states |
| **Naive agent** | Rule-based heuristics (move-to-centre, random walk, stay) |

The toolkit supports both **synthetic** city geometries and **real spatial data**
(population rasters, geographic boundaries, business location datasets).

---

## Repository Structure

```text
DEDA_LLM_Spatial_Hotelling/
│
├── hotelling/                  # Main installable package
│   ├── core/                   # City, Market, Firm, Consumer models
│   ├── agents/                 # BaseAgent, LLMAgent, QLearningAgent, NaiveAgent
│   ├── spatial/                # MapLoader, PopulationGrid, BusinessLocations
│   ├── simulation/             # SimulationEngine, SimulationConfig, metrics
│   ├── visualization/          # Static (matplotlib) & interactive (plotly) plots
│   └── utils/                  # Logging, data I/O helpers
│
├── data/
│   ├── raw/                    # Raw spatial / business data (not committed)
│   ├── processed/              # Pre-processed datasets
│   └── synthetic/              # Generated synthetic datasets
│
├── models/
│   ├── llm/                    # LLM prompts, adapter configs, fine-tune artefacts
│   └── qlearning/              # Saved Q-table JSON files
│
├── notebooks/                  # Jupyter notebooks for interactive analysis
│   ├── 01_getting_started.ipynb
│   ├── 02_general_simulation.ipynb
│   ├── 03_llm_agents.ipynb
│   ├── 04_spatial_analysis.ipynb
│   └── 05_results_analysis.ipynb
│
├── experiments/                # Runnable experiment scripts
│   ├── general_hotelling.py
│   ├── llm_competition.py
│   └── spatial_real_data.py
│
├── tests/                      # Pytest test suite
│   ├── test_core/
│   ├── test_agents/
│   └── test_simulation/
│
├── docs/
│   └── theory/                 # Background notes on the Hotelling model
│
├── configs/                    # YAML configuration files
│   ├── default.yaml
│   ├── llm_config.yaml
│   └── qlearning_config.yaml
│
├── pyproject.toml              # Package metadata & tool configuration
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development dependencies
└── .gitignore
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/jendrkk/DEDA_LLM_Spatial_Hotelling.git
cd DEDA_LLM_Spatial_Hotelling

# Install core package (editable mode)
pip install -e .

# Install all optional extras (LLM, spatial, visualisation, notebooks)
pip install -e ".[all]"
```

### Optional extras

| Extra | Installs | Use case |
|-------|----------|----------|
| `llm` | `openai` | LLM-powered agents |
| `spatial` | `geopandas`, `rasterio`, `shapely` | Real GIS / raster data |
| `viz` | `matplotlib`, `plotly` | Static & interactive plots |
| `notebooks` | `jupyter`, `ipywidgets`, `tqdm` | Jupyter notebooks |
| `yaml` | `pyyaml` | YAML config loading |
| `all` | All of the above | Full functionality |

---

## Quick Start

### Programmatic API

```python
from hotelling.agents.naive_agent import NaiveAgent
from hotelling.simulation.config import SimulationConfig
from hotelling.simulation.engine import SimulationEngine
from hotelling.simulation.metrics import aggregate_results

cfg = SimulationConfig(n_steps=100, n_consumers=2000, seed=42)
engine = SimulationEngine(config=cfg)

firms = [
    NaiveAgent("Firm_A", location=(0.1, 0.5), strategy="center"),
    NaiveAgent("Firm_B", location=(0.9, 0.5), strategy="center"),
]

engine.setup(firms)
results = engine.run()
agg = aggregate_results(results)
print(engine.market.summary())
```

### Running experiments

```bash
# General Hotelling competition (naive agents)
python experiments/general_hotelling.py

# LLM vs. naive agent (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python experiments/llm_competition.py

# Spatial experiment with real data (falls back to synthetic if data absent)
python experiments/spatial_real_data.py
```

### Jupyter notebooks

```bash
jupyter notebook notebooks/
```

Start with `01_getting_started.ipynb` for a guided tour of the package.

---

## Running Tests

```bash
pip install -e ".[all]"
pip install pytest pytest-cov
pytest
```

---

## Spatial Data

Place raw data files in `data/raw/`:

| File | Description |
|------|-------------|
| `boundary.geojson` | Geographic boundary of the study area |
| `population.tif` | Population density raster (GeoTIFF) |
| `businesses.csv` | Business locations with `lon`/`lat` columns |

Processed/derived datasets go in `data/processed/`.
Large data files should be tracked with [DVC](https://dvc.org/) rather than
committed directly to Git.

---

## LLM Agent Configuration

The LLM agent uses any OpenAI-compatible endpoint.  Configure it in
`configs/llm_config.yaml` or pass parameters directly:

```python
from hotelling.agents.llm_agent import LLMAgent

agent = LLMAgent(
    firm_id="GPT_Firm",
    model="gpt-4o-mini",
    # For local Ollama:
    # base_url="http://localhost:11434/v1",
    # api_key="ollama",
)
```

---

## Q-Learning Agent

Trained Q-tables are saved as JSON files in `models/qlearning/`:

```python
from hotelling.agents.qlearning_agent import QLearningAgent

agent = QLearningAgent(firm_id="QL_Firm", n_bins=10, alpha=0.1, gamma=0.95)
# ... run simulation ...
agent.save("models/qlearning/ql_firm.json")

# Load in a later session
agent.load("models/qlearning/ql_firm.json")
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Install dev dependencies: `pip install -e ".[all]" -r requirements-dev.txt`
3. Run the test suite: `pytest`
4. Run the linter: `ruff check . && ruff format .`
5. Open a pull request.

---

## References

* Hotelling, H. (1929). *Stability in Competition*. The Economic Journal, 39(153), 41–57.
* d'Aspremont et al. (1979). *On Hotelling's 'Stability in Competition'*. Econometrica.
* Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press.
* Sutton & Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

## License

[MIT](LICENSE)
