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
| **LLM agent** | Uses a large language model (LiteLLM + Instructor) to price each period with structured output validation |
| **Q-learning agent** | Tabular Q-policy over discretised price states (Calvano et al. 2020 calibration) |
| **Myopic agent** | Stage-game Bertrand-Nash best-response baseline |
| **Random agent** | Uniformly random baseline for benchmarking |
| **Deep-Q agent** | Neural Q-network for larger state spaces |

The toolkit supports both **synthetic** unit-square geometries and **real
spatial data** (Zensus 2022 population rasters, OpenStreetMap boundaries,
Berlin business locations).

---

## Repository Structure

```text
DEDA_LLM_Spatial_Hotelling/
│
├── src/hotelling/              # Main installable package (src layout)
│   ├── core/                   # City, Firm, market clearing, equilibrium solvers
│   ├── agents/                 # AgentProtocol, Q-learning, deep-Q, myopic, random, LLM
│   ├── env/                    # PettingZoo HotellingMarketEnv, EntryGame
│   ├── spatial/                # SquareGrid, OSM loader, raster reader, distance utils
│   ├── simulation/             # SimulationEngine, batch runner, Parquet recorder
│   ├── analysis/               # ResultsDB (DuckDB), metrics, IRF
│   ├── viz/                    # Static (matplotlib), interactive (plotly/folium), animation
│   ├── llm/                    # LiteLLM client, Pydantic schemas, Jinja2 prompts
│   ├── utils/                  # Seeding, structured logging
│   └── cli.py                  # Typer CLI (hotelling train / sweep / export)
│
├── configs/                    # Hydra-style YAML configs
│   ├── config.yaml             # Top-level defaults
│   ├── city/                   # unit_square.yaml, berlin_mitte.yaml
│   ├── env/                    # base.yaml
│   ├── agents/                 # qlearning_duopoly.yaml, llm_duopoly.yaml
│   └── sweep/                  # alpha_beta.yaml, transport_cost.yaml
│
├── apps/
│   └── explore_market.py       # Interactive marimo exploration app
│
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── unit/                   # Unit tests (core, agents, utils, llm, recorder)
│   └── integration/            # Integration tests (env construction, etc.)
│
├── notebooks/                  # Jupyter notebooks (01–05)
├── docs/
│   ├── decisions/              # Architecture Decision Records (ADR-001 – ADR-003)
│   └── theory/                 # Background notes on the Hotelling model
│
├── report/
│   ├── figures/                # Generated figures (PNG/PDF)
│   └── README.md               # Figure generation instructions
│
├── data/
│   ├── raw/                    # Raw spatial / business data (not committed)
│   ├── processed/              # Pre-processed datasets
│   └── synthetic/              # Generated synthetic datasets
│
├── pyproject.toml              # Package metadata & tool configuration (src layout)
├── Makefile                    # Development workflow shortcuts
├── .pre-commit-config.yaml     # Pre-commit hooks (ruff lint + format)
├── .python-version             # Pinned Python version (3.11)
├── requirements.txt            # Runtime pinned dependencies (for reproducibility)
└── requirements-dev.txt        # Development dependencies
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/jendrkk/DEDA_LLM_Spatial_Hotelling.git
cd DEDA_LLM_Spatial_Hotelling

# Install core package in editable mode (src layout)
pip install -e .

# Install all optional extras (LLM, spatial, viz, RL, DB, CLI, notebooks)
pip install -e ".[all]"

# Or use the Makefile shortcut
make install-dev
```

### Optional extras

| Extra | Key packages | Use case |
|-------|-------------|----------|
| `llm` | `litellm`, `instructor`, `openai` | LLM-powered agents |
| `spatial` | `geopandas`, `rasterio`, `shapely`, `osmnx` | Real GIS / raster data |
| `viz` | `matplotlib`, `plotly`, `folium`, `pydeck`, `imageio` | All visualizations |
| `rl` | `torch`, `gymnasium`, `pettingzoo` | Deep-Q and RL training |
| `db` | `duckdb`, `pyarrow` | DuckDB query layer for results |
| `cli` | `typer` | Command-line interface |
| `notebooks` | `jupyter`, `marimo`, `tqdm` | Interactive notebooks |
| `all` | All of the above | Full functionality |

---

## Quick Start

### Programmatic API

```python
from hotelling.core.city import City
from hotelling.core.firm import Firm
from hotelling.env.market_env import HotellingMarketEnv
from hotelling.agents.qlearning import QLearningAgent

# Define a unit-square city with two duopoly firms
city = City(boundary=(0.0, 0.0, 1.0, 1.0))
firms = [
    Firm(id="firm_0", location=(0.25, 0.5), marginal_cost=1.0),
    Firm(id="firm_1", location=(0.75, 0.5), marginal_cost=1.0),
]
city.firms = firms

# Create the PettingZoo-compatible environment
env = HotellingMarketEnv(city=city, firms=firms, m=15)

# Create Q-learning agents (Calvano 2020 calibration)
agents = {
    f.id: QLearningAgent(firm_id=f.id, m=15, alpha=0.10, beta=2e-5, seed=i)
    for i, f in enumerate(firms)
}

print("Environment ready. Agents:", list(agents.keys()))
```

### CLI

```bash
# Run a single training session
hotelling train --config configs/config.yaml

# Run an (α, β) parameter sweep
hotelling sweep --config configs/sweep/alpha_beta.yaml --jobs -1

# Merge all run Parquet files into one summary
hotelling export results/runs/ --out results/summary.parquet
```

### Interactive exploration

```bash
# Install marimo, then launch the exploration app
pip install marimo
marimo edit apps/explore_market.py
```

---

## Running Tests

```bash
make install-dev
make test          # all tests
make test-unit     # unit tests only
make coverage      # with HTML coverage report
```

Or directly:

```bash
pytest tests/unit/ tests/integration/ -v
```

---

## Development

```bash
make lint          # ruff check
make format        # ruff format
make lint-fix      # auto-fix lint issues
```

Pre-commit hooks (ruff lint + format + standard checks):

```bash
pip install pre-commit
pre-commit install
```

---

## Spatial Data

Place raw data files in `data/raw/`:

| File | Description |
|------|-------------|
| `zensus2022_mitte_100m.tif` | Zensus 2022 population raster at 100 m resolution |
| `boundary.geojson` | Geographic boundary of the study area (OSM admin polygon) |
| `businesses.csv` | Business locations with `lon`/`lat` columns |

Large data files should be tracked with [DVC](https://dvc.org/) rather than
committed directly to Git.

---

## LLM Agent

The LLM agent works with any OpenAI-compatible endpoint via LiteLLM.
Always pin to a model snapshot for reproducibility:

```python
from hotelling.agents.llm import LLMAgent

agent = LLMAgent(
    firm_id="gpt_firm",
    model="gpt-4o-2024-08-06",   # always pin to a snapshot
    temperature=0,
    log_path="results/llm_calls.jsonl",
)

# For local Ollama
agent_local = LLMAgent(
    firm_id="local_firm",
    model="ollama/llama3",
)
```

Set your API key: `export OPENAI_API_KEY=sk-...`

---

## Architecture Decisions

See [`docs/decisions/`](docs/decisions/) for Architecture Decision Records:

- [ADR-001](docs/decisions/ADR-001-src-layout.md) – src layout for the Python package
- [ADR-002](docs/decisions/ADR-002-llm-litellm-instructor.md) – LiteLLM + Instructor for LLM integration
- [ADR-003](docs/decisions/ADR-003-pettingzoo-env.md) – PettingZoo ParallelEnv as simulation wrapper
- [ADR-004](docs/decisions/ADR-004-per-store-independent-qtables.md) – Per-store independent Q-tables; no sharing within chain
- [ADR-005](docs/decisions/ADR-005-relative-action-space.md) – Relative action space; Q-tables survive CEO epoch changes
- [ADR-006](docs/decisions/ADR-006-three-phase-simulation.md) – Three-phase structure; burn-in before CEO activation
- [ADR-007](docs/decisions/ADR-007-llm-calls-not-batched.md) – CEO calls never batched; information isolation argument
- [ADR-008](docs/decisions/ADR-008-gamma-fixed-globally.md) – Gamma fixed globally; not a CEO parameter
- [ADR-009](docs/decisions/ADR-009-group-division-extensibility.md) – Group divisions extensible via registry; at most 2 active
- [ADR-010](docs/decisions/ADR-010-entrant-qtable-initialisation.md) – Four Q-table init strategies for entrant; LLM meta-choice option
- [ADR-011](docs/decisions/ADR-011-entrant-response-function.md) – Entrant commits to response function; not per-period LLM
- [ADR-012](docs/decisions/ADR-012-inner-ring-not-pankow.md) – Geographic scope: inner-Ringbahn, not Pankow

---

## References

* Hotelling, H. (1929). *Stability in Competition*. The Economic Journal, 39(153), 41–57.
* d'Aspremont et al. (1979). *On Hotelling's 'Stability in Competition'*. Econometrica.
* Calvano, E., Calzolari, G., Denicolo, V., & Pastorello, S. (2020).
  *Artificial intelligence, algorithmic pricing, and collusion*.
  American Economic Review, 110(10), 3267–3297.
* Anderson, de Palma, Thisse (1992). *Discrete Choice Theory of Product Differentiation*. MIT Press.
* Tabuchi, T. (1994). *Two-stage two-dimensional spatial competition between two firms*. Regional Science and Urban Economics.
* Terry, J. et al. (2021). *PettingZoo: Gym for Multi-Agent Reinforcement Learning*.

---

## License

[MIT](LICENSE)

