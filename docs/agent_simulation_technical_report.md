# Agent Simulation — Technical Report
## *Agents in the City: LLM-Driven Spatial Market Entry and Algorithmic Pricing in Berlin*

*Technical implementation memo. Describes agent architecture, simulation phases, envelope design, LLM call budget, and required repository structure. Intended for use as a refactoring specification by a subsequent Claude session. Does not duplicate the economic model equations in `economic_model_specification.md` — consult that document for demand system, calibration targets, and equilibrium benchmarks.*

---

## 1. Simulation Overview

The simulation is a discrete-time spatial Bertrand game on the inner-Ringbahn Berlin grid. It involves three types of ML agents operating at different timescales:

- **Per-store RL agents** — one independent Q-learning agent per physical store, operating every period.
- **Per-chain LLM-CEOs** — one LLM agent per incumbent chain, operating every `T_CEO` periods, setting the strategic envelope within which that chain's store agents operate.
- **Entrant LLM agent** — one LLM agent that solves the entry game at `t=0`, commits to a response function, and is periodically or event-triggered to reassess.

The simulation runs in three sequential phases. LLM calls occur only in Phase 2 and Phase 3. The RL burn-in in Phase 1 is entirely LLM-free.

---

## 2. Agent Roster

| Agent | Type | Count | Acts when | Output |
|---|---|---|---|---|
| Store RL agent | Tabular Q-learning | 1 per store (~100–150 total) | Every period `t` | `(price_index, effort_index)` within envelope |
| Chain LLM-CEO | LLM (API) | 1 per incumbent chain (~9) | Every `T_CEO = 100` periods | Extended strategy envelope per store group |
| Entrant LLM | LLM (API) | 1 | `t=0` + event/time triggers | Entry decision + response function |

### 2.1 Information segregation (economically motivated)

- **Store RL agents** observe only: own last price, own last effort, own noisy profit, nearest rivals' last prices and efforts, and the current envelope from their CEO. They never see consumer demographics, chain-level aggregates, or other stores' internal state.
- **LLM-CEOs** observe: own chain aggregates (mean price, mean effort, total profit, total demand over last `T_CEO` periods), rivals' published prices and locations, aggregated consumer demographics (zone-level, not full LOR list), and their own strategy history.
- **Entrant LLM** observes: full market map (compressed as zone-level store density by chain type), own performance since last call, and aggregated consumer demographics.

This segregation mirrors real retail delegation (Aghion–Tirole 1997) and is enforced at the prompt level — each agent type receives a distinct prompt template with only its permitted information.

---

## 3. Simulation Phases

### Phase 0 — Incumbent Burn-in (no LLM)

**Purpose:** Allow all incumbent store RL agents to populate their Q-tables and converge to stable policies before the CEO layer and entrant activate. This is necessary because tabular Q-learning requires many visits per state to yield reliable Q-values.

**Mechanics:**
- All incumbent stores run Q-learning with a fixed, chain-wide, single-group envelope (no group differentiation yet).
- No LLM calls. Zero API cost.
- Exploration rate `epsilon` fixed at a configurable default (e.g. `0.1`).
- Discount factor `gamma` fixed globally for the entire simulation (not a CEO parameter).
- Duration: configurable, suggested `T_burnin >= 50_000` periods.
- Entrant does not exist during this phase.

**Output:** Converged Q-tables for all ~100–150 incumbent stores.

### Phase 1 — Entry

**Purpose:** The entrant LLM makes its entry decision given the converged incumbent market.

**Mechanics:**
- Single LLM call to the entrant agent.
- Entrant observes: incumbent store locations (zone-level density summary), chain types, consumer demographic summary, and sunk cost schedule.
- Entrant outputs: chain type `theta_e ∈ {D, S, B}`, location `l_e` from the commercial site grid, and an initial response function (see §5.2).
- Entrant's store is instantiated in the environment.
- Entrant's tactical RL layer (if enabled) is initialised — see §5.3 for Q-table initialisation options.

**LLM calls:** 1.

### Phase 2 — Strategic Game

**Purpose:** The main simulation. Incumbent CEOs activate, the entrant operates via its response function with periodic reassessment.

**Mechanics:**
- All store RL agents continue operating every period inside their CEO's envelope.
- Incumbent LLM-CEOs called every `T_CEO = 100` periods per chain (separately, never batched — see §6.1).
- Entrant LLM called on time trigger (every 50 periods) or event trigger (whichever comes first) — see §5.4.
- Duration: configurable, suggested `T_game ∈ [1_000, 5_000]` periods.
- Analytical benchmarks (Bertrand-Nash, joint monopoly) computed at the end of each session.

**LLM calls per session:** approximately `(T_game / T_CEO) × N_chains + T_game / 50` — for `T_game = 5_000`, ~550 calls total, well within free-tier daily limits.

---

## 4. Extended Strategy Envelope and Group Divisions

### 4.1 Baseline envelope (no groups)

When no group divisions are active, the CEO outputs a single envelope for all stores in its chain:

```
sigma_c = (p_bar, delta_p, e_bar, delta_e, epsilon_explore)
```

- `p_bar` — target price centre
- `delta_p` — permitted price half-width; store prices restricted to `[p_bar - delta_p, p_bar + delta_p]`
- `e_bar` — target effort centre
- `delta_e` — permitted effort half-width
- `epsilon_explore` — RL exploration rate for stores in this group (controls how aggressively stores try new prices)

`gamma` (RL discount factor) is **fixed globally** and is not a CEO output parameter. Changing `gamma` mid-simulation invalidates existing Q-values; it is set once at initialisation.

### 4.2 Group division system

The CEO can differentiate its envelope across store groups. The system supports **at most 2 simultaneous group divisions**, each with **exactly 2 categories**, yielding at most 4 groups (2 × 2 Cartesian product).

**Supported configurations:**
- **0 divisions:** Single chain-wide envelope (5 parameters).
- **1 division:** 2 groups × 5 parameters = 10 CEO output parameters.
- **2 divisions:** 4 groups × 5 parameters = 20 CEO output parameters.

**Implemented divisions (baseline):**

| Division | Category A | Category B | Assignment criterion |
|---|---|---|---|
| `DIVISION_COMPETITION` | `HEAVY` | `EASY` | Number of rival stores within radius `R` of store location — threshold configurable |
| `DIVISION_NEIGHBOURHOOD` | `RICH` | `POOR` | LOR social-status index `S_r` of the store's LOR — threshold configurable (default: median) |

**Extensibility requirement:** The group division system must be implemented so that adding a new division (e.g., `DIVISION_RIVAL_TYPE` based on type of nearest competitor) requires only: (a) adding a new division class, (b) registering it, and (c) adding a config entry. No changes to the envelope or CEO logic should be required. See §8 for required module structure.

**Group assignment:** Groups are assigned to each store **once at startup** (Phase 0 initialisation) and remain fixed for the entire simulation. Group membership is part of the store's persistent metadata, not its per-period RL state.

**Group label in store state:** The store RL agent receives its group label(s) as part of the envelope it receives from the CEO. This means the Q-table has a dimension for group membership (binary per active division), which slightly increases Q-table size but remains within the lean design.

### 4.3 CEO output schema (2-division example)

```json
{
  "chain_id": "Edeka",
  "epoch": 200,
  "groups": {
    "HEAVY_RICH":   {"p_bar": 1.30, "delta_p": 0.18, "e_bar": 0.75, "delta_e": 0.20, "epsilon": 0.04},
    "HEAVY_POOR":   {"p_bar": 1.15, "delta_p": 0.20, "e_bar": 0.70, "delta_e": 0.20, "epsilon": 0.05},
    "EASY_RICH":    {"p_bar": 1.65, "delta_p": 0.07, "e_bar": 0.50, "delta_e": 0.12, "epsilon": 0.03},
    "EASY_POOR":    {"p_bar": 1.45, "delta_p": 0.10, "e_bar": 0.45, "delta_e": 0.12, "epsilon": 0.04}
  }
}
```

For the 0-division baseline, the schema simplifies to a single `"default"` group key. The CEO prompt template renders the group structure dynamically from the active division configuration.

---

## 5. Store RL Agent — Technical Design

### 5.1 Q-table design (lean)

Each store maintains an **independent** Q-table. Stores of the same chain do not share Q-tables. This is economically motivated (store managers act on local information) and preserves within-chain price dispersion as an observable outcome.

**State space (lean design):**

| Variable | Levels | Notes |
|---|---|---|
| Own relative position in price envelope | 5 | Replaces absolute own price |
| Own relative position in effort envelope | 3 | Replaces absolute own effort |
| Nearest rival undercut? | 2 | Binary: rival price < own price |
| Nearest rival effort high? | 2 | Binary: above/below chain median |
| Group membership (per active division) | 2 per division | 1 bit per active division; max 2 bits |

Total states (2 active divisions): `5 × 3 × 2 × 2 × 2 × 2 = 480` states.

**Action space (relative):**

Actions are defined as relative adjustments to the previous period's price and effort, not absolute levels. This makes the Q-table transferable across CEO epochs (when the envelope centre shifts, the store does not need to relearn):

- Price moves: `{-15%, -10%, -5%, 0%, +5%, +10%, +15%}` — 7 levels
- Effort moves: `{-1 level, 0, +1 level}` — 3 levels
- Total actions: `7 × 3 = 21`

Moves that would exceed the current envelope boundaries are **clipped** to the boundary at execution time.

**Q-table size:** `480 × 21 = 10,080` cells per store. At 150 stores: `~1.5M` cells total, `~6 MB` float32.

**Convergence criterion:** Policies are considered converged when the average Q-value change per update falls below a configurable threshold over a rolling window. This is checked per-store, not globally.

### 5.2 Update rule

Standard tabular Q-learning (Calvano 2020 §II.B):

```
Q(s, a) ← Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

- `alpha` (learning rate): configurable, suggested `0.1–0.15` per Calvano grid.
- `gamma` (discount factor): fixed globally, suggested `0.95`.
- Profit signal `r` is noisy: `r_tilde = r + N(0, sigma_pi)`.
- Exploration: `epsilon`-greedy, where `epsilon` is set by the CEO per group (Phase 2) or fixed at default (Phase 0).

---

## 6. LLM-CEO — Technical Design

### 6.1 Information isolation (critical)

Each chain's CEO is called in a **separate API request**. Chains are never batched into a single LLM call. Rationale: a batched call places all chains' private data (profit history, internal strategy) in a single context window, allowing the LLM to implicitly coordinate them — destroying the competitive structure of the model. Rivals' private information is never included in any CEO's prompt; only their published prices and locations are visible.

### 6.2 Prompt structure

**System prompt** (fixed per call): role description (CEO of chain `c`, chain type, marginal cost), output schema (Pydantic-validated JSON), hard constraints (price ≥ marginal cost, minimum envelope width), and active group divisions with category descriptions.

**User prompt** (dynamic, per epoch): structured JSON containing:
- Own chain summary: mean price, mean effort, total profit, total demand over last `T_CEO` periods, profit trend (delta vs previous epoch).
- Rival public state: for each rival chain — chain ID, type, mean published price, store count. **No rival profit, no rival effort, no internal data.**
- Consumer context: zone-level aggregates (5–10 zones max, not full LOR list). Each zone: population, high-status share, zone label.
- CEO history: last 3 epochs — envelope set, realised profit.
- Active group division definitions and per-group performance summary.

**Estimated token budget per call:**

| Component | Tokens |
|---|---|
| System prompt + schema | ~400 |
| Own chain state | ~80 |
| Rival public state (9 rivals) | ~150 |
| Consumer zone summary (8 zones) | ~80 |
| CEO history (3 epochs) | ~120 |
| **Total input** | **~830** |
| Output (envelope JSON, 2 divisions) | ~180 |
| **Total per call** | **~1,010** |

### 6.3 Output validation

CEO output is validated by a Pydantic schema before being applied to the stores. Validation checks:
- All `p_bar` values exceed chain marginal cost.
- All `delta_p` values meet a configurable minimum width (prevents envelope collapse).
- All `epsilon` values within `(0, 1)`.
- All required group keys are present.

On validation failure: retain the previous epoch's envelope and log the failure. Do not crash.

---

## 7. Entrant LLM Agent — Technical Design

### 7.1 Entry decision (Phase 1, 1 call)

The entrant LLM receives:
- Market map: zone-level store density by chain type (not full coordinate list).
- Consumer demographic summary: same format as CEO zone summary.
- Sunk cost schedule by chain type `{D, S, B}`.
- Candidate site grid description (zone labels, not full coordinate list).

The entrant outputs:
- `theta_e ∈ {D, S, B}` — chain type.
- `l_e` — location as a zone + site index.
- Initial response function (see §7.2).
- Optionally: Q-table initialisation preference (see §7.3).

### 7.2 Response function

Instead of calling the LLM every period, the entrant commits to a **response function** — a conditional pricing rule executed mechanically each period until the next LLM call. The response function schema:

```json
{
  "base_price": 1.35,
  "base_effort": 0.65,
  "rival_undercut_response": {
    "threshold": 0.08,
    "own_price_adjustment": -0.05
  },
  "profit_distress_response": {
    "profit_threshold": 600.0,
    "own_price_adjustment": +0.07
  },
  "envelope": {
    "p_bar": 1.35, "delta_p": 0.12,
    "e_bar": 0.65, "delta_e": 0.15,
    "epsilon": 0.08
  }
}
```

The response function is validated by Pydantic and executed by a deterministic interpreter — no LLM call per period. If the entrant has an RL tactical layer, the response function sets the envelope that the RL agent operates within.

### 7.3 Entrant Q-table initialisation (4 options)

The entrant's store-level RL agent (if used) cannot benefit from Phase 0 burn-in because the entrant did not exist then. Four initialisation strategies are supported, selectable via config or optionally by the entrant LLM itself:

| Option | Description | When appropriate |
|---|---|---|
| `BLANK` | Q-table initialised to zeros. Starts from scratch. | Conservative default |
| `INHERIT_ALGORITHM` | Algorithm selects the most similar incumbent store (by chain type + competitive density group) and copies its Q-table. | When similarity is well-defined |
| `INHERIT_LLM_CHOICE` | LLM is shown a list of 3 candidate stores (by chain type, group, location summary) and chooses which to inherit from. | When LLM reasoning about similarity is the research question |
| `INHERIT_LLM_AUTO` | LLM is asked whether it wants a pre-trained Q-table at all. If yes, the algorithm selects the most similar store. | Tests LLM meta-reasoning about learning |

Options `INHERIT_LLM_CHOICE` and `INHERIT_LLM_AUTO` add 1 LLM call at Phase 1 initialisation. All options are configurable; the repo must support all four without code changes between options.

### 7.4 Entrant LLM reassessment triggers

The entrant LLM is called again when **either** trigger fires, whichever comes first:

**Time trigger:** Every `T_entrant = 50` periods (configurable).

**Event triggers (both implemented):**
- `PROFIT_DROP`: Realised profit falls more than `X%` below the rolling mean of the last `T_window` periods. `X` and `T_window` are configurable.
- `RIVAL_EVENT`: A rival chain's mean price changes by more than `Y%` in a single CEO epoch, or a rival's store opens/closes within radius `R_event` of the entrant's store.

**Trigger system design requirement:** The trigger system must be extensible — adding a new trigger type requires only implementing a `Trigger` interface and registering it, without changes to the entrant agent or the simulation runner.

On reassessment, the entrant LLM receives its current performance since last call and the updated market state, and outputs a new response function (and optionally a new envelope for its RL layer).

### 7.5 Entrant prompt token budget

| Call type | Input tokens | Output tokens | Total |
|---|---|---|---|
| Initial entry (Phase 1) | ~1,200 | ~200 | ~1,400 |
| Reassessment (Phase 2) | ~1,000 | ~200 | ~1,200 |

---

## 8. Repository Structure — Required Refactoring

The current repository (`src/hotelling/`) was designed for a simpler agent architecture (flat Q-learners + single LLM agent, no CEO layer, no groups). The following restructuring is required. The package name `hotelling` is retained.

### 8.1 Module structure (target)

```
src/hotelling/
│
├── env/
│   ├── market_env.py          # PettingZoo ParallelEnv; orchestrates per-period loop
│   ├── market_clearing.py     # Logit demand, profit computation — unchanged
│   ├── geography.py           # Grid, distance matrix, LOR assignment — unchanged
│   └── state.py               # MarketState dataclass; StoreState dataclass — extend with group labels
│
├── agents/
│   ├── base.py                # Abstract Agent interface — update signature for envelope input
│   ├── store_rl.py            # Per-store Q-learning agent (NEW — extracted and extended from qlearner.py)
│   ├── chain_ceo.py           # LLM-CEO per chain (NEW)
│   └── entrant_llm.py         # Entrant LLM agent with response function (NEW)
│
├── envelope/                  # NEW module
│   ├── envelope.py            # GroupEnvelope and ChainEnvelope dataclasses; validation logic
│   ├── groups.py              # GroupDivision abstract base; group assignment logic; registry
│   └── divisions/             # NEW subdirectory — one file per implemented division
│       ├── competition.py     # DIVISION_COMPETITION (heavy/easy by rival count)
│       └── neighbourhood.py   # DIVISION_NEIGHBOURHOOD (rich/poor by LOR status index)
│
├── simulation/                # NEW module
│   ├── runner.py              # Top-level 3-phase simulation runner
│   ├── phases.py              # Phase0BurnIn, Phase1Entry, Phase2StrategicGame classes
│   └── triggers.py            # Trigger abstract base; TimeTrigger, ProfitDropTrigger, RivalEventTrigger
│
├── llm/
│   ├── client.py              # LiteLLM wrapper; model snapshot pinning; per-call JSONL logging — unchanged
│   ├── schemas.py             # Pydantic schemas: ChainEnvelopeOutput, EntrantEntryDecision,
│   │                          #   ResponseFunction, QtableInitChoice — EXTEND significantly
│   └── prompts/
│       ├── system_ceo.jinja           # CEO system prompt (NEW — replaces system_pricing.jinja)
│       ├── system_entrant_entry.jinja # Entrant entry prompt (NEW)
│       ├── system_entrant_reassess.jinja  # Entrant reassessment prompt (NEW)
│       ├── state_ceo.jinja            # CEO market state serialiser (NEW — replaces state_template.jinja)
│       └── state_entrant.jinja        # Entrant market state serialiser (NEW)
│
├── data/
│   ├── osm_loader.py          # unchanged
│   ├── zensus_loader.py       # unchanged
│   └── distance.py            # unchanged
│
├── analysis/
│   ├── benchmarks.py          # Bertrand-Nash, joint monopoly — unchanged
│   ├── metrics.py             # Delta, price dispersion, per-group metrics (EXTEND)
│   └── animation.py           # unchanged
│
├── utils/
│   ├── seeding.py             # unchanged
│   └── logging.py             # unchanged
│
└── cli.py                     # Add `simulate` subcommand for 3-phase runner
```

### 8.2 Config structure (target)

```
configs/
├── config.yaml                    # Add `simulation` and `groups` to defaults list
│
├── agents/
│   ├── qlearning.yaml             # Rename/update: now store-level params only
│   ├── chain_ceo.yaml             # NEW: T_CEO, model snapshot, temperature, max_retries
│   ├── entrant_llm.yaml           # NEW: T_entrant, trigger thresholds, qtable_init_strategy
│   ├── llm_openai.yaml            # unchanged
│   └── llm_ollama.yaml            # unchanged
│
├── simulation/
│   ├── phases.yaml                # NEW: T_burnin, T_game, T_CEO, T_entrant
│   └── triggers.yaml              # NEW: profit_drop_threshold, rival_event_threshold, T_window
│
├── groups/
│   ├── no_groups.yaml             # NEW: active_divisions: []
│   ├── competition_only.yaml      # NEW: active_divisions: [DIVISION_COMPETITION]
│   ├── neighbourhood_only.yaml    # NEW: active_divisions: [DIVISION_NEIGHBOURHOOD]
│   └── competition_neighbourhood.yaml  # NEW: active_divisions: [DIVISION_COMPETITION, DIVISION_NEIGHBOURHOOD]
│
├── env/
│   ├── hotelling_1d.yaml          # unchanged
│   ├── uniform_2d.yaml            # unchanged
│   └── berlin_inner_ring.yaml     # UPDATE: scope changed from Pankow to inner-Ringbahn Berlin
│
└── sweep/
    ├── calvano_replication.yaml   # unchanged
    ├── transport_dose_response.yaml  # unchanged
    ├── llm_robustness.yaml        # unchanged
    └── group_treatment.yaml       # NEW: group configuration as treatment variable
```

### 8.3 Key schema changes (Pydantic)

**Remove:**
- `FirmDecision(price_index: int, rationale: str)` — too simple for the new architecture.

**Add:**
- `GroupEnvelope(p_bar, delta_p, e_bar, delta_e, epsilon)` — single group's envelope.
- `ChainEnvelopeOutput(chain_id, epoch, groups: dict[str, GroupEnvelope])` — CEO output.
- `ResponseFunction(base_price, base_effort, rival_undercut_response, profit_distress_response, envelope)` — entrant response function.
- `EntrantEntryDecision(theta_e, location_zone, location_site_index, response_function, qtable_init_preference)` — full entry output.
- `QtableInitChoice(use_pretrained: bool, chosen_store_id: str | None)` — LLM Q-table initialisation choice.
- `EntrantReassessOutput(response_function, rationale)` — reassessment output.

### 8.4 Test additions required

| Test file | What it tests |
|---|---|
| `tests/unit/test_envelope_validation.py` | GroupEnvelope Pydantic validation; min-width enforcement |
| `tests/unit/test_group_assignment.py` | Correct group labels assigned to stores given mock geography |
| `tests/unit/test_response_function_executor.py` | Deterministic response function execution against mock price/profit state |
| `tests/unit/test_trigger_system.py` | Time trigger fires at correct period; profit-drop trigger fires on threshold crossing |
| `tests/unit/test_qtable_relative_actions.py` | Relative action clipping at envelope boundary |
| `tests/integration/test_phase0_burnin.py` | Q-tables populate and policies stabilise within T_burnin |
| `tests/integration/test_phase1_entry.py` | Entrant instantiated correctly; Q-table initialisation options all produce valid tables |
| `tests/integration/test_3phase_runner.py` | Full 3-phase run completes with mock LLM client; Parquet output written correctly |

---

## 9. LLM Infrastructure and Computational Considerations

### 9.1 Supported LLM backends

The simulation supports two deployment modes, selectable via config:

**API mode (default for development):**
- Primary: Gemini 2.5 Flash-Lite (Google AI Studio free tier: 15 RPM, 1,000 RPD).
- Alternative: Gemma 3 27B via OpenRouter free tier ($0/M tokens).
- All calls routed through LiteLLM for provider abstraction.
- Rate limiting handled by the LiteLLM client with exponential backoff.

**Local mode (recommended for full-scale runs):**
- Gemma 3 27B or Gemma 4 31B via Ollama (no rate limits).
- 4-bit quantisation required for consumer GPU (24 GB VRAM): ~14–16 GB loaded.
- Inference speed: ~2–8 seconds per call on RTX 3090/4090 or Apple M-series (32 GB).
- Set `base_url: http://localhost:11434` in `llm_ollama.yaml`.

**Important constraints:**
- LLM model snapshots must be pinned in config — never use aliases (e.g. `gemma-3-27b-it`, not `gemma-latest`).
- Temperature must be set to `0` or minimum available for reproducibility.
- Every LLM call must be logged to JSONL: full prompt, full response, model snapshot, token counts, latency, call type (`CEO_EPOCH`, `ENTRANT_ENTRY`, `ENTRANT_REASSESS`, `QTABLE_INIT`).

### 9.2 LLM call volume (T_game = 5,000)

| Call type | Count | Tokens/call | Total tokens |
|---|---|---|---|
| CEO epoch (9 chains × 50 epochs) | 450 | ~1,010 | ~455,000 |
| Entrant reassessment (~100 calls) | 100 | ~1,200 | ~120,000 |
| Entrant entry + Q-table init | 2 | ~1,500 | ~3,000 |
| **Total** | **~552** | — | **~578,000** |

All 552 calls fit within a single day's free-tier quota (1,000 RPD for Flash-Lite). Wall time at 15 RPM: ~37 minutes. In local mode: ~46 minutes at 5 s/call.

### 9.3 Memory footprint (API mode, T_game = 5,000)

| Component | RAM |
|---|---|
| Distance matrix (9,000 cells × 150 stores, float32) | 5.4 MB |
| Q-tables (150 stores × 10,080 cells, float32) | 6.0 MB |
| Population/demographic grid | 0.4 MB |
| Peak transient (V tensor + probability tensor) | ~21 MB |
| History buffer (5,000 periods, 150 stores, 3 vars, streamed) | ~9 MB in-memory |
| Python + NumPy + LiteLLM + Pydantic overhead | ~225 MB |
| **Total** | **~267 MB** |

For `T_game > 10,000`, history must be streamed to disk (Parquet, append mode) rather than held in memory. In-memory buffer should hold at most the last `T_CEO` periods at any time.

### 9.4 Geographic scope

The simulation targets **inner-Ringbahn Berlin** (S41/S42 ring), not Pankow. The existing `berlin_pankow.yaml` config must be replaced or supplemented with `berlin_inner_ring.yaml`. Key parameters: ~8,000–9,000 Zensus demand cells, ~1,000–1,500 commercial candidate sites, ~100–150 incumbent stores across ~9 chains.

---

## 10. Decisions Carried Forward from Design Discussion

The following implementation decisions were made in the design process and must not be reopened without explicit justification:

| Decision | Rationale |
|---|---|
| Per-store independent Q-tables (not shared within chain) | Preserves within-chain price dispersion as an observable; mirrors real store manager independence |
| Relative action space (not absolute prices) | Q-tables transferable across CEO epochs; drops own-price from state space; reduces Q-table size |
| `gamma` fixed globally, not a CEO output | Changing `gamma` mid-simulation invalidates all existing Q-values; economically uninterpretable |
| CEO calls never batched across chains | Batching allows the LLM to observe rivals' private data and implicitly coordinate — destroys competitive structure |
| Burn-in before CEO activation | Q-tables need ~50,000 periods to converge; running LLM during burn-in would exhaust API budget on noise |
| Entrant commits to a response function (not per-period LLM) | Per-period LLM calls at T=1,000+ exceed free-tier daily quotas; response function preserves strategic reasoning at entry |
| Event triggers extensible via interface | Anticipated need to add new trigger types without refactoring entrant agent |
| Group membership assigned once at startup | Groups reflect structural store characteristics (location, neighbourhood); not a dynamic state variable |
| At most 2 simultaneous group divisions | Beyond 2 divisions, CEO action space grows to 32+ parameters; identification of effects becomes impossible |

---

## 11. Architecture Decision Records

Full ADR documents are in `docs/decisions/`. Each record covers context, decision, rationale, consequences, and alternatives rejected.

| ADR | File | Decision |
|-----|------|----------|
| ADR-001 | [ADR-001-src-layout.md](decisions/ADR-001-src-layout.md) | src layout for the Python package |
| ADR-002 | [ADR-002-llm-litellm-instructor.md](decisions/ADR-002-llm-litellm-instructor.md) | LiteLLM + Instructor for LLM integration |
| ADR-003 | [ADR-003-pettingzoo-env.md](decisions/ADR-003-pettingzoo-env.md) | PettingZoo ParallelEnv as simulation wrapper |
| ADR-004 | [ADR-004-per-store-independent-qtables.md](decisions/ADR-004-per-store-independent-qtables.md) | Per-store independent Q-tables; no sharing within chain |
| ADR-005 | [ADR-005-relative-action-space.md](decisions/ADR-005-relative-action-space.md) | Relative action space; Q-tables survive CEO epoch changes |
| ADR-006 | [ADR-006-three-phase-simulation.md](decisions/ADR-006-three-phase-simulation.md) | Three-phase structure; burn-in before CEO activation |
| ADR-007 | [ADR-007-llm-calls-not-batched.md](decisions/ADR-007-llm-calls-not-batched.md) | CEO calls never batched; information isolation argument |
| ADR-008 | [ADR-008-gamma-fixed-globally.md](decisions/ADR-008-gamma-fixed-globally.md) | Gamma fixed globally; not a CEO parameter |
| ADR-009 | [ADR-009-group-division-extensibility.md](decisions/ADR-009-group-division-extensibility.md) | Group divisions extensible via registry; at most 2 active |
| ADR-010 | [ADR-010-entrant-qtable-initialisation.md](decisions/ADR-010-entrant-qtable-initialisation.md) | Four Q-table init strategies for entrant; LLM meta-choice option |
| ADR-011 | [ADR-011-entrant-response-function.md](decisions/ADR-011-entrant-response-function.md) | Entrant commits to response function; not per-period LLM |
| ADR-012 | [ADR-012-inner-ring-not-pankow.md](decisions/ADR-012-inner-ring-not-pankow.md) | Geographic scope: inner-Ringbahn, not Pankow |
| Group divisions extensible via registry | Anticipated need for new divisions (e.g., rival-type based) without touching envelope or CEO logic |