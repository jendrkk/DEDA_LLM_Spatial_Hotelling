# ADR 009 — Group Division System Is Extensible via Registry

**Status:** Accepted  
**Date:** April 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The CEO differentiates its strategy envelope across store groups. Two group divisions are implemented in the baseline:

- `DIVISION_COMPETITION` — classifies stores as `HEAVY` or `EASY` competition based on rival store count within radius `R`.
- `DIVISION_NEIGHBOURHOOD` — classifies stores as `RICH` or `POOR` based on the LOR social-status index `S_r` of the store's neighbourhood.

During design, additional division types were identified as likely future needs:

- `DIVISION_RIVAL_TYPE` — classify stores by the type of their nearest competitor (`{D, S, B}`), allowing the CEO to differentiate strategy based on whether it competes against discount or premium rivals.
- `DIVISION_PROXIMITY_TO_ENTRANT` — classify stores by proximity to the entrant's location, allowing the CEO to respond asymmetrically near and far from the new competitor.
- Others not yet identified.

The division system must accommodate these future additions without requiring changes to the envelope module, the CEO agent, or the simulation runner.

---

## Decision

Group divisions are implemented as a **registry pattern** with an abstract base class. Each division is a self-contained module in `hotelling/envelope/divisions/`. Adding a new division requires only: (a) creating a new file in that directory implementing the `GroupDivision` interface, (b) registering it by name, and (c) adding a config entry. No other files need to change.

---

## Rationale

**The division types are structurally identical.** Every division does the same thing: takes a store's metadata (location, LOR assignment, competitive environment) and returns a category label (`str`). The logic that differs is only the classification criterion. This is a textbook case for an abstract base class with concrete implementations.

**The CEO and envelope modules should not contain division logic.** The CEO's job is to output envelope parameters for a given set of group labels. The envelope module's job is to associate parameter sets with group labels. Neither should contain conditional logic that checks which divisions are active. All division-specific logic must be encapsulated in the division modules themselves.

**Config-driven activation.** The set of active divisions is specified in the Hydra config (`configs/groups/*.yaml`). The simulation runner reads the active division list, instantiates the corresponding division objects from the registry, and assigns group labels to all stores at Phase 0 initialisation. The CEO receives the group labels as part of each store's metadata — it does not need to know how the labels were produced.

**At most 2 simultaneous divisions.** This constraint is enforced at config validation time, not at runtime. With 2 divisions of 2 categories each, the CEO outputs 4 group envelopes (20 parameters). Beyond 2 divisions, the CEO action space grows to 8+ groups (40+ parameters), making identification of effects impossible: the LLM cannot reason coherently about that many degrees of freedom, and statistical power to distinguish group effects collapses. This constraint is a research design decision, not a technical limitation.

---

## Interface Contract

Every division implementation must satisfy:

```python
class GroupDivision(ABC):
    name: str                  # unique registry key, e.g. "DIVISION_COMPETITION"
    categories: tuple[str, str]  # exactly 2 labels, e.g. ("HEAVY", "EASY")

    @abstractmethod
    def assign(self, store_metadata: StoreMetadata) -> str:
        """Return the category label for a single store. Must return one of self.categories."""
        ...

    @abstractmethod
    def description(self) -> str:
        """Human-readable description included in the CEO system prompt."""
        ...
```

`StoreMetadata` contains all information available at Phase 0 initialisation: store location (EPSG:25833), chain type, LOR assignment, pre-computed rival count within radius `R`, LOR social-status index. No per-period information.

Group labels are **fixed at Phase 0 and never change** during the simulation. They are stored in the store registry as static metadata, not recomputed each period.

---

## Consequences

- `hotelling/envelope/divisions/` contains one file per implemented division. The baseline ships with `competition.py` and `neighbourhood.py`.
- `hotelling/envelope/groups.py` contains the `GroupDivision` ABC, the registry (`dict[str, type[GroupDivision]]`), and the group assignment logic that iterates over active divisions and assigns each store a composite group key (e.g., `"HEAVY_RICH"`).
- `hotelling/envelope/envelope.py` contains `GroupEnvelope` and `ChainEnvelope` dataclasses. `ChainEnvelope` is a `dict[str, GroupEnvelope]` keyed by composite group label.
- The CEO prompt template dynamically renders the active group structure from the config. If 0 divisions are active, it renders a single `"default"` group. The template logic for this is in `llm/prompts/system_ceo.jinja`.
- Adding a new division: create `hotelling/envelope/divisions/my_division.py`, implement `GroupDivision`, add `"MY_DIVISION": MyDivision` to the registry in `groups.py`, create `configs/groups/my_division_config.yaml`. No other changes.

---

## Alternatives Rejected

**Hardcoded if/elif chains in the envelope module.** Adding a new division would require editing `envelope.py` and `chain_ceo.py`. Rejected as non-extensible.

**Division logic in the CEO agent.** The CEO agent would need to know how divisions are defined in order to construct appropriate prompts. This tightly couples the LLM layer to the domain classification logic. Rejected — the CEO only knows group labels, not how they were produced.

**More than 2 categories per division.** For example, `DIVISION_COMPETITION` with categories `{EASY, MEDIUM, HEAVY}`. Structurally possible within the registry design, but the interface contract requires exactly 2 categories per division to maintain the constraint that the total group count is at most 2² = 4. Three-category divisions would require 2³ = 8 CEO output groups with 2 active divisions — too many parameters for reliable LLM reasoning. Rejected; the interface enforces exactly 2 categories.