# ADR 011 — Entrant Uses a Committed Response Function, Not Per-Period LLM Calls

**Status:** Accepted  
**Date:** April 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The original project design proposed operating the entrant via LLM every period — the entrant LLM would receive the current market state each period and output a price and effort decision. This would make the entrant a fully reactive LLM agent, contrasting sharply with the Q-learning incumbents.

This design faces a hard computational constraint: at `T_game = 1,000` periods, per-period LLM calls generate 1,000 API requests for the entrant alone. At 1,000 RPD (free tier), this consumes the entire daily quota before any CEO calls are made. At `T_game = 5,000`, per-period calls require 5 days of quota just for the entrant. This is not feasible for iterative research use.

Beyond the budget problem, the scientific value of per-period LLM pricing is questionable: an LLM choosing a price each period based on last-period rivals' prices is doing a task that tabular Q-learning does better (faster convergence, lower variance, reproducible). The LLM's genuine comparative advantage is in strategic reasoning about positioning, not in reactive tactical pricing.

---

## Decision

The entrant LLM outputs a **response function** at entry time and on each reassessment event. The response function is a structured conditional pricing rule that executes deterministically each period without further LLM calls. The LLM is called only at: (a) the initial entry decision (Phase 1), (b) periodic time-triggered reassessments (every `T_entrant = 50` periods), and (c) event-triggered reassessments (profit drop or rival event, whichever comes first). Each reassessment outputs a new response function that replaces the previous one.

---

## Rationale

**API budget.** Per-period calls at `T_game = 1,000` require 1,000 calls for the entrant alone — the entire free-tier daily quota. The response function approach reduces entrant calls to approximately `T_game / T_entrant = 100` calls for `T_game = 5,000`, consuming ~10% of the daily quota. This makes multi-session experiments feasible within free-tier constraints.

**Scientific coherence.** The LLM's value in this model is in *strategic reasoning* — choosing a market position, anticipating competitive responses, and periodically reassessing strategy. These are tasks that require language understanding, economic reasoning, and memory over long time horizons. Reactive per-period pricing is none of these things; it is simply best-responding to a numerical state, which tabular Q-learning does with theoretical convergence guarantees. Delegating tactical pricing to the response function and reserving LLM calls for strategic reassessments is more faithful to the model's economic motivation (CEO/manager delegation).

**Economic realism.** Real entrepreneurs do not consult a strategic advisor every day. They form a business plan, execute it for a period, observe outcomes, and periodically reassess. The response function models this: the entrant commits to a conditional strategy (if the rival undercuts by more than X, cut own price by Y) and executes it faithfully until the next reassessment. This mirrors the LLM-CEO architecture for incumbents, where the CEO sets an envelope that stores execute mechanically between CEO epochs.

**The committed strategy is itself a finding.** The response function the LLM chooses at entry is observable and interpretable: it is the entrant's stated theory of how it will compete. Comparing this stated strategy to the strategy that actually emerges from the entrant's RL tactical layer (if present) is itself a scientifically interesting comparison. Per-period LLM calls produce no such artefact.

---

## Response Function Schema

```json
{
  "base_price": float,
  "base_effort": float,
  "rival_undercut_response": {
    "threshold": float,
    "own_price_adjustment": float
  },
  "profit_distress_response": {
    "profit_threshold": float,
    "own_price_adjustment": float
  },
  "envelope": {
    "p_bar": float, "delta_p": float,
    "e_bar": float, "delta_e": float,
    "epsilon": float
  }
}
```

The response function executor is a deterministic interpreter: given the current market state (own last price, nearest rival's last price, own last profit), it applies the conditional rules in order and returns `(price_adjustment, effort_level)`. All outputs are clipped to the current envelope.

---

## Consequences

- `agents/entrant_llm.py` implements two distinct components: (a) the `EntrantLLM` agent that makes LLM calls and returns `EntrantEntryDecision` and `EntrantReassessOutput`, and (b) the `ResponseFunctionExecutor` that interprets the response function deterministically each period.
- The entrant's store RL agent (if present) receives the response function's envelope as its operating envelope — the response function sets the strategic context; the RL agent executes tactically within it.
- `simulation/triggers.py` implements the trigger system. Triggers are extensible via the `Trigger` abstract base class (see below). Adding a new trigger requires only implementing the interface and registering it.
- LLM calls for the entrant are logged with `call_type ∈ {"ENTRANT_ENTRY", "ENTRANT_REASSESS"}`.
- The response function is logged at each update: previous function, new function, trigger that fired, period.

---

## Trigger System Interface Contract

```python
class Trigger(ABC):
    @abstractmethod
    def should_fire(self, entrant_state: EntrantState, period: int) -> bool:
        """Return True if a reassessment should be triggered this period."""
        ...
```

Baseline triggers:
- `TimeTrigger(period_interval: int)` — fires every `T_entrant` periods.
- `ProfitDropTrigger(threshold_pct: float, window: int)` — fires when rolling mean profit drops by more than `threshold_pct` relative to the prior window.
- `RivalEventTrigger(price_change_threshold: float, radius: float)` — fires when any rival's mean price changes by more than `threshold_pct` in one CEO epoch, or when a rival store opens/closes within `radius` metres.

Multiple triggers are active simultaneously. The reassessment fires when *any* trigger returns `True`. The specific trigger that fired is recorded in the log.

---

## Alternatives Rejected

**Per-period LLM calls.** The original design intent. Rejected due to API budget infeasibility (1,000+ calls per `T_game = 1,000` session) and the availability of a more scientifically coherent alternative. Noted: if the project later moves to fully local LLM inference with no rate limits, per-period calls become computationally feasible and could be implemented as an alternative entrant strategy for comparison.

**Fixed strategy with no reassessment.** The entrant outputs a response function at entry and commits to it for the entire Phase 2 with no further LLM calls. This is the minimum-cost version (1 total LLM call for the entrant). Rejected because it prevents the entrant from responding to significant market developments — which is both economically unrealistic and scientifically uninteresting.

**Entrant uses Q-learning with no LLM after entry.** The entrant becomes a pure RL agent after Phase 1, indistinguishable from incumbents. This eliminates the LLM vs. RL contrast that motivates the entrant's design. Rejected; if pure RL entrant behaviour is desired, it should be implemented as a separate experimental treatment for comparison, not as the default.