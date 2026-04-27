# ADR 006 — Three-Phase Simulation Structure

**Status:** Accepted  
**Date:** April 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The simulation combines three agent types with very different computational requirements and timescales:

- Store RL agents require tens of thousands of periods to populate Q-tables and converge to stable policies.
- LLM-CEO agents require a stable market to reason about — their prompt inputs are meaningless if store prices are still in chaotic early-learning.
- The entrant LLM agent makes a one-shot entry decision that should be made against a realistic, converged incumbent market.

Running all three simultaneously from `t=0` creates conflicts: the LLM-CEO would spend its API budget on noisy, unconverged market states, and the entrant would enter a market that looks nothing like the converged steady state it will actually face during the strategic game.

---

## Decision

The simulation runs in three sequential phases:

- **Phase 0 — Burn-in:** All incumbent store RL agents run with fixed envelopes and no LLM calls until Q-tables converge (`T_burnin ≥ 50,000` periods). No entrant. Zero API cost.
- **Phase 1 — Entry:** A single LLM call instantiates the entrant. The entrant observes the converged incumbent market and makes its entry decision.
- **Phase 2 — Strategic game:** LLM-CEOs activate; the entrant's response function executes and is periodically reassessed. Duration `T_game ∈ [1,000, 5,000]` periods.

---

## Rationale

**The convergence problem.** Tabular Q-learning convergence requires that each state-action cell be visited a sufficient number of times to produce a reliable Q-value estimate. With ~10,080 cells per store and the recommendation of ~30 visits per cell for reliable estimates, each store needs on the order of 300,000 visits to its Q-table — which translates to tens of thousands of simulation periods. Running LLM-CEO calls during this convergence phase would consume the entire API budget (hundreds of calls per session) on states that are not representative of the converged market. The burn-in amortises this cost to zero.

**The CEO reasoning quality problem.** The LLM-CEO prompt includes own chain profit, mean price, and profit trend over the last `T_CEO` periods. During early learning, these signals are noise: prices jump randomly as agents explore, profits are far from any equilibrium level, and trends are meaningless. An LLM-CEO reasoning over this state will produce envelopes that are arbitrary rather than strategic. After burn-in, the market has settled: prices are near a stable range, profit signals are meaningful, and the CEO's reasoning is grounded in a real competitive situation.

**The entrant entry decision problem.** The entrant LLM is asked to choose a location and chain type by anticipating post-entry competitive dynamics. If the entrant enters during burn-in, it faces a non-converged market — the pricing behaviour it observes in its first periods bears no relation to the long-run equilibrium it will actually compete in. Phase 1 ensures the entrant enters a converged market and observes realistic prices and competitive structure from its first period.

**API budget discipline.** The free-tier daily quota (1,000 RPD for Gemini Flash-Lite) or local inference time is the binding constraint for `T_game > 5,000`. Restricting LLM calls to Phase 2 means the entire budget is spent on periods where agent behaviour is scientifically interesting, not on burn-in noise.

**The warm-start benefit.** Because store RL agents have converged Q-tables before Phase 2 begins, the CEO's first envelope in Phase 2 is immediately executed by agents that know how to respond to competitive signals. There is no "re-learning" cost when the CEO shifts the envelope — agents adapt quickly from their warm start.

---

## Consequences

- `simulation/phases.py` implements three distinct classes: `Phase0BurnIn`, `Phase1Entry`, `Phase2StrategicGame`, each with its own termination criterion.
- Phase 0 terminates on a convergence criterion (configurable: Q-value change below threshold over a rolling window), not a fixed period count. `T_burnin` is a maximum, not a target.
- The entrant's store RL agent cannot benefit from Phase 0 burn-in. Q-table initialisation for the entrant must be handled separately (see ADR 0011).
- Phase boundaries must be logged with timestamps and convergence metrics so that the Phase 0 duration is reproducible and reportable.
- `configs/simulation/phases.yaml` controls all phase parameters: `T_burnin`, `T_game`, `T_CEO`, `T_entrant`, convergence threshold.

---

## Alternatives Rejected

**Single-phase simulation with slow CEO activation.** Run everything from `t=0` but delay the first CEO call until after `T_burnin` periods. This is logically equivalent to the three-phase design but implemented as a single simulation loop with a conditional. Rejected because it conflates the conceptually distinct burn-in and strategic phases, making it harder to checkpoint, resume, and analyse each phase separately.

**DQN with faster convergence.** Deep Q-networks can converge faster than tabular methods in large state spaces. However, DQN convergence is not guaranteed by theory (unlike tabular Q-learning), introduces significant hyperparameter sensitivity, and breaks the Calvano validation benchmark. Rejected for the baseline; noted as a future extension if tabular convergence proves too slow.

**Pre-trained Q-tables from a prior run.** Initialise all stores from a cached Q-table generated in a previous simulation session, skipping Phase 0 entirely. This would reduce runtime significantly. However, pre-trained tables may embed the specific random seed, environment configuration, and envelope settings of the prior run, creating subtle confounds. Acceptable as a development-time speedup but not for production runs. Noted as an implementation option; not the default.