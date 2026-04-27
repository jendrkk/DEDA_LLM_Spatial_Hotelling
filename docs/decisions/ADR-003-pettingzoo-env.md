# ADR-003: Use PettingZoo ParallelEnv as the multi-agent simulation wrapper

**Status:** Accepted
**Date:** 2024-11

## Context

The simulation must support N heterogeneous agents (Q-learning, LLM, myopic,
random) operating simultaneously each period.  A standard gymnasium
single-agent `Env` does not capture simultaneous pricing decisions.

## Decision

Wrap the Hotelling market model as a **PettingZoo ParallelEnv**
(`HotellingMarketEnv`).  Observations are vectors of all agents' last-period
price indices.  Actions are single discrete price indices.  The environment
is designed to be headless (no graphical render) so it can run inside
multiprocessing workers for large-scale sweeps.

`gymnasium` and `pettingzoo` are optional dependencies under `[rl]`; the
environment class can be instantiated and tested without them (observation/
action space methods raise `NotImplementedError` until the full RL stack is
imported).

## Consequences

* Any RL library (Stable-Baselines3, RLlib) that supports PettingZoo can
  plug in directly.
* The `SimulationEngine` does **not** depend on PettingZoo internals; it
  calls `env.reset()` / `env.step()` duck-typed.
* The `HotellingMarketEnv` is stateful; parallel sweep workers each create
  their own instance.

## References

* https://pettingzoo.farama.org/
* Terry et al. (2021) PettingZoo: Gym for Multi-Agent Reinforcement Learning
* Calvano et al. (2020 AER) §III – simulation protocol
