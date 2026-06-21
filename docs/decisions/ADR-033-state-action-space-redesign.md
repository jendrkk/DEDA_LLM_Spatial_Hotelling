# ADR-033: State & Action Space Redesign for Collusion-Enabling Q-Learning

## Status
Accepted

## Context

Phase 0 Q-learning with the `local_summary` state mode produced Q-tables with
three pathologies that prevented the observation of algorithmically-mediated
collusion:

1. **State distribution collapse** — the vast majority of the $n_\text{bins}^C$
   states were never visited after convergence.  With $C=1$ and $n_\text{bins}=15$
   this leaves 14 of 15 states permanently at Q=0.

2. **Action-invariant Q-value stripes** — in visited states,
   $Q(s,a) \approx \text{const}$ across all $a$.  The policy is effectively
   random inside visited states because the agent cannot distinguish which action
   produced a good outcome.  Root cause: the state did not encode the store's
   *own* previous price, so the Q-learner could not learn the causal link
   $\text{own action} \to \text{competitor response} \to \text{own reward}$.

3. **Negligible per-store market power** — with $N \approx 494$ stores the
   unilateral deviation of a single store changes the local price index by
   $\approx 1/N$, which is below the noise floor of the logit demand system at
   the default price-grid resolution.

## Decision

### 1. Four new state-space designs (CLI flags)

| Mode | Flag | State tuple | State size |
|------|------|------------|------------|
| `design4_ownprice` | `--base-states B` | $(b_\text{own},\, b_\text{same})$ | $m \times B$ |
| `design5_full` | `--full-states B` | $(b_\text{own},\, b_\text{same},\, b_\text{cross})$ | $m \times B^2$ |
| `calvano_local` | `--calvano-states K` | $(b_\text{own},\, b_{r_1},\ldots,b_{r_K})$ | $m^{K+1}$ |
| `strategic_hybrid` | `--strategic-states B` | $(b_\text{own},\, b_\text{same},\, R)$ | $m \times B \times 3$ |

Notation:
- $b_\text{own}$: own previous price index $\in \{0,\ldots,m-1\}$
- $b_\text{same}$: demand-overlap-weighted mean price of same-chain-type
  competitors, discretised into $B$ bins
- $b_\text{cross}$: same for cross-chain-type competitors
- $b_{r_k}$: previous price index of the $k$-th nearest same-chain-type rival
  (ranked by shared demand cells)
- $R \in \{0, 1, 2\}$: market regime (competitive / neutral / supra-competitive),
  classified by all-type local mean vs store-specific Nash/mono benchmarks

Recommended default: `--base-states 15` (`design4_ownprice`).

### 2. Chain-type-specific price grids (`--chs-grid`)

Each chain type (discount / standard / bio) gets its own `linspace` grid
spanning from the chain's marginal cost (or Nash floor) to above its
joint-monopoly price.  The global `price_grid` is retained as the union range
for backward compatibility (bin edges, checkpointing, plotting).

### 3. Demand-overlap weighted competitor sets

`src/hotelling/env/competitors.py` constructs:
- A symmetric $(N,N)$ integer matrix counting shared demand cells between stores
- Same-type and cross-type CSR sub-graphs with float64 weights
- $(N, k_\text{max})$ nearest-same-type rival index array

Numba `@njit` kernels `weighted_mean_prices` and `bin_prices` compute the
competitor signal each step without Python overhead.

### 4. Dynamic β adaptation (`--no-auto-beta` opt-out)

For $T_\text{burnin} > 10^6$, the exploration decay rate is automatically set
to $\beta = -\ln(\varepsilon_\text{target}) / (f_\text{explore} \cdot T)$
with $\varepsilon_\text{target}=0.02$ and $f_\text{explore}=0.75$.
This ensures exploration persists through 75 % of the run regardless of its
length — the Calvano (2020) default $\beta=4\times10^{-6}$ gives effective
zero exploration after $T=1\text{M}$ on multi-million-step runs.

## Consequences

**Positive:**
- `design4_ownprice` gives the learner causal credit assignment, enabling
  detection of algorithmic collusion even at $N \approx 494$.
- Chain-specific grids concentrate the action space on economically relevant
  prices for each segment, improving Q-table coverage.
- Dynamic β prevents premature convergence on long runs.

**Negative / risks:**
- `calvano_local` with $K=3$ produces $m^4 = 50{,}625$ states per store;
  with $N=494$ stores and $m=15$ actions this requires $\approx 3$ GiB RAM.
  The existing `max_qtable_gib=8.0` guard in `BatchQLearningAgent` catches
  out-of-memory configurations.
- New state modes require `dense_distances=True` for the demand-overlap CSR;
  the Euclidean fallback is used automatically when catchment data are absent
  but produces unweighted (binary) competitor sets.

**Backward compatibility:**
- All legacy state modes (`neighbors`, `local_summary`) are fully preserved.
- Default invocation (no state-mode flag) is identical to pre-ADR-033 behaviour.
- `HotellingMarketEnv.step_array()` signature is unchanged.

## References
- Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020).
  Artificial intelligence, algorithmic pricing, and collusion. *AER*, 110(10).
- ADR-029 (local-summary state), ADR-030 (detailed local-summary)
