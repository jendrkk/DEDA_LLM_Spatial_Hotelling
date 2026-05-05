# ADR 014 — Marginal Costs Normalised to Zero for All Chain Types in the Baseline

**Status:** Accepted  
**Date:** May 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The cost structure of the model (§D.1 of `economic_model_specification.md`) includes a chain-type-specific variable marginal cost $c_{\theta_c}$ per unit sold, with the prior expectation $c_D < c_S < c_B$ reflecting the well-documented ordering that discounters operate lean cost structures while Bio chains bear higher organic-sourcing and perishable-handling costs.

Three approaches to setting these costs were evaluated:

**Option A — Literature-calibrated chain-type costs from Bundeskartellamt sector inquiry.** The Bundeskartellamt Sektoruntersuchung Lebensmitteleinzelhandel (2014) and subsequent HDE Zahlenspiegel publications report gross margins by retail format: Discount ~25–28%, Standard ~22–25%, Bio ~30–40%. Converting to marginal cost requires knowing the price level, which requires knowing the equilibrium — creating a circular dependency.

**Option B — Normalise c_D = 0 as numeraire, express c_S = δ_S and c_B = δ_B as relative premiums.** This is the approach initially explored: set the Discount marginal cost to zero and calibrate the Bio premium δ_B from the observed ~70% Bio-over-Discount price premium in the market. This decomposition turns out to be *mathematically incorrect* (see Rationale below).

**Option C — Set all c_θ = 0, following Calvano et al. (2020).** The primary methodological inspiration for this project (Calvano, Calzolari, Denicolo, Pastorello 2020) sets $c = 0.25$ not because it is an empirically identified value but as a normalisation. All results in that paper are relative to the Nash benchmark and the collusion metric Δ, both of which are invariant to uniform rescaling of the cost level.

---

## Decision

**Set $c_D = c_S = c_B = 0$ for all chain types in the baseline model.** This is a normalisation, not an empirical claim about actual costs.

A single **robustness run** (Sensitivity Run 4) uses BKartA-derived gross-margin estimates as cost inputs to verify that the primary collusion results are not artefactual to the $c = 0$ normalisation.

---

## Rationale

### Why Option B (c_D = 0 numeraire) is mathematically incorrect

In the multinomial logit Nash equilibrium, each chain type's price satisfies the first-order condition:
$$p_\theta^N = c_\theta + \frac{\mu}{1 - s_\theta}$$

The observed price difference between Bio and Discount is therefore:
$$p_B^N - p_D^N = \underbrace{(c_B - c_D)}_{\text{cost component}} + \underbrace{\mu\!\left[\tfrac{1}{1-s_B} - \tfrac{1}{1-s_D}\right]}_{\text{demand component}}$$

To decompose the observed ~70% Bio premium into a cost component and a demand component, one needs to know $\mu$, $s_B$, and $s_D$ — but these are themselves functions of $c_B$ and $c_D$ through the equilibrium. There is no off-equilibrium shortcut: the system must be solved simultaneously. Option B incorrectly assumed that $\delta_B \approx 0.35\bar{p}_D$ follows from a simple proportion of the observed price gap, but this only holds if the demand component is known independently, which it is not without a full equilibrium solution.

### Why Option C is correct for this project

1. **The project's primary contribution is the agent architecture, not the measurement of the Berlin cost structure.** Setting $c = 0$ is a deliberate modelling choice — consistent with Calvano et al. and the broader IO-simulation literature — that removes one calibration dimension without changing the scientific question.

2. **The price ordering is preserved by the demand side.** With $c_\theta = 0$, the Nash equilibrium price ordering $p_D^N < p_S^N < p_B^N$ still emerges from the demand side: Bio stores carry higher quality intercepts ($q_B > q_S > 0$) that attract high-WTP consumers ($\alpha_H$), enabling them to sustain higher prices at equilibrium even at zero marginal cost.

3. **All primary metrics are self-normalised.** The Calvano Δ collusion metric is defined relative to the joint-monopoly and Nash benchmarks:
   $$\Delta = \frac{\bar{\pi} - \pi^N}{\pi^M - \pi^N}$$
   Both $\pi^N$ and $\pi^M$ scale identically with a uniform cost shift, so $\Delta$ is invariant. Similarly, the IRF is defined in terms of price deviations relative to the pre-deviation strategy, which is unaffected by cost level. The scientific results are invariant to the normalisation.

4. **Sanity check is straightforward.** With $c = 0$, verify before the first full run that the simulated Nash price ratio $p_B^N / p_D^N \in [1.5, 1.8]$, consistent with the real-world premium structure for Berlin grocery. If this holds, the $c = 0$ specification is capturing the right price ordering.

---

## Consequences

- `configs/model/costs.yaml` sets `c_discount: 0.0`, `c_standard: 0.0`, `c_bio: 0.0`.
- `src/hotelling/core/market.py` — the `profit` computation uses `(price - c) * demand - kappa(effort) - R_ell - F_c/n_stores`, with `c = 0` loaded from config. No code changes are needed; only config.
- The thesis methods section states explicitly: "Marginal costs are normalised to zero for all chain types, following Calvano et al. (2020). Prices and profits are therefore measured in units of the logit scale parameter μ. The price premium between chain types is generated entirely by the demand side (quality intercepts and consumer WTP heterogeneity). A robustness run with BKartA-calibrated marginal costs (Discount ≈ 0.25μ, Standard ≈ 0.22μ, Bio ≈ 0.35μ) confirms results are qualitatively unchanged."
- Sensitivity Run 4 implements $c_D < c_S < c_B$ using BKartA-implied ratios. This is configured via `configs/sensitivity/costs_nonzero.yaml`.

---

## Alternatives Rejected

**Option A — Full BKartA-calibrated chain costs.** Requires solving the joint equilibrium system to convert gross margins to per-unit costs given equilibrium prices and shares — a system of nonlinear equations in $(\mu, c_D, c_S, c_B, q_S, q_B, s_D, s_S, s_B)$. This is identifiable in principle (the system is over-determined with the right moment conditions) but requires 8 jointly calibrated parameters with Berlin-specific data that do not exist at the store level. Rejected for the baseline; retained as a robustness sensitivity run.

**Option B — c_D = 0 numeraire, relative c_S and c_B.** Rejected on mathematical grounds: the decomposition of the observed price premium into cost and demand components requires solving the equilibrium, not inverting a simple formula.

**Chain-specific μ_θ instead of chain-specific c_θ.** An alternative way to generate price differentiation: Bio consumers are more brand-attached (lower effective μ_B), which sustains higher Bio prices without requiring cost differences. This is economically motivated but adds a 3rd calibration dimension (three μ parameters instead of one). Noted as a potential robustness extension, not the baseline.
