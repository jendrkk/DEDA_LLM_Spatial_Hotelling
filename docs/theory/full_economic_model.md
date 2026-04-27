# Economic Model Specification — Agents in the City (Berlin Grocery, Hierarchical LLM/RL Agents)

*Technical memo. Drafted to be simulation-ready against the agent architecture you specified: LLM-CEO sets a strategic envelope every $T_{\mathrm{CEO}}$ periods; per-store RL agents act inside the envelope each period; entrant runs on LLM at both layers. Geographic frame: inner-Ringbahn Berlin. Industry: full-line grocery + discount + bio.*

---

## Modeling philosophy

Three commitments shape every choice below.

**(i) Tractable demand, rich agency.** Demand is a smooth multinomial logit with a quadratic spatial term. This is the same family as Calvano (2020) and the seminar baseline; it guarantees a unique Bertrand–Nash benchmark, smooth gradients for RL, and avoids the d'Aspremont (1979) non-existence problem. The economic richness is loaded into the **agent layer** (CEO-LLM + per-store RL) rather than into demand. Where the demand side gets richer — vertical chain types, consumer WTP heterogeneity from LOR social-status indices, transient prime-location footfall — the additions are *additive shifters* in the utility, never structural rewrites of the choice probability.

**(ii) Match the level of decision-making to the level of information.** The CEO sees aggregate market and consumer-side information; therefore the CEO's variable is a chain-level strategic envelope $(\bar p_c, \delta^p_c, \bar e_c, \delta^e_c)$. The store operator sees only local performance and local rivals; therefore its variable is a within-envelope $(p_{jt}, e_{jt})$ pair, learned by RL. This is internally consistent with how delegated authority is modelled in organizational economics (Aghion–Tirole 1997).

**(iii) Anything that cannot be calibrated from open Berlin data is held back as an extension.** Quality differentiation and prime-location footfall make the cut because OSM, Zensus 2022, LOR social-status indices, VBB GTFS, and Bodenrichtwerte cover them. Multi-store dynamic expansion is set up structurally but **not** part of the baseline run.

What I cut from your scope without strong protest: a continuous quality index (you chose discrete; correct), endogenous chain marginal costs (calibration only), search frictions (no defensible Berlin data), time-of-day dynamics (a single static prime-location shifter does the same job).

---

## A. Executive summary

The model is a discrete-time, infinite-horizon spatial Bertrand game on the inner-Ringbahn Berlin grid. Each chain $c$ is one of three vertical types $\theta_c \in \{D, S, B\}$ — discount, standard, bio — that fixes its quality intercept and its own constant marginal cost. Each chain operates one or more stores indexed by $j$, each at a fixed location $\ell_j$. Consumers live in 100 m × 100 m Zensus cells, partitioned into low- and high-status types via LOR-level social-status indices, and choose stores by logit-weighted full price (price + quadratic transport cost − chain quality − store effort).

Decisions are split across two timescales. At the **slow timescale** (period $\tau \in \{0, T_{\mathrm{CEO}}, 2 T_{\mathrm{CEO}}, \dots\}$), each chain's LLM-CEO observes aggregate state and outputs a *strategy envelope* $\sigma_c = (\bar p_c, \delta^p_c, \bar e_c, \delta^e_c)$. At the **fast timescale** (every period $t$), each store's RL agent observes local state and chooses $(p_{jt}, e_{jt})$ inside the envelope. The entrant is operated by LLM at both layers because it begins with one store; if and when it expands, new stores are added to the same RL pool.

Entry is a one-shot stage at $t = 0$: the entrant LLM picks $\theta_e \in \{D, S, B\}$ and a location $\ell_e$ from a coarse candidate-site grid. Repeated play follows. Multi-store expansion (incumbents and entrant alike) is a future extension specified below but not part of the baseline run.

The benchmarks reported each session are the static Bertrand–Nash and joint-monopoly profits given the realized location/type configuration; the headline outcome is the Calvano profit gain $\Delta$ and a price-dispersion metric induced by the store-level RL within the chain envelope.

---

## B. Core economic environment

### B.1 Index conventions

| Symbol | Meaning | Level |
|---|---|---|
| $i \in \mathcal{I}$ | demand cell (100 m × 100 m) | location |
| $\ell \in \mathcal{L}$ | candidate store site (coarser, e.g. 250 m, restricted to commercial zoning) | location |
| $h \in \{L, H\}$ | consumer type (low / high social status) | consumer |
| $r \in \mathcal{R}$ | LOR Planungsraum (~542 in Berlin, ~150 in inner Ring) | location |
| $c \in \mathcal{C}$ | chain (incumbent or entrant) | chain |
| $j \in \mathcal{J}_c$ | store of chain $c$, at location $\ell_j$ | store |
| $\theta_c \in \{D, S, B\}$ | chain type (discount / standard / bio) | chain |
| $t \in \{0, 1, 2, \dots\}$ | period | time |
| $\tau \in \{0, T_{\mathrm{CEO}}, 2 T_{\mathrm{CEO}}, \dots\}$ | CEO-update epoch | time |

### B.2 Decision architecture

Three layers, ordered from slow to fast.

**Entry stage ($t = 0$, once).** Entrant LLM chooses $(\theta_e, \ell_e)$. Pays sunk cost $F^{\mathrm{entry}}_{\theta_e}$. Incumbent locations and types are data.

**Strategic stage (every $T_{\mathrm{CEO}}$ periods).** Each chain's CEO-LLM observes aggregate state $S^{\mathrm{CEO}}_{c\tau}$ (defined in §I) and outputs $\sigma_c = (\bar p_c, \delta^p_c, \bar e_c, \delta^e_c)$ with $\delta^p_c, \delta^e_c \geq 0$.

**Tactical stage (every period).** Each store's RL agent observes local state $S^{\mathrm{store}}_{jt}$ and chooses $(p_{jt}, e_{jt})$ subject to
$$p_{jt} \in [\bar p_c - \delta^p_c,\ \bar p_c + \delta^p_c], \qquad e_{jt} \in [\max\{0, \bar e_c - \delta^e_c\},\ \bar e_c + \delta^e_c].$$
Discretize each interval to ~5 actions for the RL action set.

The market clears each period: the demand system in §C delivers $q_{jt}$, profits $\pi_{jt}$ accrue, and RL updates run. The CEO's reward is the realised chain profit summed over the next $T_{\mathrm{CEO}}$ periods.

### B.3 Why this works

- **Action-space discipline.** Without the envelope, the joint per-period chain action vector is $2 |\mathcal{J}_c|$-dimensional and chain-coordinated pricing is intractable for RL. The envelope gives the CEO a low-dimensional ($\mathbb{R}^4$) strategic instrument while preserving local responsiveness at the store layer.
- **Information segregation is economically meaningful.** A store manager who does not see consumer demographics and only sees nearby competitors mirrors actual delegation in retail chains; the CEO who sees everything but acts only quarterly mirrors HQ.
- **Backward-compatibility with Calvano.** If you set $|\mathcal{J}_c| = 1$ for all $c$, $T_{\mathrm{CEO}} = \infty$ (no CEO), $\delta^p = \infty$, transport cost $t = 0$, effort frozen, and one consumer type — you recover Calvano (2020) exactly. This is your sanity check.

---

## C. Consumer demand specification

### C.1 Utility

Consumer of type $h$ in cell $i$, choosing store $j$ of chain $c(j)$ with type $\theta_{c(j)}$:
$$U_{hijt} = \alpha_h \, \mathbf{q}_{\theta_{c(j)}} + \beta\, e_{jt} - p_{jt} - t\, d_{ij}^2 + \varepsilon_{hijt}.$$

- $\mathbf{q}_{\theta} \in \{q_D, q_S, q_B\}$ — chain-type quality intercept, normalised so $q_D = 0$.
- $\alpha_h$ — type-$h$ marginal utility of chain quality (the *vertical preference*); $\alpha_H > \alpha_L \geq 0$.
- $\beta$ — homogeneous marginal utility of store effort (per your spec: clean is clean, regardless of who runs the store).
- $e_{jt}$ — store effort, $e_{jt} \in [0, \bar e^{\max}]$.
- $p_{jt}$ — store price.
- $d_{ij}$ — Euclidean (or OSM-network) distance from cell $i$'s centroid to store $\ell_j$, in km.
- $t$ — transport cost, $€/km^2$, quadratic to preserve equilibrium existence.
- $\varepsilon_{hijt}$ — i.i.d. Type-1 EV with scale $\mu$.

The outside option (no purchase / shopping outside the simulation, e.g., at periphery stores or online) has utility $\varepsilon_{hi0t}$ with intercept $a_0 \leq 0$.

### C.2 Choice probability and aggregate demand

By the standard logit derivation,
$$\Pr(j \mid h, i; \mathbf{p}, \mathbf{e}) = \frac{\exp\!\big(V_{hij}/\mu\big)}{\exp(a_0/\mu) + \sum_{k \in \mathcal{F}_t} \exp(V_{hik}/\mu)}, \qquad V_{hij} \equiv \alpha_h q_{\theta_{c(j)}} + \beta e_{jt} - p_{jt} - t d_{ij}^2.$$

Aggregate store-$j$ demand at $t$:
$$q_{jt} = \sum_{i \in \mathcal{I}} \big(\omega_i + \lambda \phi_i\big) \sum_{h \in \{L,H\}} \pi_{hi}\, \Pr(j \mid h, i),$$
where $\omega_i$ is residential population in cell $i$ (Zensus 2022), $\phi_i$ is the prime-location index (§G), $\lambda$ converts the prime index into "equivalent residents", and $\pi_{hi}$ is the share of consumers of type $h$ in cell $i$.

### C.3 Consumer heterogeneity from LOR social status

This is the design your recent memory pinned, and I think it is the right move — better than pure income proxies, because the Berlin LOR social-status index is *purpose-built* by the SenStadtUm to summarize the social composition of small areas (it is a composite of unemployment, transfer-recipient share, child-poverty rate, school-leaver share, etc.). Your equation needs a clean discipline for mapping the index to type weights.

**Recommended mapping.** Let $S_r \in [0, 1]$ be the standardised LOR social-status index (0 = lowest, 1 = highest). For every cell $i$ in LOR $r$:
$$\pi_{Hi} = S_{r(i)}, \qquad \pi_{Li} = 1 - S_{r(i)}.$$

This treats the LOR index as the local probability that a random shopper is a high-status (high-WTP-for-quality) shopper. It is interpretable, monotone, unit-free, and uses the index at exactly the resolution it was designed for. It also implies *non-degenerate type mixing within every cell*, which keeps demand smooth — a strict separator (cells "above 0.5" → all H) would make the equilibrium ill-behaved at the boundary.

**Why this is better than a postcode income proxy.** Berlin postcodes (PLZ) are coarser than LORs, are not designed for socio-economic measurement, and confound rich and poor blocks. Mietspiegel rent indices are biased by rent control and tenancy duration, not present demographics. The LOR index is the right object.

**Where this can break.** The index is updated every two years and small-area releases lag. Use the most recent vintage (2023 or 2024 release) and treat $S_r$ as static for the simulation horizon. If you need within-LOR heterogeneity later, *Bodenrichtwerte* at cell level can serve as a continuous shifter on top of the discrete LOR index.

### C.4 Two types vs. continuous

Two types is enough for the baseline. Going to a continuous distribution of $\alpha$ (mixed logit) requires Monte-Carlo integration each period and roughly 10× the runtime — not justified before you have a working pipeline. Note this limitation in §P.

### C.5 Within-cell distance

100 m cells are small enough that the centroid approximation is innocuous *for stores outside the cell*. For stores *inside the same cell as the consumer*, set $d_{ij} = d^{\mathrm{intra}}$ with $d^{\mathrm{intra}} = c_{\mathrm{cell}}/3 \approx 33\text{m}$ (the expected distance from a uniform point to the centre of a square of side $c_{\mathrm{cell}}$, scaled). At 100 m resolution, $t \cdot (d^{\mathrm{intra}})^2 \approx 0.0005 \cdot t$ — negligible. If you ever go to 500 m cells, this matters and the same formula gives $\approx 0.028 \cdot t$, still small but no longer trivial. Honest treatment, no fudge.

---

## D. Firm / chain / store supply specification

### D.1 Cost structure

For store $j$ of chain $c$ at period $t$:
$$\mathrm{TC}_{jt} = c_{\theta_c} \cdot q_{jt} + \kappa(e_{jt}) + R_{\ell_j} + \frac{F_c}{|\mathcal{J}_c|},$$

where:
- $c_{\theta_c}$ — chain-type marginal cost, $c_D < c_S < c_B$ (calibration parameter, see §J).
- $\kappa(e) = \tfrac12 \kappa_0 e^2$ — homogeneous quadratic effort cost (per your spec, hiring cleaners costs the same anywhere; $\kappa_0$ is a chain-invariant constant).
- $R_{\ell_j}$ — periodic rent at store location (§H).
- $F_c$ — chain-level fixed costs (HQ, logistics) allocated equally across stores; constant in the short run, only relevant at entry/expansion.

### D.2 The quality-cost relationship — the substantive question you asked

Your instinct is right: in standard vertical-IO models (Mussa–Rosen, Shaked–Sutton), high quality unambiguously costs more. Discount retail breaks this monotonicity in a specific and well-documented way. **The right map is non-monotone in quality but monotone within each "operating model" cluster.**

Decompose the chain's cost into:
- **Assortment cost** (private labels, simpler SKUs, fewer perishables): D < S < B.
- **Operations cost** (store size, staff per m², freshness of perishables): D < S < B.
- **Sourcing premium** (organic certification, branded suppliers): D ≈ S < B.

Each of these *is* monotone in quality. The reason discounters look "low cost AND low quality" is that they sit at the bottom of all three; bio chains sit at the top of all three. Therefore, a single chain-type marginal cost $c_\theta$ that is monotone in $\theta$ is economically defensible. **There is no "discount paradox" to model.**

The empirical Berlin numbers from BKartA (LEH 2014) and HDE Zahlenspiegel — discounters ~18–22% gross margin, full-line ~24–28%, bio ~28–35% — are consistent with $c_D < c_S < c_B$ when combined with $p_D < p_S < p_B$. So you calibrate cost from observed margins and prices, not from any assumption about the quality-cost slope.

What you *should not* do: try to make $c_\theta$ depend on store effort. Effort cost is separate, store-level, and does not interact with chain type in the baseline. Conflating them would make calibration impossible.

### D.3 Chain quality — what gets parameterised

Three discrete intercepts $(q_D, q_S, q_B)$ with $q_D = 0$ as normalisation. Two free numbers: $q_S, q_B$. These are *not* directly observable. They are pinned by matching observed market shares (or chain footprints, as a second-best) given calibrated $(\alpha_L, \alpha_H, c_\theta, t, \mu)$. This is a moment-matching exercise, §J.

### D.4 Store-level effort — the AI experiment hook

Effort $e_{jt}$ enters demand (utility $+\beta e$) and cost ($+\kappa_0 e^2 / 2$). It is **not** chain-specific in either, by your design. The store agent therefore faces a within-period trade-off between attracting traffic and incurring labour cost. This is exactly the kind of choice you want the RL agent to learn, and exactly the kind of choice the CEO's envelope $[\bar e_c - \delta^e_c, \bar e_c + \delta^e_c]$ should constrain — because a too-low envelope eats market share, a too-high envelope kills margin.

The static (one-shot, full-information) interior optimum given price $p_{jt}$ and rivals' effort is $e^*_{jt} = \beta q_{jt} / \kappa_0$ — proportional to the store's expected demand and to the consumer's WTP for effort, inversely proportional to the marginal cost of effort. This gives you a useful sanity reference point for the RL converged effort levels.

### D.5 Profit objects

- **Store $j$ per-period profit:** $\pi_{jt} = (p_{jt} - c_{\theta_{c(j)}}) q_{jt} - \tfrac12 \kappa_0 e_{jt}^2 - R_{\ell_j}$.
- **Chain $c$ per-period profit:** $\Pi_{ct} = \sum_{j \in \mathcal{J}_c} \pi_{jt}$.
- **Chain $c$ discounted strategic value (from CEO's perspective at epoch $\tau$):** $V_{c\tau}(\sigma_c, \boldsymbol{\sigma}_{-c}) = \mathbb{E}\!\left[ \sum_{t = \tau}^{\infty} \delta^{t-\tau} \Pi_{ct} \,\Big|\, \sigma_c, \boldsymbol{\sigma}_{-c}, \text{store policies} \right]$.

---

## E. Entry and market structure

### E.1 Baseline: two-stage with single entrant

**Stage 0 (data).** The set of incumbent chains, their types, and their store locations are taken from OSM + Wikidata QIDs (Edeka, Rewe, Aldi Nord, Lidl, Penny, Netto MD, Kaufland, Bio Company, Norma). Inside the inner Ring, expect roughly 100–150 stores across these 9 chains.

**Stage 1 (entry, $t = 0$).** A single entrant chooses
$$(\theta_e, \ell_e) \in \{D, S, B\} \times \mathcal{L}^{\mathrm{commercial}},$$
where $\mathcal{L}^{\mathrm{commercial}}$ is the set of commercially-zoned candidate sites (FIS-Broker zoning layer, ~250 m grid). On entry, the entrant pays a sunk cost $F^{\mathrm{entry}}_{\theta_e}$ that depends on chain type (bio is more expensive — fit-out, certification). The entrant's first store is at $\ell_e$.

**Stage 2 (repeated play, $t = 1, 2, \dots, T$).** The CEO-LLM/RL-store machinery of §B.2 runs.

### E.2 Extension: dynamic store opening (specified, not run)

Each chain's CEO can, at each epoch $\tau$, evaluate one candidate new-store action $\mathrm{open}(\ell)$ for $\ell \in \mathcal{L}^{\mathrm{commercial}}$. The action is feasible if cumulated chain earnings since the last opening exceed $F^{\mathrm{entry}}_{\theta_c} \cdot (1 + \rho)$ (a self-financing constraint with safety margin $\rho$). The CEO's value function in §D.5 then has an additional term over the option to open. Implementation: small discrete action space at the CEO layer, augment the LLM's action set with `{open: ℓ, no_open}`. This extension nests the baseline ($\rho = \infty$ or option disabled).

### E.3 Why one entrant, not many

With a single entrant the entry stage is a one-shot decision with no equilibrium-of-many-entrants problem (the kind that makes Seim 2006 hard). It also lets you cleanly compare entrant-LLM behaviour across model versions. Multi-entrant entry is a clean later extension once the rest of the model behaves.

---

## F. Spatial component and transport costs

### F.1 Grid

Recommended **dual-grid** structure:
- **Demand grid $\mathcal{I}$:** 100 m INSPIRE cells (Zensus 2022 native), restricted to the inner-Ringbahn polygon (S41/S42 ring from FIS-Broker `s_lor_plr_2021` aggregated up). Roughly 8,000–9,000 cells.
- **Site grid $\mathcal{L}$:** 250 m cells (every 2nd Zensus cell along each axis, or a custom commercial-zoned set), filtered to where land-use allows retail. Roughly 1,000–1,500 candidate sites.

Computation is not the binding constraint at 100 m: distance matrix $|\mathcal{I}| \times |\mathcal{J}| \approx 9{,}000 \times 150 = 1.35 \times 10^6$ entries, ~10 MB float32, milliseconds to evaluate per period. Precompute and cache.

### F.2 Transport cost

Quadratic in distance, parameter $t$ in $€/\mathrm{km}^2$:
$$\mathrm{TC}^{\mathrm{trans}}_{ij} = t \cdot d_{ij}^2.$$

**Distance metric.** OSM-network distance via OSMnx + a routing back-end (OSRM) for the baseline; cache once. Euclidean as robustness. Quadratic (not linear) is non-negotiable for equilibrium existence.

**Calibration anchor.** $t \approx 0.5\ €/\mathrm{km}$ at the linear scale is the standard urban-grocery anchor; under quadratic with mean trip distance $\bar d \approx 0.7\,\mathrm{km}$, $t \approx 0.7\ €/\mathrm{km}^2$ gives a comparable disutility per trip. Treat as a tuned parameter; document the anchor.

### F.3 The grid problem, honestly

Yes, you do not have within-cell household locations. Three things fix this without pretending the problem is gone:

1. **Use the centroid for inter-cell distances** ($d_{ij}$ when consumer cell $i \neq$ store cell). Bias is bounded by half the cell diagonal ($\sim 70$ m at 100 m resolution; $\sim 350$ m at 500 m).
2. **Use the analytic intra-cell expectation $d^{\mathrm{intra}}$ for same-cell pairs** (§C.5).
3. **Stratify the cell into multiple "representative consumers"** if you suspect the within-cell variance matters near a competitive boundary. For 100 m cells in inner Berlin this is unnecessary.

What you *should not* do is randomly sample within-cell positions each period; this introduces simulation noise without economic content.

---

## G. Prime-location demand component

### G.1 Recommendation: static prime-location index, no time dynamics

A clean compact device: every cell $i$ gets a non-negative *prime-location index* $\phi_i$ that boosts the effective demand mass without changing any consumer's WTP. In §C.2 it appears as $(\omega_i + \lambda \phi_i)$.

**Construction of $\phi_i$.** Three additive components, normalised:
$$\phi_i = w_1\, \mathbb{1}[\text{station within } 200\text{m of } i] + w_2 \cdot (\text{employment density}_i / \bar{\mathrm{emp}}) + w_3\, \mathbb{1}[i \in \mathrm{CBD}],$$
with $w_1 + w_2 + w_3 = 1$ as a normalisation and $\lambda$ as the only free conversion parameter. CBD cells in inner-Ringbahn: Mitte (especially around Friedrichstraße, Alexanderplatz), Hauptbahnhof, Zoologischer Garten — defined as a polygon you draw once.

**Why this beats a time-of-day model.** A day/night/weekend cycle would require simulating temporally distinct demand systems; you would calibrate three sets of parameters from data you do not have. The static shifter captures the same first-order effect — *some cells generate more demand than their residential count implies* — at a tiny fraction of the modelling cost. The interpretive loss is small for a Berlin grocery study (intra-week cycles are second-order for grocery vs. e.g., cafés or fast food).

**Calibration of $\lambda$.** Choose so that total $\lambda \sum_i \phi_i$ equals an estimate of Berlin's daily transient + worker grocery demand, e.g., 10–15% of residential demand. Treat as a sensitivity parameter.

**Treat transients homogeneously.** Do not introduce a third consumer type for them; add the prime mass to the existing two-type weighting using the same $\pi_{hi}$. Fewer parameters, same first-order effect.

### G.2 What you give up

This formulation cannot answer "do prices fall during weekday lunch hours?" It can answer "do supermarkets near transit hubs charge more?" — which is the spatial-IO question you actually care about. The trade-off is good.

---

## H. Rent and cost structure

### H.1 Where rent enters

$R_{\ell_j}$ enters store profit additively (§D.5). It does not affect the within-period $(p, e)$ best response because it is sunk per period. It matters for:

1. **Entry/expansion decisions** (§E): the entrant compares discounted profit at $\ell$ against $R_\ell + F^{\mathrm{entry}}_{\theta_e}$.
2. **Selection effects** in observed incumbent locations (high-rent locations stay only when expected per-period demand justifies them — relevant for calibration sanity checks).
3. **The CEO's envelope choice**: a chain whose stores are in expensive locations may set higher $\bar p_c$ to compensate.

### H.2 Data — best resolution and source

In order of preference for Berlin commercial rent:

1. **Bodenrichtwerte (BORIS-Berlin / FIS-Broker `s_wfs_brw`).** Annual land-value zones, dl-de/by-2.0. Convert to commercial rent via standard cap-rate transformation: $R^{\mathrm{annual}}_\ell \approx \mathrm{BRW}_\ell \cdot A_{\mathrm{store}} \cdot r_{\mathrm{cap}}$, with $A_{\mathrm{store}} \approx 600\,\mathrm{m}^2$ for typical inner-Berlin LEH and $r_{\mathrm{cap}} \approx 0.06$. Then divide by the period count to match simulation cadence.
2. **IVD or JLL Berlin commercial rent reports.** Paid; coarser (~Bezirk level); use for calibration sanity check only.
3. **Mietspiegel residential rent.** Spatially correlated with commercial rent but biased; use as last-resort fallback.

### H.3 Spatial resolution — what to ask for

You want rent at the *site grid* $\mathcal{L}$ (250 m), not at the demand grid (100 m). Aggregate the Bodenrichtwerte zones to site-grid cells by spatial join (mean BRW within the cell). Asking for "rent at $\mathcal{L}$" is the right granularity — finer is overkill (BRW zones themselves are coarser than 100 m), coarser misses inner-Berlin rent gradients that *are* economically meaningful (Mitte vs. Wedding).

### H.4 In the baseline run, $R_\ell$ can be a placeholder

Until you wire in BRW data, set $R_\ell$ as a fixed function of distance to the inner-Ring centroid (say, the Berlin Cathedral): $R_\ell = R_0 \cdot \exp(-\eta \|\ell - \ell^*\|)$. This generates the correct centre–periphery gradient and lets you test the full model before calibration. Document the placeholder; replace cleanly.

---

## I. Equilibrium concept and decision rules

This is layered to match the agent architecture.

### I.1 Within-period market clearing

Given $(\mathbf{p}_t, \mathbf{e}_t)$, demand is determined by §C.2. No equilibrium concept is needed at this layer — it is a deterministic mapping.

### I.2 Tactical layer (RL store agents, fast timescale)

State for store $j$ of chain $c$ at period $t$:
$$S^{\mathrm{store}}_{jt} = \big(\,p_{j,t-1},\ e_{j,t-1},\ \tilde\pi_{j,t-1},\ \{p_{k,t-1}\}_{k \in \mathcal{N}(j)},\ \{e_{k,t-1}\}_{k \in \mathcal{N}(j)},\ \sigma_c \,\big),$$
where $\mathcal{N}(j)$ is the set of stores within radius $R$ of $\ell_j$ (suggest $R = 1\,\mathrm{km}$), and $\tilde\pi_{j,t-1}$ is a noisy realisation of the store's previous profit (add Gaussian noise with stdev $\sigma_\pi$ to match your "noisy past performance" spec). No consumer-side variables.

Action: $(p_{jt}, e_{jt})$ on a discretised grid, restricted to the envelope.

The store's policy is a function $\rho_j: S^{\mathrm{store}} \to \mathcal{A}^{\mathrm{store}}$ trained by tabular Q-learning (Calvano-style) with $\varepsilon$-greedy exploration. Per-store table size with 5 own-price × 3 effort × 5 envelope-price-bucket × 3 nearest-rival-price × 2 nearest-rival-effort levels: ~450 states × 15 actions = ~7,000 cells. Trivial. With $\sim 150$ stores, total memory ~10 MB.

The convergence object is a **Markov perfect equilibrium of the store-layer game given the envelope vector** — what Calvano calls "asymptotic policy". It need not be Nash in the structural sense; it is what RL converges to under independent learning.

### I.3 Strategic layer (LLM-CEO, slow timescale)

State for chain $c$ at epoch $\tau$:
$$S^{\mathrm{CEO}}_{c\tau} = \big(\,\text{own chain summary, all rivals' summaries, consumer-side LOR statistics, period}\,\tau\,\big).$$

Concretely: own chain mean price, mean effort, total demand, total profit over the last $T_{\mathrm{CEO}}$ periods; rivals' published prices and store locations; aggregated LOR-level demographic distribution with the corresponding population mass. (A precise JSON schema is given in §L.)

Action: $\sigma_c = (\bar p_c, \delta^p_c, \bar e_c, \delta^e_c) \in \mathbb{R}^4_+$, discretised to e.g. 5 × 3 × 5 × 3 = 225 strategy cells if you want a tractable grid, or continuous if the LLM outputs floats.

The LLM's "decision rule" is its policy under prompt + structured-output constraint (Pydantic schema). It is *not* an equilibrium-computation device. Therefore the strategic layer has no formal equilibrium concept beyond *behavioural strategies*. You compare these behavioural strategies to two fully-rational benchmarks computed analytically:
- **Static Bertrand–Nash given current locations and types**, with all envelopes infinitely wide and effort at the best-response level $e^* = \beta q_j / \kappa_0$. Computed by iterating $p_j - c_{\theta_{c(j)}} = \mu / (1 - s_j)$ until convergence. Logit demand makes this a contraction (Anderson–de Palma–Thisse 1992).
- **Joint-profit monopoly** given the same locations, types, effort levels.

The difference between the realised path and these benchmarks is your $\Delta$ measure (Calvano).

### I.4 Entry stage

The entrant LLM solves a one-shot best-response over $(\theta_e, \ell_e) \in \{D, S, B\} \times \mathcal{L}^{\mathrm{commercial}}$ given anticipated post-entry play. There is no formal Nash here either — the LLM's action is its empirical decision, and you compare it to the analytic best-response computed against either Bertrand–Nash or Calvano-baseline post-entry play.

### I.5 What the equilibrium machinery delivers

You get four convergence objects per session:
- An entrant choice $(\hat\theta_e, \hat\ell_e)$.
- A converged path of strategy envelopes $\{\hat\sigma_c\}$.
- A converged store policy profile $\{\hat\rho_j\}$.
- A profit/price/effort time series.

And two analytic benchmarks (Bertrand–Nash and joint-monopoly) given each session's realised location/type configuration. From these you compute $\Delta$, price dispersion, and any other reduced-form statistic.

---

## J. Calibration targets and required real-world data

### J.1 Hierarchy of data needs

**Tier 1 — must have for the baseline to mean anything.**
- *OSM chain coordinates* in inner-Ringbahn (OSMnx, free, ODbL).
- *Zensus 2022 100 m population grid* (DESTATIS, dl-de/by-2.0).
- *Berlin LOR boundaries + social-status index* (FIS-Broker `s_lor_plr_2021` and Sozialatlas — CC-BY-3.0-DE). The social-status index is the single most important non-trivial input.
- *Bundeskartellamt LEH 2014 sector inquiry* + *HDE Zahlenspiegel* — for chain-type margin anchors.

**Tier 2 — needed for full feature set.**
- *VBB GTFS* for transit stops (CC-BY 3.0 DE).
- *Bodenrichtwerte* for rent (FIS-Broker `s_wfs_brw`, dl-de/by-2.0).
- *FIS-Broker zoning layer* for commercial-zone restriction in $\mathcal{L}^{\mathrm{commercial}}$.

**Tier 3 — nice to have, not required.**
- Employment / workplace density (Bundesagentur für Arbeit at PLZ; difficult at LOR).
- Weekly footfall counters from Berlin retail BIDs (ad-hoc availability).

### J.2 What gets calibrated to what moment

| Target moment | Estimated parameter |
|---|---|
| Median chain-type gross margin from BKartA | $c_D, c_S, c_B$ given prices |
| Observed chain-type market shares within Berlin | $q_S, q_B$ (with $q_D = 0$) |
| Income/status gradient in chain-type market shares across LORs | $\alpha_L, \alpha_H$ |
| Average grocery trip distance (~700 m urban) | $t$ |
| Demand elasticity of −2 to −4 (literature for grocery) | $\mu$ |
| Commercial rent gradient (BRW transformation) | $R_\ell$ (no free parameter — direct from data) |
| Effort cost — no Berlin data | $\kappa_0$ free; sensitivity analysis |
| Effort utility — no Berlin data | $\beta$ free; sensitivity analysis |

Calibration is *moment-matching*, not full structural estimation. You do not need a likelihood — you need parameter values that reproduce a small set of stylised facts. This is honest for a thesis-scale project.

---

## K. Parameters ordered by importance for calibration

**Critical (model breaks without these in the right ballpark):**
1. $t$ — transport cost.
2. $\mu$ — logit scale / horizontal differentiation.
3. $(c_D, c_S, c_B)$ — chain-type marginal costs.
4. $(q_S, q_B)$ — quality intercepts.
5. $(\alpha_L, \alpha_H)$ — type-specific WTP for quality.
6. $\omega_i$ — Zensus population (already pinned).
7. $\pi_{hi}$ — type weights from LOR index.
8. Incumbent locations $\{\ell_j\}$ — already pinned.

**Important (matters for realism, can be approximated initially):**
9. $\beta$ — effort utility.
10. $\kappa_0$ — effort cost.
11. $\lambda, \phi_i$ — prime-location component.
12. $R_\ell$ — rent.
13. $a_0$ — outside option.

**Secondary (sensitivity, free choice):**
14. $T_{\mathrm{CEO}}$, $\delta$ (discount factor), envelope discretisation.
15. $F^{\mathrm{entry}}_\theta$ — entrant sunk cost.
16. RL hyperparameters ($\alpha_{\mathrm{lr}}, \beta_{\mathrm{exp}}$) — set per Calvano grid.

**Free / pick by hand for first run:**
17. Noise stdev $\sigma_\pi$ on store-level profit signal.
18. Local-rivals radius $R$.

---

## L. Closed-form expressions and analytical objects for simulation

### L.1 Logit choice probability (per consumer-cell-store triple)

$$\Pr(j \mid h, i) = \frac{e^{V_{hij}/\mu}}{e^{a_0/\mu} + \sum_k e^{V_{hik}/\mu}}, \qquad V_{hij} = \alpha_h q_{\theta_{c(j)}} + \beta e_{jt} - p_{jt} - t d_{ij}^2.$$

Vectorised: $V$ is a 3-D tensor of shape (types, cells, stores); a single softmax along the store axis yields the probability tensor.

### L.2 Aggregate demand and revenue

$$q_{jt} = \sum_{i,h}(\omega_i + \lambda \phi_i)\, \pi_{hi}\, \Pr(j \mid h, i; \mathbf{p}_t, \mathbf{e}_t).$$

### L.3 Bertrand–Nash logit FOC (the iteration target)

For each store $j$ of chain $c$ in the *static, full-info* Bertrand benchmark:
$$p^N_j - c_{\theta_{c(j)}} = \frac{\mu}{1 - s_j(\mathbf{p}^N, \mathbf{e}^N)}, \qquad e^N_j = \frac{\beta q_j(\mathbf{p}^N, \mathbf{e}^N)}{\kappa_0}.$$

These are coupled; iterate jointly until both converge. Anderson–de Palma–Thisse (1992) guarantee uniqueness in $\mathbf{p}$ given $\mathbf{e}$; the effort FOC is a contraction in $\mathbf{e}$ given $\mathbf{p}$. Joint convergence in 50–500 iterations.

### L.4 Joint-profit monopoly

Maximise $\sum_j (p_j - c_{\theta_{c(j)}}) q_j - \tfrac12 \kappa_0 e_j^2 - R_{\ell_j}$ over all $(p_j, e_j)$ jointly. Use `scipy.optimize.minimize` with the gradient (closed-form via the logit derivative — this is fast).

### L.5 Static effort interior optimum (sanity reference)

$$e^*_j = \frac{\beta q_j(\mathbf{p}, \mathbf{e})}{\kappa_0}.$$

### L.6 Consumer surplus (logit inclusive value)

For policy/welfare analysis:
$$\mathrm{CS}_t = \sum_i (\omega_i + \lambda \phi_i) \sum_h \pi_{hi} \cdot \mu \cdot \log\!\Big( e^{a_0/\mu} + \sum_j e^{V_{hij}/\mu} \Big).$$

This is the right object for any welfare claim — Calvano-style $\Delta$ measures only producer-side gain.

### L.7 LLM-CEO state JSON (the prompt input schema)

```json
{
  "chain_id": "Edeka",
  "chain_type": "S",
  "epoch": 100,
  "T_ceo": 50,
  "own_chain": {
    "n_stores": 13,
    "mean_price_last_T": 1.42,
    "mean_effort_last_T": 0.6,
    "total_demand_last_T": 84200,
    "total_profit_last_T": 9850.0,
    "store_locations_m": [[3900, 8100], [4600, 7900], ...]
  },
  "rivals": [
    {"id": "Aldi_Nord", "type": "D", "n_stores": 18, "last_published_price": 1.05, "store_locations_m": [...]},
    ...
  ],
  "consumers": {
    "total_population": 1_080_000,
    "high_status_share": 0.41,
    "lor_demographics": [{"lor_id": "010101", "pop": 7820, "S_index": 0.62}, ...]
  },
  "constraints": {
    "price_must_cover_marginal_cost": true,
    "envelope_min_width_p": 0.05,
    "envelope_min_width_e": 0.1
  },
  "history": [{"tau": 50, "sigma": [1.40, 0.10, 0.55, 0.15], "profit_realized": 9120.0}, ...]
}
```

The CEO returns `{"p_bar": 1.45, "delta_p": 0.10, "e_bar": 0.65, "delta_e": 0.20}` validated by Pydantic.

---

## M. Analytical vs. numerical

### M.1 Analytical

- Demand probabilities (closed-form softmax).
- Bertrand–Nash and joint-monopoly benchmarks (FOC + iteration).
- Static effort interior optimum.
- Consumer surplus (inclusive-value formula).
- LOR-to-cell type weight mapping (closed-form).
- Within-cell distance approximation.

### M.2 Numerical

- The repeated game itself (RL training loop).
- The CEO's strategy choice (LLM call).
- Distance-matrix computation (one-time precompute).
- Joint-monopoly maximiser (small `scipy` call).
- IRF benchmark from converged policies (replay).

### M.3 What is *not* computed at all

- A full SPNE of the multi-stage game with multi-store CEOs and dynamic store opening. The problem is too high-dimensional. The point of the simulation is precisely to replace this with behavioural play and measure deviation from static benchmarks.

---

## N. Recommended baseline vs. extensions

### N.1 Baseline run (what you build first)

- 8–9 incumbent chains, types fixed by chain identity, ~100–150 stores from OSM.
- 1 entrant; entry stage chooses $(\theta_e, \ell_e)$ once at $t = 0$.
- Repeated game $T \in [500, 1000]$ periods, $T_{\mathrm{CEO}} = 50$.
- Per-store Q-learning (price, effort) inside CEO envelope.
- LLM CEO once every $T_{\mathrm{CEO}}$.
- Demand grid 100 m, site grid 250 m, network distance.
- Two consumer types via LOR social-status index.
- Static prime-location index $\phi_i$.
- Rent $R_\ell$ as exponential placeholder if BRW not yet wired.
- Discrete chain types $\{D, S, B\}$ with three quality intercepts.
- Sessions $\geq 30$ seeds per (model, treatment) cell.

### N.2 Extensions (specified, slot in cleanly)

1. **Dynamic store opening** (§E.2). High research interest.
2. **Hybrid quality:** discrete type $\theta$ + within-type continuous shifter $\delta_q$. Lets the entrant pick "premium discounter" or "budget bio".
3. **Mixed-logit consumer types** (continuous $\alpha$). For when LOR two-type is challenged.
4. **Multi-entrant entry stage** (Seim 2006-style).
5. **Time-varying prime-location** (weekday vs weekend) — only if the data justifies it.
6. **Multimodal LLM input** (rendered map). For the CEO's geographic reasoning ablation.
7. **Bilateral contracts / wholesale costs** that vary by city block — almost certainly out of scope for an M.Sc./PhD seminar.

### N.3 What I deliberately keep out of the baseline

- Search frictions (no Berlin data; large literature; high risk).
- Stochastic demand (Calvano §IV.D — robustness later).
- Promotional cycles / Hi-Lo pricing (separate dynamic problem).
- Competitor-store closure (the inverse of opening — symmetric extension).

---

## O. Key simplifications and assumptions

1. **Chain type is observable and fixed.** Real-world repositioning (e.g., Penny refresh) is ignored.
2. **Consumer effort valuation $\beta$ is homogeneous.** "A clean store is a clean store" — your spec.
3. **Effort cost $\kappa_0$ is homogeneous and chain-invariant.** Hiring cleaners costs the same in Mitte as in Wedding (a mild simplification — wages probably do vary by district, but the spread is small).
4. **The outside option is a homogeneous logit shock.** No structural model of "shop online" or "go to store outside the Ring".
5. **Population $\omega_i$ is static.** No demographic dynamics over the simulation horizon.
6. **All stores in a chain share the same $c_\theta$.** No store-level cost heterogeneity within chain (e.g., centre-city store with smaller back-of-house has higher unit cost). Adding store-level $c_j$ is trivial structurally; calibration data is the bottleneck.
7. **CEO observes consumer-side state perfectly.** A simplification — real CEOs use research firms and panel data with measurement error.
8. **Store agents have memory-1 RL state.** Calvano-style; richer memory is a known extension and a known computational pain point.
9. **Two consumer types.** Mixed-logit is an extension.
10. **Quadratic transport cost.** Necessary for equilibrium existence; consistent with Berlin practice.

---

## P. Risks, pitfalls, and what to avoid

### P.1 Modelling risks

- **Identifying $\alpha_L, \alpha_H$ from observed shares** is under-identified without an exclusion restriction. The clean approach: fix $(\alpha_L, \alpha_H)$ at literature-anchored values (e.g., Hausman 1996 grocery elasticities by income) and treat as sensitivity parameters. Do not over-claim.
- **Effort identifiability** is genuinely weak — there is no Berlin data on store cleanliness or service. Treat $\beta, \kappa_0$ as policy parameters and report sensitivity.
- **Quality intercepts $q_S, q_B$** are calibration-dependent on the share moments you choose. Document your moment-matching criterion.
- **Logit smoothing $\mu$** controls how "near-Voronoi" demand is. Larralde–Stehlé–Jensen (2009) suggest $\mu$ between 0.1 and 0.3 of $t \bar d$. Calibrate to match observed price elasticity.

### P.2 Simulation risks

- **Independent multi-agent Q-learning may not converge** in the presence of an LLM-driven CEO whose envelope drifts. Mitigations: long training, slow LLM cadence ($T_{\mathrm{CEO}} \geq 50$), envelope-width minimum to keep the action set non-degenerate.
- **LLM CEO inconsistency across calls** at the same state. Use `temperature = 0` and pinned model snapshots; expect drift anyway, especially with Anthropic where seed is unsupported (April 2026). Log every call.
- **Overfitting incumbent locations.** With ~150 fixed stores, the spatial structure is largely determined; the entrant's behaviour and CEO envelopes are the real degrees of freedom. Make sure your figures show *those*, not the (already-pinned) spatial Voronoi.
- **The grid problem leaking.** If you ever animate a within-cell consumer trajectory, you will be inventing data.

### P.3 Interpretation risks

- **$\Delta$ confounds collusion with price-quality bundling** when chain types differ. Compute $\Delta$ separately per chain type, not just pooled.
- **Effort is unobserved in real data.** Any claim about "the store agent learned to clean more" cannot be empirically validated. Frame as a thought experiment.
- **The CEO–store delegation result is a model-internal finding**, not a general claim about retail organisations. Be careful in the thesis defence.
- **One-entrant findings do not generalise to many-entrant equilibria.** Caveat any "where would the entrant locate" claim.

### P.4 Things to avoid outright

- **Do not** let the LLM CEO see noise-free profit history — it makes the comparison to the RL store agent's noisy state unfair.
- **Do not** randomise within-cell consumer positions per period.
- **Do not** treat the LOR social-status index as a continuous WTP without the type-mix mapping in §C.3 — direct continuous WTP from a composite index has no economic interpretation.
- **Do not** interpret converged effort levels as "real cleanliness". They are parameter-dependent objects.
- **Do not** publish or distribute the simulation outputs without including the OSM, Zensus, Bodenrichtwerte, and VBB licence attributions.

---

## Compact final recommendation

Build the **baseline as specified in §N.1**: dual 100 m / 250 m grid restricted to inner-Ringbahn Berlin, two consumer types from LOR social-status indices, three discrete chain types, quadratic transport, LLM CEO every 50 periods setting $(\bar p, \delta^p, \bar e, \delta^e)$, per-store Q-learning inside the envelope, single entrant choosing $(\theta, \ell)$ at $t = 0$, static prime-location shifter, BRW-derived rents (exponential placeholder until data is wired), Calvano $\Delta$ as headline outcome computed against analytic Bertrand–Nash and joint-monopoly benchmarks. Defer dynamic store opening, mixed-logit, and multimodal LLM input to phase 2.

The single decision that most affects whether this works in practice is **the envelope discretisation and width**. Start with $\bar p \in \{5\text{ levels}\}$, $\delta^p \in \{0.05, 0.10, 0.15\}$, similar for effort. If the RL store agents converge tightly inside the envelope, widen $\delta^p$. If they bounce on the envelope boundaries, widen $\delta^p$ first; if they still bounce, the problem is in the local-information state, not the envelope.

The single result that will most validate the model is **reproducing the Calvano $\Delta \approx 0.7$–$0.85$ in the degenerate single-store-per-chain, no-CEO, no-effort, no-spatial-term limit**. Get this passing before turning on any extension. Everything else is calibration-and-experiment work on top.